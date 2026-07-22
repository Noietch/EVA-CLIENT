"""RL workspace activation, telemetry, and critic lifecycle."""

from __future__ import annotations

import copy
import time

import numpy as np

import critic_client  # noqa: F401
from core.app.state import RuntimeState
from core.config import ConfigDict
from core.registry import CRITIC_REGISTRY
from critic_client.base import CriticBuildContext, CriticConnectionError
from critic_client.runner import CriticRunner


def build_rl_active_config(config: ConfigDict, policy_slot: int) -> ConfigDict:
    """Build the selected policy deployment with the fixed RL recording contract."""
    rl_cfg = config.rl
    if rl_cfg is None:
        raise ValueError("RL workspace is not configured")
    if policy_slot < 0 or policy_slot >= len(rl_cfg.policies):
        raise ValueError(f"RL policy slot is out of range: {policy_slot}")
    active = copy.deepcopy(rl_cfg.policies[policy_slot].config)
    active.rl = copy.deepcopy(rl_cfg)
    active.rl_cfg = active.rl
    active.eval = None
    active.eval_cfg = None
    active.rollout.storage = copy.deepcopy(rl_cfg.data.storage)
    active.rollout.storage.enabled = True
    active.rollout.intervention = copy.deepcopy(rl_cfg.intervention)
    return active


def connect_rl_critic(config: ConfigDict, runtime: RuntimeState, critic_slot: int) -> bool:
    """Connect the selected Critic and start its non-blocking request runner."""
    close_rl_critic(runtime)
    rl_cfg = config.rl
    if rl_cfg is None:
        runtime.rl_critic_error = "RL workspace is not configured"
        return False
    if critic_slot < 0 or critic_slot >= len(rl_cfg.critics):
        runtime.rl_critic_error = f"RL critic slot is out of range: {critic_slot}"
        return False
    critic_cfg = rl_cfg.critics[critic_slot]
    try:
        client = CRITIC_REGISTRY.build(
            str(critic_cfg.type),
            critic_cfg,
            CriticBuildContext(retry_until_connected=False, max_retries=0),
        )
    except CriticConnectionError as error:
        runtime.rl_critic_error = str(error.__cause__ or error)
        return False
    runtime.rl_critic_runner = CriticRunner(client)
    metadata = runtime.rl_critic_runner.status()["metadata"]
    runtime.rl_critic_action_horizon = (
        int(metadata["action_horizon"]) if "action_horizon" in metadata else None
    )
    runtime.rl_critic_error = ""
    if (
        runtime.rl_pending_critic_observation is not None
        and runtime.rl_pending_critic_action is not None
    ):
        runtime.rl_critic_runner.submit(
            runtime.rl_pending_critic_observation,
            _align_critic_actions(runtime, runtime.rl_pending_critic_action),
            runtime.rl_pending_critic_timestamp or time.time(),
            "policy",
        )
    return True


def close_rl_critic(runtime: RuntimeState) -> None:
    """Close the active Critic runner and clear its connection state."""
    runner = runtime.rl_critic_runner
    runtime.rl_critic_runner = None
    runtime.rl_critic_action_horizon = None
    if runner is not None:
        runner.close()


def close_rl_workspace(runtime: RuntimeState) -> None:
    """Release RL-only model and replay resources when leaving the workspace."""
    close_rl_critic(runtime)
    with runtime.rl_replay_lock:
        runtime.rl_replay_generation += 1
        for source in runtime.rl_replay_sources:
            source.close()
        runtime.rl_replay_sources = []
        runtime.rl_replay_source = None
        runtime.rl_replay_dataset_dir = ""
        runtime.rl_replay_episode_id = None
        runtime.rl_replay_timestamps = []
    runtime.rl_live_samples = []
    runtime.rl_pending_critic_observation = None
    runtime.rl_pending_critic_action = None
    runtime.rl_pending_critic_timestamp = None
    runtime.rl_active = False
    runtime.rl_selected_policy_slot = None
    runtime.rl_selected_critic_slot = None
    runtime.rl_critic_error = ""


def reset_rl_series(runtime: RuntimeState) -> None:
    """Clear rollout telemetry and begin a fresh Critic curve."""
    runtime.rl_live_samples = []
    runtime.rl_pending_critic_observation = None
    runtime.rl_pending_critic_action = None
    runtime.rl_pending_critic_timestamp = None
    if runtime.rl_critic_runner is not None:
        runtime.rl_critic_runner.reset_series()


def record_rl_sample(
    runtime: RuntimeState,
    state: np.ndarray | None,
    action: np.ndarray | None,
    source: str,
    timestamp: float | None = None,
    segment_index: int = -1,
) -> None:
    """Append one executed rollout or intervention sample for live charts."""
    if not runtime.rl_active or state is None or action is None:
        return
    runtime.rl_live_samples.append(
        (
            float(time.time() if timestamp is None else timestamp),
            np.asarray(state, dtype=np.float32).copy(),
            np.asarray(action, dtype=np.float32).copy(),
            str(source),
            int(segment_index),
        )
    )


def build_rl_critic_observation(runtime: RuntimeState, observation: dict) -> dict:
    """Flatten a policy observation into the LeRobot keys consumed by the Critic.

    Args:
        runtime: Active runtime carrying the robot camera schema.
        observation: Nested policy observation with ``state`` and ``images`` fields.

    Returns:
        Flat mapping containing ``observations.state.qpos`` and
        ``observation.images.<camera>`` entries for Critic preprocessing.
    """
    raw = dict(observation)
    if "state" in observation:
        raw["observations.state.qpos"] = observation["state"]
    images = observation.get("images")
    if isinstance(images, dict):
        for camera in runtime.robot.observation_schema.cameras:
            image = None
            if camera.observation_key in images:
                image = images[camera.observation_key]
            elif camera.name in images:
                image = images[camera.name]
            if image is not None:
                raw[f"observation.images.{camera.observation_key}"] = image
        for key, image in images.items():
            if key.startswith("observation.images."):
                raw[key] = image
    return raw


def submit_rl_critic(
    runtime: RuntimeState,
    observation: dict,
    actions: np.ndarray,
    source: str,
    timestamp: float | None = None,
) -> None:
    """Submit one ready observation/action pair without blocking its producer."""
    if not runtime.rl_active:
        return
    critic_observation = build_rl_critic_observation(runtime, observation)
    critic_action = np.asarray(actions, dtype=np.float32).copy()
    if runtime.rl_critic_runner is not None:
        critic_action = _align_critic_actions(runtime, critic_action)
    critic_timestamp = time.time() if timestamp is None else timestamp
    runtime.rl_pending_critic_observation = critic_observation
    runtime.rl_pending_critic_action = critic_action
    runtime.rl_pending_critic_timestamp = critic_timestamp
    if runtime.rl_critic_runner is None:
        return
    runtime.rl_critic_runner.submit(
        critic_observation,
        critic_action,
        critic_timestamp,
        source,
    )


def _align_critic_actions(runtime: RuntimeState, actions: np.ndarray) -> np.ndarray:
    """Align an action sequence with the Critic server's declared horizon."""
    horizon = runtime.rl_critic_action_horizon
    if horizon is None:
        return actions
    if actions.ndim != 2:
        raise ValueError(f"Critic actions must be 2D, got shape={actions.shape}")
    if horizon <= 0:
        raise ValueError(f"Critic action horizon must be positive, got {horizon}")
    if actions.shape[0] == horizon:
        return actions
    if actions.shape[0] == 0:
        raise ValueError("Critic actions cannot be empty")
    if actions.shape[0] > horizon:
        return actions[:horizon]
    padding = np.repeat(actions[-1:, :], horizon - actions.shape[0], axis=0)
    return np.concatenate((actions, padding), axis=0)


def rl_live_series(runtime: RuntimeState, since: int, critic_since: int) -> dict:
    """Serialize incremental action/state/source and Critic samples."""
    start = max(0, int(since))
    samples = runtime.rl_live_samples[start:]
    critic = (
        runtime.rl_critic_runner.series(max(0, int(critic_since)))
        if runtime.rl_critic_runner is not None
        else {"n": 0, "timestamp": [], "value": [], "source": []}
    )
    return {
        "active": runtime.rl_active,
        "n": len(runtime.rl_live_samples),
        "timestamp": [sample[0] for sample in samples],
        "state": [sample[1].tolist() for sample in samples],
        "action": [sample[2].tolist() for sample in samples],
        "control_source": [sample[3] for sample in samples],
        "intervention": [sample[3] == "intervention" for sample in samples],
        "intervention_segment_index": [sample[4] for sample in samples],
        "critic": critic,
    }


__all__ = [
    "build_rl_active_config",
    "build_rl_critic_observation",
    "close_rl_critic",
    "close_rl_workspace",
    "connect_rl_critic",
    "record_rl_sample",
    "reset_rl_series",
    "rl_live_series",
    "submit_rl_critic",
]
