"""Policy connection, inference fetch, IK solving, and dataset replay sourcing."""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import numpy as np
from tqdm import tqdm

from core.app.handlers.imaging import normalize_action_chunk
from core.app.handlers.recording import (
    build_policy_observation,
)
from core.app.handlers.space import EEFPose, JointState
from core.app.handlers.utils import (
    _chunk_delta_norm,
    _format_diag_vector,
    _resolve_runtime_path,
    is_offline_transport,
    reset_ik_solver,
    reset_infer_strategy,
    reset_run_state,
    reset_session_progress,
)
from core.app.rl import submit_rl_critic
from core.app.state import (
    OutputTarget,
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
    format_task_label,
    set_status,
)
from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from core.utils.lerobot import LeRobotDatasetIO
from policy_client.base import PolicyBuildContext, PolicyConnectionError, PolicyRequestError
from transport.base import TransportBridge
from transport.dataset import DatasetTransport

logger = logging.getLogger(__name__)


def connect_policy(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    force_reconnect: bool = False,
    retry_until_connected: bool = False,
    max_retries: int = 0,
) -> bool:
    """Connect (or reconnect) the policy client and cache its metadata on the runtime.

    Builds the backend named by config.policy.type (e.g. "openpi" vs "openpi_rtc"
    select the plain vs RTC client); any replay action trajectory is threaded
    through. A connection failure is swallowed (recorded on
    runtime.last_policy_error) rather than raised.

    Args:
        force_reconnect: rebuild even when a policy is already connected.
        retry_until_connected: keep retrying inside the backend builder until it
            succeeds (used by eval bootstrap) instead of failing fast.
        max_retries: retry budget when retry_until_connected is set.

    Returns:
        True if a policy is connected on return, False on connection failure.
    """
    if runtime.policy is not None and not force_reconnect:
        logger.debug("Policy already connected")
        return True
    _ = session  # kept for a uniform connect/disconnect signature across callers

    policy_cfg = config.policy

    action_trajectory = runtime.replay_trajectory
    if action_trajectory is None and isinstance(runtime.transport, DatasetTransport):
        action_trajectory = runtime.transport.get_action_trajectory()
    ctx = PolicyBuildContext(
        action_dim=runtime.robot.total_action_dim,
        action_trajectory=action_trajectory,
        retry_until_connected=retry_until_connected,
        max_retries=max_retries,
    )

    backend = policy_cfg.type
    logger.info(
        "%s policy backend=%s %s:%s",
        "Reconnecting" if force_reconnect else "Connecting",
        backend,
        policy_cfg.host,
        policy_cfg.port,
    )
    try:
        policy = POLICY_REGISTRY.build(backend, policy_cfg, ctx)
    except PolicyConnectionError as error:
        runtime.policy = None
        runtime.policy_metadata = None
        runtime.last_policy_error = str(error.__cause__ or error)
        logger.info(
            "Policy backend %s unavailable at %s:%s",
            backend,
            policy_cfg.host,
            policy_cfg.port,
        )
        return False
    runtime.policy = policy
    runtime.policy_metadata = policy.metadata
    runtime.last_policy_error = ""
    logger.info("Policy connected (%s): %s", backend, policy.metadata)
    return True


def invalidate_policy_connection(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    error: Exception,
) -> None:
    """Drop the policy connection after a request error and reset session/strategy/IK.

    Records the error on runtime.last_policy_error so the UI can surface it; the next
    connect or setup re-establishes the connection.
    """
    runtime.policy = None
    runtime.policy_metadata = None
    runtime.last_policy_error = str(error.__cause__ or error)
    reset_run_state(config, runtime, session)
    logger.info("Policy request failed: %s. Run connect or setup again.", runtime.last_policy_error)


def disconnect_policy(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Drop the policy connection by operator request and clear cached setup state."""
    if runtime.policy is None:
        logger.info("Policy server is already disconnected")
        return
    runtime.policy = None
    runtime.policy_metadata = None
    runtime.last_policy_error = ""
    reset_run_state(config, runtime, session)
    logger.info("Disconnected policy server and cleared cached setup state")


def parse_policy_action_mode(raw_mode: object) -> JointState | EEFPose | None:
    """Parse a raw action-mode string into a Format instance; None if unrecognized."""
    if not isinstance(raw_mode, str):
        return None
    normalized = raw_mode.strip().lower()
    if normalized in {"joint", "qpos"}:
        return JointState()
    if normalized == "eef":
        return EEFPose()
    return None


def resolve_policy_action_mode(
    config: ConfigDict,
    runtime: RuntimeState,
    response: dict[str, object] | None = None,
) -> JointState | EEFPose:
    """Resolve the policy's action mode (joint vs. eef).

    Precedence: the per-inference response, then the cached policy metadata, then the
    configured default.
    """
    if response is not None:
        response_mode = parse_policy_action_mode(response.get("action_mode"))
        if response_mode is not None:
            return response_mode
    if isinstance(runtime.policy_metadata, dict):
        metadata_mode = parse_policy_action_mode(runtime.policy_metadata.get("action_mode"))
        if metadata_mode is not None:
            return metadata_mode
    return config.inference_cfg.action_space


def ensure_ik_solver(config: ConfigDict, runtime: RuntimeState) -> Any:
    """Return the cached IK solver, building it from the robot's kinematics on first use.

    Raises RuntimeError when EEF control is requested but the robot provides no
    kinematics. The solver is seeded from the robot's initial qpos and runs at the
    configured control dt.
    """
    if runtime.ik_solver is not None:
        return runtime.ik_solver
    solver_kwargs: dict[str, object] = {
        "dt": 1.0 / max(config.inference_cfg.publish_rate, 1),
        "initial_qpos_groups": runtime.robot.initial_qpos_by_group(),
    }
    if config.robot.eef_reference_frame in runtime.robot.supported_reference_frames:
        solver_kwargs["reference_frame"] = config.robot.eef_reference_frame
    solver = runtime.robot.build_kinematics(**solver_kwargs)
    if solver is None:
        raise RuntimeError("EEF mode requires an IK solver but the robot does not provide one")
    runtime.ik_solver = solver
    return solver


def normalize_policy_action_chunk(
    config: ConfigDict,
    runtime: RuntimeState,
    action_chunk: np.ndarray,
    response: dict[str, object] | None = None,
) -> np.ndarray:
    """Convert a policy action chunk into joint space, running IK when the mode is EEF.

    Joint-mode chunks pass through unchanged. EEF chunks are normalized to the canonical
    EEF layout and solved to qpos, seeded from the latest robot qpos.

    Args:
        action_chunk: [T, D] float32 chunk in the policy's native action mode.

    Returns:
        [T, total_action_dim] float32 joint-space chunk.
    """
    if not resolve_policy_action_mode(config, runtime, response=response).is_eef():
        return action_chunk
    canonical_eef_chunk = (
        config.inference_cfg.action_space.normalize_chunk_to_canonical(action_chunk)
    )
    ik_solver = ensure_ik_solver(config, runtime)
    seed_qpos = runtime.transport.get_latest_qpos()
    return ik_solver.solve_chunk(canonical_eef_chunk, seed_qpos=seed_qpos)


def _resolve_replay_action_mode(config: ConfigDict, runtime: RuntimeState) -> JointState | EEFPose:
    replay_mode = parse_policy_action_mode(runtime.replay_action_mode)
    if replay_mode is None:
        return config.inference_cfg.action_space
    if not replay_mode.is_eef():
        return replay_mode

    if isinstance(config.inference_cfg.action_space, EEFPose):
        return config.inference_cfg.action_space
    return EEFPose(
        n_arms=len(runtime.robot.arm_groups),
        rotation="quat",
        include_gripper=True,
    )


def _replay_qpos_at(runtime: RuntimeState, index: int) -> np.ndarray | None:
    replay = runtime.replay_source
    if replay is not None and hasattr(replay, "get_scene_qpos"):
        return np.asarray(replay.get_scene_qpos(index), dtype=np.float32)
    return runtime.transport.get_latest_qpos()


def _fill_missing_replay_eef_gripper(
    runtime: RuntimeState,
    fmt: EEFPose,
    action_chunk: np.ndarray,
    index: int,
) -> np.ndarray:
    source_per_arm_dim = 3 + fmt.rotation_dim()
    if not fmt.include_gripper or action_chunk.shape[1] != source_per_arm_dim * fmt.n_arms:
        return action_chunk

    qpos = _replay_qpos_at(runtime, index)
    gripper_indices = runtime.robot.gripper_indices
    if qpos is None or len(gripper_indices) < fmt.n_arms:
        raise ValueError(
            "EEF replay action is missing gripper values and no recorded qpos gripper is available"
        )
    gripper_values = np.asarray(
        [qpos[gripper_indices[arm_idx]] for arm_idx in range(fmt.n_arms)],
        dtype=np.float32,
    )

    rows = []
    for row in action_chunk:
        parts = []
        for arm_idx in range(fmt.n_arms):
            offset = arm_idx * source_per_arm_dim
            parts.append(row[offset : offset + source_per_arm_dim])
            parts.append(gripper_values[arm_idx : arm_idx + 1])
        rows.append(np.concatenate(parts, axis=0))
    return np.asarray(rows, dtype=np.float32)


def _replay_action_at(config: ConfigDict, runtime: RuntimeState, index: int) -> np.ndarray:
    actions = runtime.replay_trajectory
    if actions is not None and len(actions) > 0:
        raw_action = np.asarray(actions[min(index, len(actions) - 1)], dtype=np.float32)
    elif runtime.replay_source is not None and hasattr(runtime.replay_source, "get_scene_qpos"):
        raw_action = np.asarray(
            runtime.replay_source.get_scene_qpos(index),
            dtype=np.float32,
        )
    else:
        raise RuntimeError("Replay trajectory is unavailable")
    action_chunk = normalize_action_chunk(raw_action)
    replay_mode = _resolve_replay_action_mode(config, runtime)
    if isinstance(replay_mode, EEFPose):
        action_chunk = _fill_missing_replay_eef_gripper(
            runtime, replay_mode, action_chunk, index
        )
        canonical_eef_chunk = replay_mode.normalize_chunk_to_canonical(action_chunk)
        ik_solver = ensure_ik_solver(config, runtime)
        seed_qpos = _replay_qpos_at(runtime, index) if index == 0 else None
        action_chunk = ik_solver.solve_chunk(canonical_eef_chunk, seed_qpos=seed_qpos)
    action = action_chunk[0].astype(np.float32, copy=True)
    runtime.robot.snap_grippers(
        action,
        config.robot.gripper_threshold,
        config.robot.gripper_open,
        config.robot.gripper_close,
    )
    return action


def _first_replay_qpos(config: ConfigDict, runtime: RuntimeState) -> np.ndarray | None:
    replay = runtime.replay_source
    if replay is not None and hasattr(replay, "get_scene_qpos"):
        return np.asarray(replay.get_scene_qpos(0), dtype=np.float32)
    actions = runtime.replay_trajectory
    if (
        actions is not None
        and len(actions) > 0
        and not _resolve_replay_action_mode(config, runtime).is_eef()
    ):
        return np.asarray(actions[0], dtype=np.float32)
    return None


def fetch_action_chunk(
    config: ConfigDict,
    runtime: RuntimeState,
    prompt: str,
    session: SessionState | None = None,
    prebuilt_observation: dict | None = None,
) -> np.ndarray:
    """Run one policy inference and return a publish-ready joint-space action chunk.

    Waits for a transport frame (unless a prebuilt observation is given), calls the
    policy, then time-scales, IK-converts (for EEF modes), and gripper-snaps the
    predicted actions. The raw chunk is recorded to the debug sidecar when a session
    is present.

    Args:
        prompt: task instruction passed to the policy.
        prebuilt_observation: pre-assembled observation dict to reuse instead of
            pulling and building a fresh frame (used by the async/rtc loop).

    Returns:
        [T, total_action_dim] float32 joint-space action chunk.

    Raises:
        RuntimeError: the policy client is unavailable.
        KeyError: the policy response has no "actions" field.
        SystemExit: the transport shut down while waiting for a frame.
    """
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    while not runtime.transport.is_shutdown():
        observation = prebuilt_observation
        if observation is None:
            frame = runtime.transport.get_frame()
            if frame is None:
                rate.sleep()
                continue
            observation = build_policy_observation(frame, prompt, config, runtime)
        infer_start = time.monotonic()
        if runtime.policy is None:
            raise RuntimeError("Policy client is unavailable")
        logger.debug("[INFER] prompt=%r", prompt)
        response = runtime.policy.infer(observation)
        infer_ms = 1000.0 * (time.monotonic() - infer_start)
        if session is not None:
            session.last_infer_ms = infer_ms
        logger.debug("Policy infer %.1f ms", infer_ms)
        if "actions" not in response:
            raise KeyError(f"Response does not contain 'actions': {response}")
        original_chunk = normalize_action_chunk(response["actions"])
        submit_rl_critic(runtime, observation, original_chunk, "policy")
        action_chunk = normalize_policy_action_chunk(
            config, runtime, original_chunk, response=response
        )
        runtime.robot.snap_grippers(
            action_chunk,
            config.robot.gripper_threshold,
            config.robot.gripper_open,
            config.robot.gripper_close,
        )
        step_index = session.step_index if session is not None else -1
        logger.debug(
            "[INFER_CHUNK] step=%d infer_ms=%.1f raw_shape=%s action_shape=%s "
            "delta=%.4f first=%s last=%s",
            step_index,
            infer_ms,
            original_chunk.shape,
            action_chunk.shape,
            _chunk_delta_norm(action_chunk),
            _format_diag_vector(action_chunk[0]),
            _format_diag_vector(action_chunk[-1]),
        )
        return action_chunk
    raise SystemExit(0)


def _loop_observation(
    config: ConfigDict, runtime: RuntimeState, prompt: str
) -> dict | None:
    """Build a policy observation from the latest frame; None if none is available."""
    frame = runtime.transport.get_frame()
    if frame is None:
        return None
    return build_policy_observation(frame, prompt, config, runtime)


def _loop_fetch_chunk(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    prompt: str,
    obs: dict | None,
) -> np.ndarray:
    """Run one loop/warmup inference, reusing a prebuilt observation when given."""
    return fetch_action_chunk(config, runtime, prompt, session, prebuilt_observation=obs)


def start_inference_loop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Start the continuous inference loop for async/rtc strategies."""
    if runtime.infer_strategy is None or session.selected_task is None:
        return
    if not runtime.infer_strategy.runs_background_loop:
        return

    prompt = session.selected_task
    runtime.infer_strategy.start_loop(
        prompt,
        partial(_loop_fetch_chunk, config, runtime, session),
        partial(_loop_observation, config, runtime, prompt),
    )


def run_warmup_and_start(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Fetch fresh warmup chunks into the buffer and start the async loop.

    Called when the user presses start. Discards any stale state and does N
    fresh inferences so the smooth buffer has overlap data before the first
    action is published.

    Fast path: if eval.skip_warmup_after_first is true and warmup has already
    happened once in this process (tracked via runtime._eval_warmup_done),
    we skip the N pre-inferences and only anchor the buffer + start the loop.
    """
    assert session.selected_task is not None

    replay = runtime.replay_source
    if replay is not None:
        replay.seek(0)
        if runtime.policy is not None:
            runtime.policy.reset()
        if runtime.infer_strategy is not None:
            runtime.infer_strategy.reset()
        session.action_chunk = None
        session.chunk_index = 0
        session.step_index = 0
        session.pending_real_chunk = None
        session.sim_preview_qpos = None
        return True

    warmup_n = max(1, config.inference_cfg.setup_warmup_chunks)

    skip_warmup = (
        config.eval
        and config.eval.skip_warmup_after_first
        and runtime._eval_warmup_done
    )

    if runtime.infer_strategy is not None:
        runtime.infer_strategy.reset()
    reset_ik_solver(config, runtime)
    session.action_chunk = None
    session.chunk_index = 0
    _anchor_buffer_to_current_qpos(runtime)

    if skip_warmup:
        logger.info("Eval fast-path: skipping per-trial warmup")
        start_inference_loop(config, runtime, session)
        return True

    try:
        for i in range(warmup_n):
            runtime.setup_stage = f"warming up {i + 1}/{warmup_n}"
            if runtime.infer_strategy is None:
                chunk = fetch_action_chunk(config, runtime, session.selected_task, session)
            else:
                chunk = runtime.infer_strategy.prepare_warmup_chunk(
                    session.selected_task,
                    partial(_loop_fetch_chunk, config, runtime, session),
                )
            if i < warmup_n - 1:
                logger.info("Warmup %d/%d: integrated %d actions", i + 1, warmup_n, len(chunk))
    except PolicyRequestError as error:
        invalidate_policy_connection(config, runtime, session, error)
        runtime.setup_stage = ""
        return False
    except Exception as error:
        reset_run_state(config, runtime, session)
        session.last_error = f"Warmup failed: {error}"
        runtime.setup_stage = ""
        logger.info("Warmup failed: %s", error)
        return False

    logger.info("Warmup complete: %d chunks, %d actions ready", warmup_n, len(chunk))
    runtime._eval_warmup_done = True
    runtime.setup_stage = ""

    start_inference_loop(config, runtime, session)
    return True


def _anchor_buffer_to_current_qpos(runtime: RuntimeState) -> None:
    """Anchor the smooth buffer to the robot's actual qpos.

    Warmup integrates its first prediction against this measured position so the
    first published action does not jump from an unexecuted prediction.
    """
    if runtime.infer_strategy is None:
        return
    current_qpos = runtime.transport.get_latest_qpos()
    if current_qpos is None:
        return
    runtime.infer_strategy.seed_buffer(np.asarray(current_qpos, dtype=np.float32))


def ensure_action_chunk(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Ensure a fresh action chunk is buffered for chunk-based (sync) publishing.

    Returns early when the current chunk still has unconsumed actions; otherwise it
    connects the policy if needed and fetches a new chunk (via the strategy when one is
    active), recording it to the ensembled debug sidecar. On policy/setup failure it
    resets the session and records last_error.

    Returns:
        True when session.action_chunk holds a usable chunk, False on failure.
    """
    if session.action_chunk is not None and session.chunk_index < len(session.action_chunk):
        return True
    if not connect_policy(config, runtime, session):
        return False
    if runtime.policy is None:
        return False
    assert session.selected_task is not None
    try:
        if runtime.infer_strategy is None:
            session.action_chunk = fetch_action_chunk(
                config, runtime, session.selected_task, session
            )
        else:
            session.action_chunk = runtime.infer_strategy.take_or_fetch_chunk(
                session.selected_task,
                partial(_loop_fetch_chunk, config, runtime, session),
            )
    except PolicyRequestError as error:
        invalidate_policy_connection(config, runtime, session, error)
        return False
    except Exception as error:
        reset_run_state(config, runtime, session)
        session.last_error = f"Setup failed: {error}"
        logger.info("Action chunk preparation failed: %s", error)
        return False
    session.chunk_index = 0
    if session.action_chunk is not None:
        logger.info(
            "[CHUNK_READY] step=%d len=%d delta=%.4f first=%s last=%s",
            session.step_index,
            len(session.action_chunk),
            _chunk_delta_norm(session.action_chunk),
            _format_diag_vector(session.action_chunk[0]),
            _format_diag_vector(session.action_chunk[-1]),
        )
    return True


def _replay_total_frames(runtime: RuntimeState, replay: TransportBridge) -> int:
    total_frames = int(getattr(replay, "n_steps", 0) or 0)
    if total_frames <= 0 and runtime.replay_trajectory is not None:
        total_frames = int(len(runtime.replay_trajectory))
    return total_frames


def _replay_progress_update(runtime: RuntimeState, step_index: int, total_frames: int) -> None:
    """Drive a single-line tqdm bar for replay playback, recreating it on rewind."""
    pbar = runtime.replay_pbar
    if pbar is None or pbar.total != total_frames or step_index < pbar.n:
        _replay_progress_close(runtime)
        pbar = tqdm(total=total_frames, desc="replay", unit="frame", leave=True)
        runtime.replay_pbar = pbar
    pbar.update(step_index - pbar.n)


def _replay_progress_close(runtime: RuntimeState) -> None:
    """Close and drop the active replay progress bar, if any."""
    if runtime.replay_pbar is not None:
        runtime.replay_pbar.close()
        runtime.replay_pbar = None


def _publish_replay_frame(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    from core.app.handlers.control import apply_gripper_control_overrides, publish_action

    replay = runtime.replay_source
    if replay is None:
        return False
    total_frames = _replay_total_frames(runtime, replay)
    if total_frames <= 0:
        set_status(session, SessionStatus.READY, reason="empty replay episode")
        return False
    if session.step_index >= total_frames:
        set_status(session, SessionStatus.READY, reason="replay reached end of episode")
        return False

    frame_index = max(0, min(session.step_index, total_frames - 1))
    replay.seek(frame_index)
    data_only = is_offline_transport(runtime) or session.mode is SessionMode.STEP
    action = apply_gripper_control_overrides(
        runtime,
        _replay_action_at(config, runtime, frame_index),
        session,
    )
    target = "data"
    if not data_only:
        if session.mode is SessionMode.SIM:
            publish_action(runtime, action, target=OutputTarget.SIM.value)
            target = OutputTarget.SIM.value
        elif session.mode is SessionMode.REAL:
            if runtime.transport.has_sim_publishers():
                publish_action(runtime, action, target=OutputTarget.SIM.value)
            publish_action(runtime, action, target=OutputTarget.REAL.value)
            target = OutputTarget.REAL.value

    session.step_index += 1
    next_frame = min(session.step_index, total_frames - 1)
    replay.seek(next_frame)
    if session.mode is SessionMode.SIM:
        if hasattr(replay, "get_scene_qpos"):
            session.sim_preview_qpos = np.asarray(
                replay.get_scene_qpos(),
                dtype=np.float32,
            ).copy()
        else:
            session.sim_preview_qpos = action.copy()

    logger.debug(
        "[REPLAY_FRAME] target=%s episode=%d frame=%d->%d/%d step=%d",
        target,
        getattr(replay, "episode_id", -1),
        frame_index,
        next_frame,
        total_frames,
        session.step_index,
    )
    _replay_progress_update(runtime, session.step_index, total_frames)
    if session.step_index >= total_frames:
        set_status(session, SessionStatus.READY, reason="replay reached end of episode")
    return True


def _publish_replay_step_chunk(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
    from core.app.handlers.control import apply_gripper_control_overrides, publish_action

    replay = runtime.replay_source
    if replay is None:
        return False
    total_frames = _replay_total_frames(runtime, replay)
    if total_frames <= 0:
        set_status(session, SessionStatus.READY, reason="empty replay episode")
        return False
    if session.step_index >= total_frames:
        set_status(session, SessionStatus.READY, reason="replay reached end of episode")
        return False

    actions: list[np.ndarray] = []
    rate = runtime.transport.create_rate(runtime.replay_fps)
    n = max(1, int(runtime.replay_exec_steps))
    start_frame = session.step_index
    for _ in range(n):
        if session.step_index >= total_frames:
            break
        frame_index = max(0, min(session.step_index, total_frames - 1))
        replay.seek(frame_index)
        action = apply_gripper_control_overrides(
            runtime,
            _replay_action_at(config, runtime, frame_index),
            session,
        )
        session.sim_preview_qpos = action.copy()
        publish_action(runtime, action, target=OutputTarget.SIM.value)
        actions.append(action.copy())
        session.step_index += 1
        rate.sleep()

    if not actions:
        return False
    pending = np.asarray(actions, dtype=np.float32)
    session.pending_real_chunk = pending
    session.sim_preview_qpos = pending[-1].copy()
    next_frame = min(session.step_index, total_frames - 1)
    replay.seek(next_frame)
    logger.info(
        "[REPLAY_STEP_CHUNK] episode=%d frame=%d->%d/%d actions=%d",
        getattr(replay, "episode_id", -1),
        start_frame,
        next_frame,
        total_frames,
        len(pending),
    )
    if session.step_index >= total_frames:
        set_status(session, SessionStatus.READY, reason="replay reached end of episode")
    return True


def select_episode(
    episode_id: int, config: ConfigDict, session: SessionState, runtime: RuntimeState
) -> None:
    """Replay-only: load a different dataset episode and adopt its task as the prompt.

    Swaps the dataset transport to the requested episode, takes its recorded task as the
    prompt, settles SELECT mode to SIM so auto-setup can proceed, and reconnects the
    replay policy so its trajectory matches. No-op without a dataset transport / out of
    range.
    """
    # Replay-only: swap the dataset episode, adopt its recorded task as the prompt,
    # and reconnect the replay policy so its trajectory matches the new episode.
    transport = runtime.transport
    if not isinstance(transport, DatasetTransport):
        logger.warning("select_episode requires a dataset transport")
        return
    if not transport.has_episode(episode_id):
        session.last_error = f"Episode {episode_id} out of range (0..{transport.n_episodes - 1})"
        logger.warning("select_episode: %s", session.last_error)
        return
    transport.reload_episode(episode_id)
    session.selected_task = transport.current_task
    reset_session_progress(session)
    # Settle the run mode so the console's auto-setup can proceed; without a concrete
    # mode it sits on "AWAITING CONFIG…" forever (it never fires while mode is SELECT).
    # Launch-replay plays recorded data, so SIM is the right default; the operator can
    # still switch to REAL/STEP afterwards.
    if session.mode is SessionMode.SELECT:
        session.mode = SessionMode.SIM
    reset_infer_strategy(runtime)
    reset_ik_solver(config, runtime)
    if config.policy.type == "replay":
        connect_policy(config, runtime, session, force_reconnect=True)
    logger.info(
        "Selected episode %d (%d steps); task -> %s",
        episode_id,
        transport.n_steps,
        format_task_label(session.selected_task),
    )


def _build_replay_dataset_config(
    config: ConfigDict,
    dataset_dir: str,
    episode_id: int,
    state_key: str,
    action_key: str,
    series_state_key: str,
    video_keys: dict[str, str],
) -> ConfigDict:
    """Clone the live config but point its transport at the chosen dataset + keys.

    The mounted replay DatasetTransport reads dataset_keys / image size / disabled lists
    off config.transport, but the live config is a real-robot (zmq) one. This overrides
    only the dataset-relevant fields so the source decodes the right columns/videos.
    """
    import copy
    new_config = copy.deepcopy(config)
    new_config.transport.dataset_dir = dataset_dir
    new_config.transport.episode_id = episode_id
    new_config.transport.dataset_keys = ConfigDict(
        state_key=state_key,
        action_key=action_key,
        series_state_key=series_state_key,
        video_keys=dict(video_keys),
    )
    return new_config


def _is_eef_action_key(key: str) -> bool:
    normalized = key.lower().replace(".", "_").replace("/", "_")
    return normalized in {"action_eef", "actions_eef"} or (
        normalized.startswith("action") and "eef" in normalized
    )


def _is_eef_state_key(key: str) -> bool:
    normalized = key.lower().replace(".", "_").replace("/", "_")
    return normalized in {
        "state_eef",
        "observation_eef",
        "observations_eef",
        "observation_state_eef",
        "observations_state_eef",
    }


def _resolve_replay_dataset_action_key(
    inferred: dict,
    action_key: str,
    replay_mode: JointState | EEFPose,
) -> str:
    action = inferred.get("action") or {}
    candidates = list(action.get("candidates") or [])
    if replay_mode.is_eef() and action_key.lower() in {"", "action", "actions"}:
        for candidate in candidates:
            if _is_eef_action_key(candidate):
                return candidate
    if action_key:
        return action_key
    return action.get("default") or "action"


def _resolve_replay_dataset_series_state_key(
    inferred: dict,
    replay_mode: JointState | EEFPose,
) -> str:
    if not replay_mode.is_eef():
        return ""
    state = inferred.get("state") or {}
    for candidate in list(state.get("candidates") or []):
        if _is_eef_state_key(candidate):
            return candidate
    return ""


def load_replay_dataset(
    dataset_dir: str,
    episode_id: int,
    config: ConfigDict,
    session: SessionState,
    runtime: RuntimeState,
    state_key: str = "",
    action_key: str = "",
    video_keys: dict[str, str] | None = None,
    action_mode: str = "",
) -> None:
    """Mount a recorded dataset episode as the runtime replay/QC source.

    Backs the REPLAY tab: dataset dir / episode / column roles are chosen online. A
    read-only DatasetTransport is built (decoding the selected state/action/video keys)
    and stored on ``runtime.replay_source`` — its presence makes is_replay() true, the
    console reads recorded images/qpos from it, and the recorded action trajectory is
    cached for frame-by-frame playback. Unreadable dirs / bad episodes -> last_error.
    """
    video_keys = video_keys or {}
    if dataset_dir:
        dataset_dir = str(_resolve_runtime_path(dataset_dir))
    runtime.collection_replay_qpos = None
    runtime.collection_replay_episode = None
    replay_mode = parse_policy_action_mode(action_mode) or config.inference_cfg.action_space
    should_infer_keys = not state_key or not action_key or not video_keys or replay_mode.is_eef()
    series_state_key = ""
    if should_infer_keys:
        try:
            inferred = LeRobotDatasetIO(dataset_dir).infer_keys()
        except Exception:
            inferred = {}
        if not state_key:
            state_key = (inferred.get("state") or {}).get("default") or "observations.state.qpos"
        action_key = _resolve_replay_dataset_action_key(inferred, action_key, replay_mode)
        series_state_key = _resolve_replay_dataset_series_state_key(inferred, replay_mode)
        if not video_keys:
            image = inferred.get("image") or {}
            candidates = image.get("candidates") or []
            default_image = image.get("default")
            for camera in runtime.robot.observation_schema.cameras:
                cam_key = camera.observation_key
                match = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate == f"observation.images.{cam_key}"
                        or candidate.endswith(f".{cam_key}")
                    ),
                    None,
                )
                if match is not None:
                    video_keys[cam_key] = match
            if not video_keys and default_image:
                video_keys[runtime.robot.observation_schema.cameras[0].observation_key] = (
                    default_image
                )
    try:
        n_episodes = LeRobotDatasetIO(dataset_dir).count_episodes()
    except Exception as error:
        session.last_error = f"Cannot read dataset {dataset_dir}: {error}"
        logger.warning("load_replay_dataset: %s", session.last_error)
        return
    if n_episodes > 0 and not 0 <= episode_id < n_episodes:
        session.last_error = f"Episode {episode_id} out of range (0..{n_episodes - 1})"
        logger.warning("load_replay_dataset: %s", session.last_error)
        return
    try:
        replay_config = _build_replay_dataset_config(
            config, dataset_dir, episode_id, state_key, action_key, series_state_key, video_keys
        )
        source = DatasetTransport(replay_config, runtime.robot, dataset_dir, episode_id)
        actions = source.get_action_trajectory()
        task = source.current_task
    except Exception as error:
        session.last_error = f"Cannot load episode {episode_id}: {error}"
        logger.warning("load_replay_dataset: %s", session.last_error)
        return

    if runtime.replay_source is not None:
        runtime.replay_source.close()
    runtime.replay_source = source
    runtime.replay_dataset_dir = dataset_dir
    runtime.replay_episode_id = episode_id
    runtime.replay_n_episodes = n_episodes
    runtime.replay_trajectory = actions
    runtime.replay_task = task
    runtime.replay_action_key = action_key
    runtime.replay_action_mode = action_mode
    runtime.replay_fps = source.fps
    session.selected_task = task
    session.last_error = ""
    reset_session_progress(session)
    # Default a freshly loaded replay to SIM so the run mode is settled immediately;
    # without a concrete mode the console's auto-setup sits on "AWAITING CONFIG…"
    # forever (it never fires while mode is SELECT). The operator can still switch to
    # REAL/STEP in the panel afterwards.
    if session.mode is SessionMode.SELECT:
        session.mode = SessionMode.SIM
    reset_infer_strategy(runtime)
    reset_ik_solver(config, runtime)
    runtime.policy = None
    runtime.policy_metadata = None
    runtime.last_policy_error = ""
    logger.info(
        "[REPLAY_LOAD] dataset=%s episode=%d steps=%d fps=%d action_mode=%s "
        "state_key=%s action_key=%s frame=%d qpos=%s task=%s",
        dataset_dir,
        episode_id,
        actions.shape[0],
        source.fps,
        action_mode or "config",
        state_key,
        action_key,
        source.frame_index,
        _format_diag_vector(source.get_scene_qpos()),
        format_task_label(task),
    )


def clear_replay(config: ConfigDict, session: SessionState, runtime: RuntimeState) -> None:
    """Unmount the replay/QC source and return to normal (real-policy) inference.

    Closes the dataset source, clears the cached trajectory, and reconnects the policy
    backend declared in config (e.g. openpi) so the console leaves replay mode.
    """
    if runtime.replay_source is not None:
        runtime.replay_source.close()
    runtime.replay_source = None
    _replay_progress_close(runtime)
    runtime.replay_trajectory = None
    runtime.replay_task = ""
    runtime.replay_action_key = ""
    runtime.replay_dataset_dir = ""
    runtime.replay_n_episodes = 0
    runtime.replay_action_mode = ""
    session.selected_task = None
    session.last_error = ""
    reset_run_state(config, runtime, session)
    runtime.policy = None
    connect_policy(config, runtime, session, force_reconnect=True)
    logger.info("Cleared replay source; reconnected %s policy", config.policy.type)


def set_replay_fps(runtime: RuntimeState, fps: int) -> None:
    """Set the offline replay playback rate (frames/sec); clamped to a sane minimum."""
    runtime.replay_fps = max(1, int(fps))
    logger.info("Replay playback rate set to %d fps", runtime.replay_fps)


__all__ = [
    "connect_policy",
    "invalidate_policy_connection",
    "disconnect_policy",
    "parse_policy_action_mode",
    "resolve_policy_action_mode",
    "ensure_ik_solver",
    "normalize_policy_action_chunk",
    "_resolve_replay_action_mode",
    "_replay_qpos_at",
    "_fill_missing_replay_eef_gripper",
    "_replay_action_at",
    "_first_replay_qpos",
    "fetch_action_chunk",
    "_loop_observation",
    "_loop_fetch_chunk",
    "start_inference_loop",
    "run_warmup_and_start",
    "_anchor_buffer_to_current_qpos",
    "ensure_action_chunk",
    "_replay_total_frames",
    "_replay_progress_update",
    "_replay_progress_close",
    "_publish_replay_frame",
    "_publish_replay_step_chunk",
    "select_episode",
    "_build_replay_dataset_config",
    "_is_eef_action_key",
    "_resolve_replay_dataset_action_key",
    "load_replay_dataset",
    "clear_replay",
    "set_replay_fps",
]
