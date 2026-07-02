"""Shared leaf helpers: path/diagnostic formatting + stateless session/strategy/IK resets."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.app.state import (
    RuntimeState,
    SessionState,
    SessionStatus,
)
from core.config import ConfigDict

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_runtime_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return _REPO_ROOT / resolved


def _format_diag_vector(value: np.ndarray, max_dims: int = 8) -> str:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.array2string(vector[:max_dims], precision=4, separator=", ")


def _chunk_delta_norm(chunk: np.ndarray) -> float:
    if len(chunk) < 2:
        return 0.0
    return float(np.linalg.norm(chunk[-1] - chunk[0]))


def is_offline_transport(runtime: RuntimeState) -> bool:
    """True when the transport has no live robot (debug/dataset)."""
    return runtime.transport.is_offline()


def is_replay(runtime: RuntimeState) -> bool:
    """True when a dataset replay source is currently mounted."""
    return runtime.replay_source is not None


def reset_session_progress(session: SessionState) -> None:
    """Clear the session back to UNSET: drop chunks, counters, and the setup flag."""
    session.status = SessionStatus.UNSET
    session.action_chunk = None
    session.pending_real_chunk = None
    session.sim_preview_qpos = None
    session.chunk_index = 0
    session.step_index = 0
    session.is_setup_done = False


def reset_infer_strategy(runtime: RuntimeState) -> None:
    """Stop the strategy loop and reset both the strategy and the policy's internal state."""
    if runtime.infer_strategy is not None:
        runtime.infer_strategy.stop_loop()
    if runtime.policy is not None and hasattr(runtime.policy, "reset"):
        runtime.policy.reset()
    if runtime.infer_strategy is not None:
        runtime.infer_strategy.reset()


def reset_ik_solver(config: ConfigDict, runtime: RuntimeState) -> None:
    """Reset the IK solver to its seed pose at the configured control dt (no-op if absent)."""
    if runtime.ik_solver is None:
        return
    if hasattr(runtime.ik_solver, "reset"):
        reset_kwargs: dict[str, object] = {
            "initial_qpos_groups": runtime.robot.initial_qpos_by_group(),
            "dt": 1.0 / max(config.inference_cfg.publish_rate, 1),
        }
        runtime.ik_solver.reset(**reset_kwargs)


def reset_run_state(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Clear session progress and reset the inference strategy + IK solver together.

    The standard teardown applied whenever the active run is abandoned (policy
    drop/disconnect, replay (un)mount, episode/strategy switch).
    """
    reset_session_progress(session)
    reset_infer_strategy(runtime)
    reset_ik_solver(config, runtime)


__all__ = [
    "_REPO_ROOT",
    "_resolve_runtime_path",
    "_format_diag_vector",
    "_chunk_delta_norm",
    "is_offline_transport",
    "is_replay",
    "reset_session_progress",
    "reset_infer_strategy",
    "reset_ik_solver",
    "reset_run_state",
]
