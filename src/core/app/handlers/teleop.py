"""Application orchestration for client-driven teleoperation."""

from __future__ import annotations

import dataclasses
import enum
import logging
import time
from collections.abc import Callable

import numpy as np

from core.app.collection_capture import stop_collection_capture
from core.app.state import OutputTarget, RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict
from teleop_client import build_client
from teleop_client.base import (
    CanonicalEefCommand,
    QposCommand,
    TeleopClient,
    TeleopClientError,
    TeleopContext,
    TeleopOperatorEvent,
    TeleopResultKind,
)

logger = logging.getLogger(__name__)

TELEOP_CONTROL_SOURCE_TRANSPORT = "transport"
TELEOP_CONTROL_SOURCE_CLIENT = "client"


class TeleopExecutionError(RuntimeError):
    """Raised for one invalid client-driven control tick."""


class TeleopExecutionCondition(str, enum.Enum):
    """Stable application-level condition for client-driven teleoperation."""

    IDLE = "idle"
    COMMANDING = "commanding"
    REJECTED = "rejected"


@dataclasses.dataclass
class TeleopExecutionState:
    control_source: str = TELEOP_CONTROL_SOURCE_TRANSPORT
    client_type: str = ""
    active: bool = False
    condition: TeleopExecutionCondition = TeleopExecutionCondition.IDLE
    motion_started: bool = False
    published_actions: int = 0
    ik_prewarmed: bool = False
    last_safe_qpos: np.ndarray | None = None
    last_fault: str = ""
    max_qpos_step: float = 0.08
    max_position_error_m: float = 0.08
    max_orientation_error_rad: float = 0.35
    gripper_min: float = 0.0
    gripper_max: float = 1.0


@dataclasses.dataclass(frozen=True)
class PublishedTeleopAction:
    """The exact qpos successfully handed to the execution transport."""

    qpos: np.ndarray
    timestamp: float = dataclasses.field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        value = np.asarray(self.qpos, dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("published teleop action must be finite and non-empty")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("published teleop action timestamp must be finite and non-negative")
        object.__setattr__(self, "qpos", value.copy())
        object.__setattr__(self, "timestamp", timestamp)


def teleop_control_source(config: ConfigDict) -> str:
    cfg = (config.get("collection") or {}).get("teleop") or {}
    return str(cfg.get("control_source", TELEOP_CONTROL_SOURCE_TRANSPORT))


def _execution(runtime: RuntimeState) -> TeleopExecutionState:
    state = getattr(runtime, "teleop_execution", None)
    if not isinstance(state, TeleopExecutionState):
        state = TeleopExecutionState()
        runtime.teleop_execution = state
    return state


def setup_teleop(config: ConfigDict, runtime: RuntimeState) -> None:
    """Configure the selected input source without starting runtime resources."""
    source = teleop_control_source(config)
    teleop_cfg = config.collection.teleop
    safety = teleop_cfg.get("safety") or {}
    state = TeleopExecutionState(
        control_source=source,
        max_qpos_step=float(safety.get("max_qpos_step", 0.08)),
        max_position_error_m=float(safety.get("max_position_error_m", 0.08)),
        max_orientation_error_rad=float(safety.get("max_orientation_error_rad", 0.35)),
    )
    runtime.teleop_execution = state
    runtime.teleop_client = None
    if source == TELEOP_CONTROL_SOURCE_TRANSPORT:
        return
    client_cfg = teleop_cfg.client
    state.client_type = str(client_cfg.type)
    gripper_values = [
        float(config.robot.gripper_open),
        float(config.robot.gripper_close),
    ]
    state.gripper_min = min(gripper_values)
    state.gripper_max = max(gripper_values)
    client = build_client(
        client_cfg,
        arm_group_names=tuple(group.name for group in runtime.robot.arm_groups),
    )
    runtime.teleop_client = client


def _ensure_teleop_setup(config: ConfigDict, runtime: RuntimeState) -> None:
    """Build the configured client once, immediately before collection activation."""
    source = teleop_control_source(config)
    state = getattr(runtime, "teleop_execution", None)
    client = getattr(runtime, "teleop_client", None)
    configured = bool(
        isinstance(state, TeleopExecutionState)
        and state.control_source == source
        and (source == TELEOP_CONTROL_SOURCE_TRANSPORT or client is not None)
    )
    if not configured:
        setup_teleop(config, runtime)


def start_teleop_input(config: ConfigDict, runtime: RuntimeState) -> None:
    """Start the configured input listener independently of ARM state."""
    _ensure_teleop_setup(config, runtime)
    state = _execution(runtime)
    client = getattr(runtime, "teleop_client", None)
    if state.control_source == TELEOP_CONTROL_SOURCE_CLIENT:
        if client is None:
            raise TeleopExecutionError("Configured teleop client is unavailable")
        client.start()


def _current_qpos(runtime: RuntimeState) -> np.ndarray:
    qpos = runtime.transport.get_latest_qpos()
    if qpos is None:
        raise TeleopExecutionError("teleop requires current robot qpos feedback")
    value = np.asarray(qpos, dtype=np.float32).reshape(-1)
    expected = runtime.robot.total_action_dim
    if value.shape != (expected,) or not np.all(np.isfinite(value)):
        raise TeleopExecutionError(
            f"teleop qpos must be finite shape ({expected},), got {value.shape}"
        )
    return value


def _set_hil_relay_enabled(runtime: RuntimeState, enabled: bool) -> None:
    setter = getattr(runtime.transport, "set_hil_relay_enabled", None)
    if callable(setter):
        setter(enabled)


def forward_canonical_eef(
    config: ConfigDict,
    runtime: RuntimeState,
    qpos: np.ndarray,
) -> np.ndarray:
    from core.app.handlers.io import ensure_ik_solver

    solver = ensure_ik_solver(config, runtime)
    return np.asarray(solver.fk_chunk(np.asarray(qpos).reshape(1, -1))[0], dtype=np.float32)


def solve_canonical_eef(
    config: ConfigDict,
    runtime: RuntimeState,
    target_eef: np.ndarray,
    *,
    seed_qpos: np.ndarray,
) -> np.ndarray:
    from core.app.handlers.io import ensure_ik_solver

    solver = ensure_ik_solver(config, runtime)
    solved = solver.solve_chunk(np.asarray(target_eef).reshape(1, -1), seed_qpos=seed_qpos)
    return np.asarray(solved[0], dtype=np.float32)


def prewarm_teleop_ik(config: ConfigDict, runtime: RuntimeState) -> bool:
    """Compile the client-teleop FK/IK path without publishing a robot action."""
    try:
        _ensure_teleop_setup(config, runtime)
        state = _execution(runtime)
        if state.control_source != TELEOP_CONTROL_SOURCE_CLIENT:
            return False
        if state.ik_prewarmed and runtime.ik_solver is not None:
            return True

        started = time.monotonic()
        seed_qpos = np.asarray(runtime.robot.initial_qpos, dtype=np.float32).reshape(-1)
        if seed_qpos.shape != (runtime.robot.total_action_dim,) or not np.all(
            np.isfinite(seed_qpos)
        ):
            raise TeleopExecutionError("teleop IK prewarm requires finite robot initial qpos")
        expected_eef = 8 * len(runtime.robot.arm_groups)
        target_eef = forward_canonical_eef(config, runtime, seed_qpos).reshape(-1)[:expected_eef]
        solved_qpos = solve_canonical_eef(
            config,
            runtime,
            target_eef,
            seed_qpos=seed_qpos,
        )
        tracked_eef = forward_canonical_eef(config, runtime, solved_qpos).reshape(-1)[:expected_eef]
        if tracked_eef.shape != (expected_eef,) or not np.all(np.isfinite(tracked_eef)):
            raise TeleopExecutionError("teleop IK prewarm returned invalid FK")
    except Exception as error:
        logger.warning("Teleop IK prewarm deferred: %s", error)
        return False

    state.ik_prewarmed = True
    logger.info(
        "Teleop IK prewarm complete: arms=%d elapsed_ms=%.1f",
        len(runtime.robot.arm_groups),
        (time.monotonic() - started) * 1000.0,
    )
    return True


def configure_fixed_vr_home_eef(config: ConfigDict, runtime: RuntimeState) -> None:
    """Anchor VR retargeting to the configured robot initial pose, never live qpos."""
    client = getattr(runtime, "teleop_client", None)
    setter = getattr(client, "set_home_eef", None)
    if not callable(setter):
        return
    initial_qpos = np.asarray(runtime.robot.initial_qpos, dtype=np.float32).reshape(-1)
    expected_qpos = int(runtime.robot.total_action_dim)
    if initial_qpos.shape != (expected_qpos,) or not np.all(np.isfinite(initial_qpos)):
        raise TeleopExecutionError(
            f"configured VR home qpos must be finite shape ({expected_qpos},)"
        )
    initial_eef = np.asarray(
        forward_canonical_eef(config, runtime, initial_qpos), dtype=np.float32
    ).reshape(-1)
    expected_eef = 8 * len(runtime.robot.arm_groups)
    if initial_eef.shape != (expected_eef,) or not np.all(np.isfinite(initial_eef)):
        raise TeleopExecutionError(
            f"configured VR home EEF must be finite shape ({expected_eef},)"
        )
    setter(
        {
            group.name: initial_eef[index * 8 : (index + 1) * 8].copy()
            for index, group in enumerate(runtime.robot.arm_groups)
        }
    )
    logger.info("VR calibration origin set from robot.initial_qpos")


def activate_teleop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Enter the selected collection control lifecycle without opening an episode."""
    if not runtime.collection_teleop_armed:
        session.last_error = "Collection teleop requires COLLECT activation"
        return False
    state = _execution(runtime)
    client = getattr(runtime, "teleop_client", None)
    if runtime.collection_teleop_active:
        session.mode = SessionMode.COLLECT
        if session.status is not SessionStatus.RUNNING:
            session.status = SessionStatus.READY
        return True
    if not runtime.transport.supports_collection():
        session.last_error = "Transport does not support collection frames"
        return False
    try:
        _ensure_teleop_setup(config, runtime)
    except Exception as error:
        session.last_error = f"Cannot configure teleop: {error}"
        logger.exception("Teleop setup failed during collection activation")
        return False
    state = _execution(runtime)
    client = getattr(runtime, "teleop_client", None)
    collection_start_attempted = False
    try:
        if state.control_source == TELEOP_CONTROL_SOURCE_CLIENT:
            if client is None:
                raise TeleopExecutionError("Configured teleop client is unavailable")
            client.start()
            configure_fixed_vr_home_eef(config, runtime)
            # Collection ARM is an explicit operator gate. Do not make the gate
            # depend on the transient neutral snapshot; the per-arm grip latch
            # remains the source of which arm is allowed to follow.
            client.reset(require_neutral=False)
        runtime.transport.reset_hil_control()
        _set_hil_relay_enabled(
            runtime,
            state.control_source == TELEOP_CONTROL_SOURCE_TRANSPORT,
        )
        collection_start_attempted = True
        runtime.transport.start_collection()
    except Exception as error:
        cleanup_error = _cleanup_teleop(
            runtime,
            stop_transport=collection_start_attempted
            or state.active
            or runtime.collection_teleop_active,
            raise_error=False,
            close_client=False,
        )
        message = f"Cannot activate teleop: {error}"
        if cleanup_error is not None:
            message += f"; cleanup failed: {cleanup_error}"
        session.last_error = message
        logger.exception("Teleop activation failed")
        return False
    state.active = True
    state.condition = TeleopExecutionCondition.IDLE
    state.motion_started = False
    state.last_safe_qpos = None
    state.last_fault = ""
    runtime.collection_teleop_active = True
    runtime.last_collection_timestamp = None
    session.mode = SessionMode.COLLECT
    if session.status is not SessionStatus.RUNNING:
        session.status = SessionStatus.READY
    session.last_error = ""
    logger.info("Collection teleop activated: source=%s", state.control_source)
    return True


def reset_teleop(runtime: RuntimeState, *, require_neutral: bool = False) -> None:
    """Drop application execution state and client anchors; safe to call repeatedly."""
    state = _execution(runtime)
    state.condition = TeleopExecutionCondition.IDLE
    state.motion_started = False
    state.last_safe_qpos = None
    state.last_fault = ""
    client = getattr(runtime, "teleop_client", None)
    if client is not None:
        client.reset(require_neutral=require_neutral)


def _cleanup_teleop(
    runtime: RuntimeState,
    *,
    stop_transport: bool,
    raise_error: bool,
    close_client: bool = False,
) -> Exception | None:
    """Best-effort shutdown that makes publishing impossible before cleanup starts."""
    state = _execution(runtime)
    state.active = False
    runtime.collection_teleop_active = False
    runtime.last_collection_timestamp = None
    state.condition = TeleopExecutionCondition.IDLE
    state.motion_started = False
    state.last_safe_qpos = None
    client = getattr(runtime, "teleop_client", None)
    errors: list[Exception] = []

    def attempt(label: str, callback) -> None:
        try:
            callback()
        except Exception as error:
            errors.append(error)
            logger.error("Teleop cleanup step failed (%s): %s", label, error, exc_info=True)

    attempt("collection capture stop", lambda: stop_collection_capture(runtime))
    if client is not None:
        attempt("client reset", client.reset)
    attempt("HIL relay disable", lambda: _set_hil_relay_enabled(runtime, False))
    if stop_transport:
        attempt("transport collection stop", runtime.transport.stop_collection)
    if client is not None and close_client:
        # Retain the client object after a successful close so the next activation can
        # call start() again. A failed close is also retained for explicit diagnostics.
        attempt("client close", client.close)

    if errors:
        first = errors[0]
        state.condition = TeleopExecutionCondition.REJECTED
        state.last_fault = f"teleop cleanup failed: {first}"
        if raise_error:
            raise first
        return first
    state.condition = TeleopExecutionCondition.IDLE
    state.last_fault = ""
    return None


def deactivate_teleop(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> None:
    """Stop capture and leave collection control in a safe, reset state."""
    state = _execution(runtime)
    should_stop_transport = runtime.collection_teleop_active or state.active
    try:
        _cleanup_teleop(
            runtime,
            stop_transport=should_stop_transport,
            raise_error=True,
            close_client=False,
        )
    except Exception as error:
        session.last_error = f"Teleop cleanup failed: {error}"
        raise
    finally:
        if session.mode is SessionMode.COLLECT and session.status is not SessionStatus.RUNNING:
            session.status = SessionStatus.UNSET
        logger.info("Collection teleop deactivated")
        _ = config


def close_teleop(runtime: RuntimeState) -> None:
    """Best-effort final shutdown of every teleop-owned resource.

    This path is deliberately non-raising because it runs from the application's
    outer ``finally`` block. Every individual cleanup step is attempted before the
    error is recorded, so a failed client close cannot prevent relay/transport and
    later logger cleanup. A successfully closed client is discarded; a failed one is
    retained for diagnostics and a later explicit retry.
    """
    state = _execution(runtime)
    client = getattr(runtime, "teleop_client", None)
    should_stop_transport = bool(state.active or runtime.collection_teleop_active)
    cleanup_error = _cleanup_teleop(
        runtime,
        stop_transport=should_stop_transport,
        raise_error=False,
        close_client=False,
    )
    errors: list[Exception] = []
    if cleanup_error is not None:
        errors.append(cleanup_error)
    client_close_error: Exception | None = None
    if client is not None:
        try:
            client.close()
        except Exception as error:
            client_close_error = error
            errors.append(error)
            logger.error("Teleop client shutdown failed: %s", error, exc_info=True)
        else:
            runtime.teleop_client = None
    if errors:
        details = "; ".join(str(error) for error in errors)
        state.condition = TeleopExecutionCondition.REJECTED
        state.last_fault = f"Cannot close teleop: {details}"
        if client_close_error is not None:
            logger.error("Teleop client remains available for shutdown retry")
        return
    state.condition = TeleopExecutionCondition.IDLE
    state.last_fault = ""


def _quaternion_error(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return float("inf")
    cosine = float(np.clip(abs(np.dot(first / first_norm, second / second_norm)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


def _limit_qpos(
    runtime: RuntimeState,
    state: TeleopExecutionState,
    action: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if value.shape != previous.shape or not np.all(np.isfinite(value)):
        raise TeleopExecutionError("teleop command has invalid qpos shape or values")
    grippers = set(runtime.robot.gripper_indices)
    for index in range(value.size):
        if index in grippers:
            value[index] = np.clip(value[index], state.gripper_min, state.gripper_max)
        else:
            delta = value[index] - previous[index]
            value[index] = previous[index] + np.clip(
                delta, -state.max_qpos_step, state.max_qpos_step
            )
    return value.astype(np.float32)


def _apply_grippers(
    runtime: RuntimeState,
    state: TeleopExecutionState,
    action: np.ndarray,
    target_eef: np.ndarray,
) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).copy()
    indices = runtime.robot.gripper_indices
    if len(indices) * 8 != target_eef.size:
        raise TeleopExecutionError("teleop gripper and EEF arm counts differ")
    for arm_index, qpos_index in enumerate(indices):
        result[qpos_index] = np.clip(
            target_eef[arm_index * 8 + 7], state.gripper_min, state.gripper_max
        )
    return result


def _freeze_inactive_arm_joints(
    runtime: RuntimeState,
    action: np.ndarray,
    previous: np.ndarray,
    active_arms: tuple[bool, ...],
) -> np.ndarray:
    """Restore inactive arm joints while preserving their independent gripper target."""
    result = np.asarray(action, dtype=np.float32).copy()
    reference = np.asarray(previous, dtype=np.float32).reshape(-1)
    arm_index = 0
    offset = 0
    for group in runtime.robot.actuator_groups:
        end = offset + group.dof
        if group.gripper_index is not None:
            if not active_arms[arm_index]:
                gripper_index = offset + group.gripper_index
                gripper_target = result[gripper_index]
                result[offset:end] = reference[offset:end]
                result[gripper_index] = gripper_target
            arm_index += 1
        offset = end
    return result


def _solve_eef_command(
    config: ConfigDict,
    runtime: RuntimeState,
    state: TeleopExecutionState,
    command: CanonicalEefCommand,
    current_qpos: np.ndarray,
) -> np.ndarray | None:
    target = np.asarray(command.value, dtype=np.float32).reshape(-1)
    expected = 8 * len(runtime.robot.arm_groups)
    if target.shape != (expected,) or not np.all(np.isfinite(target)):
        raise TeleopExecutionError(
            f"teleop EEF command must be finite shape ({expected},), got {target.shape}"
        )
    if command.active_arms and len(command.active_arms) != len(runtime.robot.arm_groups):
        raise TeleopExecutionError("teleop active-arm mask has invalid length")
    active = command.active_arms or tuple(True for _ in runtime.robot.arm_groups)
    previous = current_qpos if state.last_safe_qpos is None else state.last_safe_qpos
    if not any(active):
        return _apply_grippers(runtime, state, previous, target)
    try:
        solved = solve_canonical_eef(config, runtime, target, seed_qpos=current_qpos)
        tracked = forward_canonical_eef(config, runtime, solved).reshape(-1)[:expected]
    except Exception as error:
        raise TeleopExecutionError(f"teleop IK failed: {error}") from error
    if tracked.shape != target.shape or not np.all(np.isfinite(tracked)):
        raise TeleopExecutionError("teleop FK validation returned invalid EEF")
    for arm_index, arm_active in enumerate(active):
        if not arm_active:
            continue
        offset = arm_index * 8
        position_error = float(
            np.linalg.norm(target[offset : offset + 3] - tracked[offset : offset + 3])
        )
        orientation_error = _quaternion_error(
            target[offset + 3 : offset + 7], tracked[offset + 3 : offset + 7]
        )
        if position_error > state.max_position_error_m:
            state.condition = TeleopExecutionCondition.REJECTED
            state.last_fault = f"arm {arm_index} IK position error {position_error:.4f} m"
            return None
        if orientation_error > state.max_orientation_error_rad:
            state.condition = TeleopExecutionCondition.REJECTED
            state.last_fault = f"arm {arm_index} IK orientation error {orientation_error:.4f} rad"
            return None
    limited = _limit_qpos(runtime, state, solved, previous)
    return _freeze_inactive_arm_joints(runtime, limited, previous, active)


def _step_active_teleop(
    config: ConfigDict,
    runtime: RuntimeState,
    state: TeleopExecutionState,
    client: TeleopClient,
    current_time: float,
    *,
    still_active: Callable[[], bool] | None = None,
) -> PublishedTeleopAction | None:
    current_qpos = _current_qpos(runtime)
    try:
        measured_eef = np.asarray(
            forward_canonical_eef(config, runtime, current_qpos), dtype=np.float32
        ).reshape(-1)[: 8 * len(runtime.robot.arm_groups)]
    except Exception as error:
        raise TeleopExecutionError(f"teleop FK failed: {error}") from error
    try:
        result = client.poll(
            TeleopContext(
                measured_qpos=current_qpos.copy(),
                measured_eef=measured_eef,
                now=current_time,
            )
        )
    except TeleopClientError as error:
        state.last_fault = str(error)
        state.condition = TeleopExecutionCondition.IDLE
        return None
    except Exception as error:
        raise TeleopExecutionError(f"teleop client failed: {error}") from error
    if result.kind is TeleopResultKind.REJECTED:
        state.condition = TeleopExecutionCondition.REJECTED
        state.last_fault = result.message
        return None
    if result.kind is TeleopResultKind.IDLE:
        state.condition = TeleopExecutionCondition.IDLE
        state.last_fault = ""
        return None
    if result.kind is not TeleopResultKind.COMMAND or result.command is None:
        raise TeleopExecutionError(f"unsupported teleop result kind: {result.kind!r}")
    command = result.command
    if isinstance(command, CanonicalEefCommand):
        action = _solve_eef_command(config, runtime, state, command, current_qpos)
        if action is None:
            return None
    elif isinstance(command, QposCommand):
        previous = current_qpos if state.last_safe_qpos is None else state.last_safe_qpos
        action = _limit_qpos(runtime, state, command.value, previous)
    else:
        raise TeleopExecutionError(f"unsupported teleop command: {type(command).__name__}")
    if not state.active or (still_active is not None and not still_active()):
        return None
    validator = getattr(client, "validate_result", None)
    if callable(validator):
        try:
            result_is_current = bool(validator(result))
        except Exception as error:
            raise TeleopExecutionError(f"teleop input validation failed: {error}") from error
        if not result_is_current:
            raise TeleopExecutionError("teleop input changed before publish")
    try:
        runtime.transport.publish_action(action, target=OutputTarget.REAL.value)
    except Exception as error:
        raise TeleopExecutionError(f"teleop action publish failed: {error}") from error
    state.last_safe_qpos = action.copy()
    state.condition = TeleopExecutionCondition.COMMANDING
    state.motion_started = True
    state.published_actions += 1
    state.last_fault = ""
    return PublishedTeleopAction(qpos=action)


def step_teleop(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    *,
    now: float | None = None,
) -> PublishedTeleopAction | None:
    """Poll one client command and publish at most one robot action."""
    return _step_client_teleop(
        config,
        runtime,
        active=(
            runtime.collection_teleop_armed
            and runtime.collection_teleop_active
            and session.mode is SessionMode.COLLECT
        ),
        still_active=lambda: (
            runtime.collection_teleop_armed
            and runtime.collection_teleop_active
            and session.mode is SessionMode.COLLECT
        ),
        dropped_label="teleop",
        now=now,
    )


def activate_rollout_teleop(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> bool:
    """Arm client teleop for rollout intervention without touching collection state."""
    try:
        state = _execution(runtime)
        client = getattr(runtime, "teleop_client", None)
        if state.control_source != TELEOP_CONTROL_SOURCE_CLIENT or client is None:
            _ensure_teleop_setup(config, runtime)
            state = _execution(runtime)
            client = getattr(runtime, "teleop_client", None)
        if state.control_source != TELEOP_CONTROL_SOURCE_CLIENT or client is None:
            raise TeleopExecutionError("Configured teleop client is unavailable")
        client.start()
        status = client.status()
        if not status.connected:
            raise TeleopExecutionError(status.source_error or "Teleop client is not connected")
        client.reset(require_neutral=False)
    except Exception as error:
        session.last_error = f"Cannot activate rollout teleop: {error}"
        logger.exception("Rollout teleop activation failed")
        return False
    state.active = True
    state.condition = TeleopExecutionCondition.IDLE
    state.motion_started = False
    state.last_safe_qpos = None
    state.last_fault = ""
    session.last_error = ""
    return True


def deactivate_rollout_teleop(runtime: RuntimeState) -> None:
    """Drop rollout-intervention teleop state while keeping the client alive."""
    client = getattr(runtime, "teleop_client", None)
    if client is not None:
        client.reset()
    state = _execution(runtime)
    state.active = False
    state.condition = TeleopExecutionCondition.IDLE
    state.motion_started = False
    state.last_safe_qpos = None
    state.last_fault = ""


def step_rollout_teleop(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    *,
    now: float | None = None,
) -> PublishedTeleopAction | None:
    """Poll one client command and publish at most one RL intervention action."""
    return _step_client_teleop(
        config,
        runtime,
        active=runtime.rollout_intervention_active and session.mode is SessionMode.REAL,
        still_active=lambda: (
            runtime.rollout_intervention_active and session.mode is SessionMode.REAL
        ),
        dropped_label="rollout teleop",
        now=now,
    )


def _step_client_teleop(
    config: ConfigDict,
    runtime: RuntimeState,
    *,
    active: bool,
    still_active: Callable[[], bool],
    dropped_label: str,
    now: float | None,
) -> PublishedTeleopAction | None:
    state = _execution(runtime)
    client = getattr(runtime, "teleop_client", None)
    if (
        state.control_source != TELEOP_CONTROL_SOURCE_CLIENT
        or client is None
        or not state.active
        or not active
    ):
        return None
    current_time = time.monotonic() if now is None else float(now)
    try:
        return _step_active_teleop(
            config,
            runtime,
            state,
            client,
            current_time,
            still_active=still_active,
        )
    except Exception as error:
        state.condition = TeleopExecutionCondition.REJECTED
        state.last_fault = str(error)
        logger.warning("Dropped %s tick: %s", dropped_label, error)
        return None


def drain_teleop_events(runtime: RuntimeState) -> tuple[TeleopOperatorEvent, ...]:
    client = getattr(runtime, "teleop_client", None)
    return () if client is None else client.drain_events()


def acknowledge_teleop_event(
    runtime: RuntimeState,
    event: TeleopOperatorEvent,
    *,
    accepted: bool,
    message: str,
) -> None:
    client = getattr(runtime, "teleop_client", None)
    if client is not None:
        client.acknowledge_event(event, accepted=accepted, message=message)


def teleop_status(runtime: RuntimeState) -> dict[str, object] | None:
    """Return the lightweight input-source status exposed by the Console API."""
    state = _execution(runtime)
    if state.control_source != TELEOP_CONTROL_SOURCE_CLIENT:
        return None
    client = getattr(runtime, "teleop_client", None)
    source = None if client is None else client.status()
    return {
        "control_source": state.control_source,
        "client_type": state.client_type,
        "active": state.active,
        "condition": state.condition.value,
        "connected": False if source is None else source.connected,
        "input_age_ms": None if source is None else source.input_age_ms,
        "engaged_groups": [] if source is None else list(source.engaged_groups),
        "held_groups": [] if source is None else list(source.held_groups),
        "authorized_groups": [] if source is None else list(source.authorized_groups),
        "pressed_controls": [] if source is None else list(source.pressed_controls),
        "hold_progress": {} if source is None else dict(source.hold_progress),
        "neutral": False if source is None else source.neutral,
        "motion_started": state.motion_started,
        "published_actions": state.published_actions,
        "last_fault": state.last_fault,
        "source_error": "" if source is None else source.source_error,
    }


__all__ = [
    "TELEOP_CONTROL_SOURCE_CLIENT",
    "TELEOP_CONTROL_SOURCE_TRANSPORT",
    "TeleopExecutionError",
    "TeleopExecutionCondition",
    "TeleopExecutionState",
    "PublishedTeleopAction",
    "acknowledge_teleop_event",
    "activate_teleop",
    "activate_rollout_teleop",
    "close_teleop",
    "configure_fixed_vr_home_eef",
    "deactivate_rollout_teleop",
    "deactivate_teleop",
    "drain_teleop_events",
    "prewarm_teleop_ik",
    "reset_teleop",
    "setup_teleop",
    "start_teleop_input",
    "step_rollout_teleop",
    "step_teleop",
    "teleop_control_source",
    "teleop_status",
]
