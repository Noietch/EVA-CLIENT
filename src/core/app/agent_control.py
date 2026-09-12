"""Agent-facing robot operations executed by the EVA main loop.

The MCP server is a separate process. It submits typed operations through the
existing ZMQ control channel, while this module owns the small amount of runtime
logic that must stay in-process with the robot transport: FK/IK, joint-space
interpolation, publication, cancellation, and capability/state snapshots.
"""

from __future__ import annotations

import base64
import dataclasses
import math
import uuid
from typing import Any

import cv2
import numpy as np

from core.app.handlers.control import publish_action_chunk
from core.app.handlers.imaging import build_linear_trajectory
from core.app.handlers.io import ensure_ik_solver
from core.app.handlers.utils import reset_infer_strategy
from core.app.state import (
    OutputTarget,
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
    set_phase,
    set_status,
)
from core.config import ConfigDict
from robots.base import Robot

_DIRECT_ACTIONS = frozenset({"move_eef", "move_joints", "set_gripper", "solve_ik"})
_ACTIVE_OPERATION_STATES = frozenset({"queued", "running", "cancel_requested"})
_POSITION_TOLERANCE_M = 0.02
_ORIENTATION_TOLERANCE_RAD = 0.2
_DEFAULT_MAX_QPOS_STEP = 0.02


@dataclasses.dataclass(frozen=True)
class AgentCommand:
    """One validated command queued by the control channel for the main loop."""

    operation_id: str
    action: str
    arguments: dict[str, Any]


class MotionCancelled(RuntimeError):
    """The active agent motion was interrupted before its trajectory completed."""


def _operation_snapshot(runtime: RuntimeState) -> dict[str, Any] | None:
    with runtime.agent_operation_lock:
        operation = runtime.agent_operation
        return None if operation is None else dict(operation)


def _update_operation(runtime: RuntimeState, operation_id: str, **updates: Any) -> bool:
    with runtime.agent_operation_lock:
        operation = runtime.agent_operation
        if operation is None or operation["operation_id"] != operation_id:
            return False
        runtime.agent_operation = {**operation, **updates}
        return True


def queue_agent_command(runtime: RuntimeState, payload: object) -> dict[str, Any]:
    """Validate and enqueue one direct robot operation without blocking ZMQ."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "agent_command must be a JSON object"}
    action = str(payload.get("action", "")).strip()
    if action not in _DIRECT_ACTIONS:
        return {"ok": False, "error": f"unknown agent action: {action!r}"}
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "agent command arguments must be a JSON object"}

    operation_id = uuid.uuid4().hex
    with runtime.agent_operation_lock:
        active = runtime.agent_operation
        if active is not None and active["status"] in _ACTIVE_OPERATION_STATES:
            return {
                "ok": False,
                "error": "another agent operation is active",
                "operation": dict(active),
            }
        runtime.agent_operation = {
            "operation_id": operation_id,
            "action": action,
            "status": "queued",
            "result": None,
            "error": "",
        }
    runtime.agent_command_queue.put(AgentCommand(operation_id, action, dict(arguments)))
    return {"ok": True, "operation": _operation_snapshot(runtime)}


def request_agent_stop(runtime: RuntimeState) -> dict[str, Any]:
    """Request immediate cancellation of direct motion and any policy rollout."""
    operation = _operation_snapshot(runtime)
    if operation is not None and operation["status"] in _ACTIVE_OPERATION_STATES:
        _update_operation(runtime, operation["operation_id"], status="cancel_requested")
        if runtime.console_ctx is not None:
            runtime.console_ctx.session.interrupt_requested = True
    if runtime.command_queue is not None:
        runtime.command_queue.put("web:halt")
    return {
        "ok": True,
        "operation": _operation_snapshot(runtime),
        "direct_control_capable": True,
        "direct_control_available": (
            runtime.transport.get_latest_qpos() is not None and not runtime.transport.is_offline()
        ),
    }


def serialize_agent_operation(runtime: RuntimeState, operation_id: str | None = None) -> dict:
    """Return the latest direct-control operation and reject stale ids explicitly."""
    operation = _operation_snapshot(runtime)
    if operation_id and (operation is None or operation["operation_id"] != operation_id):
        return {"ok": False, "error": f"unknown operation_id: {operation_id}"}
    return {"ok": True, "operation": operation}


def _active_config(runtime: RuntimeState) -> ConfigDict:
    if runtime.active_config is not None:
        return runtime.active_config
    if runtime.console_ctx is None:
        raise RuntimeError("EVA runtime context is not ready")
    return runtime.console_ctx.config


def _latest_qpos(runtime: RuntimeState) -> np.ndarray:
    qpos = runtime.transport.get_latest_qpos()
    if qpos is None:
        raise RuntimeError("robot joint feedback is unavailable")
    vector = np.asarray(qpos, dtype=np.float32).reshape(-1)
    expected = runtime.robot.total_action_dim
    if vector.shape != (expected,):
        raise RuntimeError(f"robot qpos has shape {vector.shape}; expected ({expected},)")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("robot qpos contains non-finite values")
    return vector


def _group_slice(runtime: RuntimeState, group_name: str) -> tuple[slice, Any]:
    offset = 0
    for group in runtime.robot.actuator_groups:
        group_slice = slice(offset, offset + group.dof)
        if group.name == group_name:
            return group_slice, group
        offset += group.dof
    valid = [group.name for group in runtime.robot.actuator_groups]
    raise ValueError(f"unknown actuator group {group_name!r}; expected one of {valid}")


def _arm_eef_slice(runtime: RuntimeState, group_name: str) -> slice:
    names = [group.name for group in runtime.robot.arm_groups]
    if group_name not in names:
        raise ValueError(f"group {group_name!r} has no end effector; expected one of {names}")
    index = names.index(group_name)
    return slice(index * 8, (index + 1) * 8)


def _vector_norm(vector: np.ndarray) -> float:
    return math.sqrt(float(np.dot(vector, vector)))


def _quaternion_error(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm < 1e-8 or right_norm < 1e-8:
        return math.inf
    left = left / left_norm
    right = right / right_norm
    return 2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))


def _solve_eef_target(
    config: ConfigDict,
    runtime: RuntimeState,
    group_name: str,
    position: Any,
    quaternion_wxyz: Any,
    gripper: Any = None,
    frame: Any = None,
) -> tuple[np.ndarray, dict[str, float]]:
    current_qpos = _latest_qpos(runtime)
    configured_frame = str(getattr(config.robot, "eef_reference_frame", ""))
    if frame not in (None, "", configured_frame):
        raise ValueError(f"frame must be the configured EEF frame {configured_frame!r}")
    target_position = np.asarray(position, dtype=np.float64)
    target_quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if target_position.shape != (3,):
        raise ValueError(f"position must contain 3 values, got {target_position.shape}")
    if not np.all(np.isfinite(target_position)):
        raise ValueError("position must contain only finite values")
    if (
        target_quaternion.shape != (4,)
        or not np.all(np.isfinite(target_quaternion))
        or _vector_norm(target_quaternion) < 1e-8
    ):
        raise ValueError("quaternion_wxyz must contain a non-zero 4D quaternion")
    target_quaternion /= _vector_norm(target_quaternion)

    solver = ensure_ik_solver(config, runtime)
    current_eef = np.asarray(solver.fk_chunk(current_qpos)[0], dtype=np.float64)
    eef_slice = _arm_eef_slice(runtime, group_name)
    target_eef = current_eef.copy()
    target_eef[eef_slice.start : eef_slice.start + 3] = target_position
    target_eef[eef_slice.start + 3 : eef_slice.start + 7] = target_quaternion
    if gripper is not None:
        target_gripper = float(gripper)
        if not math.isfinite(target_gripper):
            raise ValueError("gripper must be finite")
        target_eef[eef_slice.start + 7] = target_gripper

    solved = np.asarray(
        solver.solve_chunk(np.stack([current_eef, target_eef]), seed_qpos=current_qpos)[-1],
        dtype=np.float32,
    )
    if solved.shape != current_qpos.shape or not np.all(np.isfinite(solved)):
        raise RuntimeError("IK returned an invalid qpos vector")
    group_slice, _ = _group_slice(runtime, group_name)
    keep = np.ones(len(current_qpos), dtype=bool)
    keep[group_slice] = False
    solved[keep] = current_qpos[keep]

    reached = np.asarray(solver.fk_chunk(solved)[0], dtype=np.float64)[eef_slice]
    position_error = _vector_norm(reached[:3] - target_position)
    orientation_error = _quaternion_error(reached[3:7], target_quaternion)
    if position_error > _POSITION_TOLERANCE_M:
        raise RuntimeError(
            f"IK position error {position_error:.4f} m exceeds {_POSITION_TOLERANCE_M:.4f} m"
        )
    if orientation_error > _ORIENTATION_TOLERANCE_RAD:
        raise RuntimeError(
            f"IK orientation error {orientation_error:.4f} rad exceeds "
            f"{_ORIENTATION_TOLERANCE_RAD:.4f} rad"
        )
    return solved, {
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
    }


def _execute_target(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    target_qpos: np.ndarray,
    duration_s: float | None,
) -> dict[str, Any]:
    current_qpos = _latest_qpos(runtime)
    if not np.all(np.isfinite(target_qpos)):
        raise ValueError("target qpos must contain only finite values")
    max_step = float(config.inference_cfg.get("manual_max_qpos_step", _DEFAULT_MAX_QPOS_STEP))
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("inference_cfg.manual_max_qpos_step must be positive")
    gripper_mask = np.asarray(runtime.robot.gripper_mask, dtype=bool)
    joint_delta = np.abs(target_qpos - current_qpos)[~gripper_mask]
    delta_steps = max(1, int(np.ceil(float(np.max(joint_delta)) / max_step)))
    rate_hz = int(config.inference_cfg.publish_rate)
    if rate_hz <= 0:
        raise ValueError("inference_cfg.publish_rate must be positive")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s < 0):
        raise ValueError("duration_s must be a finite non-negative value")
    duration_steps = 0 if duration_s is None else int(np.ceil(duration_s * rate_hz))
    steps = max(delta_steps, duration_steps, 1)
    trajectory = build_linear_trajectory(current_qpos, target_qpos, steps + 1)[1:]
    trajectory[:, gripper_mask] = target_qpos[gripper_mask]
    completed = publish_action_chunk(
        config,
        runtime,
        trajectory,
        OutputTarget.REAL,
        session=session,
    )
    if not completed:
        raise MotionCancelled("agent motion was interrupted")
    return {
        "target_qpos": target_qpos.astype(float).tolist(),
        "steps": steps,
        "duration_s": steps / max(rate_hz, 1),
        "motion_mode": "ik_joint_interpolation",
        "collision_aware": False,
    }


def _move_eef(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    arguments: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    target_qpos, residual = _solve_eef_target(
        config,
        runtime,
        str(arguments.get("group", "")),
        arguments.get("position"),
        arguments.get("quaternion_wxyz"),
        arguments.get("gripper"),
        arguments.get("frame"),
    )
    result: dict[str, Any] = {
        "target_qpos": target_qpos.astype(float).tolist(),
        "residual": residual,
        "motion_mode": "ik_joint_interpolation",
        "collision_aware": False,
    }
    if execute:
        result.update(
            _execute_target(config, runtime, session, target_qpos, arguments.get("duration_s"))
        )
    return result


def _move_joints(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    group_slice, group = _group_slice(runtime, str(arguments.get("group", "")))
    positions = np.asarray(arguments.get("positions"), dtype=np.float32)
    if positions.shape != (group.dof,):
        raise ValueError(f"positions for {group.name!r} must have {group.dof} values")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values")
    target = _latest_qpos(runtime).copy()
    target[group_slice] = positions
    return _execute_target(config, runtime, session, target, arguments.get("duration_s"))


def _set_gripper(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    group_slice, group = _group_slice(runtime, str(arguments.get("group", "")))
    if group.gripper_index is None:
        raise ValueError(f"actuator group {group.name!r} has no gripper")
    state = str(arguments.get("state", "")).lower()
    if state not in {"open", "close"}:
        raise ValueError("gripper state must be 'open' or 'close'")
    target = _latest_qpos(runtime).copy()
    value = config.robot.gripper_open if state == "open" else config.robot.gripper_close
    target[group_slice.start + group.gripper_index] = float(value)
    result = _execute_target(config, runtime, session, target, duration_s=1.0)
    result.update({"group": group.name, "gripper_state": state, "gripper_value": float(value)})
    return result


def handle_agent_command(
    command: AgentCommand,
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
) -> None:
    """Execute one queued operation on the EVA main thread and publish its result."""
    operation = _operation_snapshot(runtime)
    if operation is None or operation["operation_id"] != command.operation_id:
        return
    if operation["status"] == "cancel_requested":
        _update_operation(runtime, command.operation_id, status="cancelled")
        session.interrupt_requested = False
        return
    teleop_execution = runtime.teleop_execution
    if (
        runtime.collection_teleop_armed
        or runtime.collection_teleop_active
        or runtime.rollout_intervention_active
        or bool(getattr(teleop_execution, "active", False))
    ):
        _update_operation(
            runtime,
            command.operation_id,
            status="failed",
            error="operator teleoperation currently owns robot control",
        )
        return

    # Direct agent control always preempts policy execution but never teleoperation.
    if session.status is SessionStatus.RUNNING and session.mode in {
        SessionMode.REAL,
        SessionMode.SIM,
    }:
        set_status(session, SessionStatus.READY, reason="coding agent direct-control takeover")
        set_phase(runtime, "ready")
    reset_infer_strategy(runtime)
    session.pending_real_chunk = None
    session.follow_human_gripper = False
    session.gripper_locks.clear()
    session.interrupt_requested = False
    _update_operation(runtime, command.operation_id, status="running")

    try:
        if command.action == "move_eef":
            result = _move_eef(config, runtime, session, command.arguments, execute=True)
        elif command.action == "solve_ik":
            result = _move_eef(config, runtime, session, command.arguments, execute=False)
        elif command.action == "move_joints":
            result = _move_joints(config, runtime, session, command.arguments)
        else:
            result = _set_gripper(config, runtime, session, command.arguments)
    except MotionCancelled as error:
        _update_operation(
            runtime,
            command.operation_id,
            status="cancelled",
            error=str(error),
        )
    except Exception as error:
        _update_operation(
            runtime,
            command.operation_id,
            status="failed",
            error=str(error),
        )
    else:
        _update_operation(
            runtime,
            command.operation_id,
            status="succeeded",
            result=result,
            error="",
        )


def serialize_agent_status(runtime: RuntimeState) -> dict[str, Any]:
    """Describe direct-control and optional-policy capabilities for a coding agent."""
    config = _active_config(runtime)
    qpos = runtime.transport.get_latest_qpos()
    eef_capable = (
        runtime.ik_solver is not None
        or type(runtime.robot).build_kinematics is not Robot.build_kinematics
    )
    direct_ready = qpos is not None and not runtime.transport.is_offline()
    modes = ["direct"] if direct_ready else []
    if runtime.policy is not None:
        modes.append("policy")
    disabled_cameras = set(config.transport.get("disabled_cameras", []))
    return {
        "ok": True,
        "robot": runtime.robot.name,
        "transport": config.transport.type,
        "direct_control_capable": True,
        "direct_control_available": direct_ready,
        "eef_control_capable": eef_capable,
        "eef_control_available": direct_ready and eef_capable,
        "actuator_groups": [
            {
                "name": group.name,
                "dof": group.dof,
                "eef": group in runtime.robot.arm_groups,
                "gripper": group.gripper_index is not None,
            }
            for group in runtime.robot.actuator_groups
        ],
        "cameras": [
            camera.observation_key
            for camera in runtime.robot.observation_schema.cameras
            if camera.observation_key not in disabled_cameras
        ],
        "direct_control_tools": [
            "robot_get_state",
            "camera_capture",
            "robot_solve_ik",
            "robot_move_eef",
            "robot_move_joints",
            "robot_set_gripper",
            "robot_stop",
        ],
        "control_modes": modes,
        "motion": {"mode": "ik_joint_interpolation", "collision_aware": False},
        "policy": {
            "configured_type": config.policy.type,
            "connected": runtime.policy is not None,
            "metadata": runtime.policy_metadata or {},
            "error": runtime.last_policy_error,
        },
        "operation": _operation_snapshot(runtime),
        "guidance": (
            "The coding agent can control the robot directly without a policy model. "
            "Use direct tools whenever the model is offline or direct correction is useful."
        ),
    }


def serialize_robot_state(runtime: RuntimeState) -> dict[str, Any]:
    """Return current joint feedback and EEF state when the IK solver is initialized."""
    qpos = _latest_qpos(runtime)
    groups = {}
    offset = 0
    for group in runtime.robot.actuator_groups:
        groups[group.name] = {
            "qpos": qpos[offset : offset + group.dof].astype(float).tolist(),
            "dof": group.dof,
            "has_gripper": group.gripper_index is not None,
        }
        offset += group.dof
    eef = None
    if runtime.ik_solver is not None:
        eef = np.asarray(runtime.ik_solver.fk_chunk(qpos)[0], dtype=float).tolist()
    return {
        "ok": True,
        "qpos": qpos.astype(float).tolist(),
        "groups": groups,
        "eef": eef,
        "eef_layout": "per arm: xyz + quaternion_wxyz + gripper",
        "direct_control_available": not runtime.transport.is_offline(),
    }


def serialize_camera(runtime: RuntimeState, camera_name: str) -> dict[str, Any]:
    """Capture one camera frame and return a JPEG payload for the MCP adapter."""
    config = _active_config(runtime)
    ctx = runtime.console_ctx
    if ctx is None:
        raise RuntimeError("EVA runtime context is not ready")
    reader = ctx.obs_reader or runtime.transport
    frame = reader.get_frame()
    if frame is None:
        raise RuntimeError("camera frame is unavailable")
    image = frame.images.get(camera_name)
    if image is None:
        raise ValueError(f"unknown camera {camera_name!r}; expected one of {sorted(frame.images)}")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if not config.transport.convert_bgr_to_rgb and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError(f"failed to encode camera {camera_name!r}")
    return {
        "ok": True,
        "camera": camera_name,
        "mime_type": "image/jpeg",
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }
