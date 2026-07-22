"""Device-independent operator action routing shared by all producers."""

from __future__ import annotations

import logging

from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict

logger = logging.getLogger(__name__)


def _collection_recording(runtime: RuntimeState, session: SessionState) -> bool:
    logger_obj = runtime.episode_logger
    return (
        session.mode is SessionMode.COLLECT
        and session.status is SessionStatus.RUNNING
        and logger_obj is not None
        and logger_obj.has_active_episode
    )


def resolve_operator_action(
    runtime: RuntimeState,
    session: SessionState,
    action: str,
    *,
    source: str = "ui",
) -> str | None:
    """Resolve one device-independent operator action into an internal command.

    Args:
        runtime: Current runtime state.
        session: Current session state.
        action: Semantic operator action emitted by UI or a hardware adapter.
        source: Human-readable producer used only for logs.

    Returns:
        A concrete ``web:*`` command, or None when the action is invalid for the
        current state.
    """
    normalized = action.strip().lower()
    command = {
        "gripper_left_open": "web:gripper:l:open:0",
        "gripper_left_close": "web:gripper:l:close:0",
        "gripper_right_open": "web:gripper:r:open:0",
        "gripper_right_close": "web:gripper:r:close:0",
    }.get(normalized)
    if command is not None:
        session.last_error = ""
        return command

    if normalized not in {
        "start",
        "accept",
        "cancel",
        "rollout_toggle",
        "rollout_reset",
        "intervention_toggle",
        "intervention_accept",
    }:
        session.last_error = f"Unsupported operator action: {normalized!r}"
        return None

    if normalized == "rollout_toggle":
        if runtime.rollout_intervention_active:
            command = None
        elif (
            runtime.rollout_episode_logger is not None
            and runtime.rollout_episode_logger.has_active_episode
        ) or (
            session.status is SessionStatus.RUNNING
            and session.mode in (SessionMode.REAL, SessionMode.SIM)
        ):
            command = "web:rollout_save"
        elif (
            session.status is SessionStatus.READY
            and session.mode in (SessionMode.REAL, SessionMode.SIM)
            and session.is_setup_done
        ):
            command = "web:run"
        else:
            command = None
    elif normalized == "rollout_reset":
        command = (
            "web:console_reset" if session.mode in (SessionMode.REAL, SessionMode.SIM) else None
        )
    elif normalized == "intervention_toggle":
        if runtime.rollout_intervention_active:
            command = "web:rollout_intervention_abandon"
        elif session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        ):
            command = "web:halt"
        else:
            command = None
    elif normalized == "intervention_accept":
        command = "web:run" if runtime.rollout_intervention_active else None
    elif normalized == "start":
        if session.mode is SessionMode.COLLECT and session.status is not SessionStatus.RUNNING:
            command = "web:collect_start"
        elif session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        ):
            command = "web:halt"
        elif runtime.rollout_intervention_active:
            command = "web:run"
        else:
            command = None
    elif normalized == "accept":
        if _collection_recording(runtime, session):
            command = "web:collect_stop"
        elif runtime.rollout_intervention_active:
            command = "web:run"
        elif runtime.rollout_save_ready:
            command = "web:rollout_save"
        else:
            command = None
    else:
        if _collection_recording(runtime, session):
            command = "web:collect_cancel"
        elif runtime.rollout_intervention_active:
            command = "web:rollout_intervention_abandon"
        else:
            command = None

    if command is None:
        session.last_error = (
            f"Operator action {normalized!r} is not valid in "
            f"mode={session.mode.value} status={session.status.value}"
        )
        logger.info(
            "[OPERATOR] source=%s action=%s ignored mode=%s status=%s",
            source,
            normalized,
            session.mode.value,
            session.status.value,
        )
        return None

    session.last_error = ""
    logger.info(
        "[OPERATOR] source=%s action=%s resolved=%s",
        source,
        normalized,
        command,
    )
    return command


def maybe_start_operator_action_listener(
    config: ConfigDict,
    runtime: RuntimeState,
) -> None:
    """Subscribe to semantic operator actions and enqueue internal web commands."""
    operator_cfg = config.get("operator_control") or {}
    if not operator_cfg.get("enabled", False):
        return
    if config.transport.type != "ros2":
        logger.warning("operator_control requires ros2 transport, got %s", config.transport.type)
        return
    if runtime.command_queue is None:
        raise RuntimeError("operator_control requires runtime.command_queue")
    command_queue = runtime.command_queue

    from std_msgs.msg import String

    from transport.ros2 import get_ros2_runtime

    topic = str(operator_cfg.get("action_topic", "/eva/operator_action"))
    ros_runtime = get_ros2_runtime(config.transport.node_name)

    def on_action(msg: String) -> None:
        action = msg.data.strip().lower()
        if not action:
            logger.warning("[OPERATOR] ignored empty operator event")
            return
        command_queue.put(f"web:operator_action:{action}:external")

    ros_runtime.node.create_subscription(String, topic, on_action, 10)
    logger.info("[OPERATOR] listening for semantic actions on %s", topic)
