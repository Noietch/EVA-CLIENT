"""Operator action routing shared by HTTP and externally normalized input events."""

from __future__ import annotations

import dataclasses
import logging

from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class OperatorEvent:
    """Generic operator event after backend-specific normalization."""

    intent: str | None = None
    gripper_side: str | None = None
    gripper_command: str | None = None


def _collection_recording(runtime: RuntimeState, session: SessionState) -> bool:
    logger_obj = runtime.episode_logger
    return (
        session.mode is SessionMode.COLLECT
        and session.status is SessionStatus.RUNNING
        and logger_obj is not None
        and logger_obj.has_active_episode
    )


def _gripper_command(event: OperatorEvent) -> str | None:
    side = (event.gripper_side or "").strip().lower()
    command = (event.gripper_command or "").strip().lower()
    if side in {"left", "l"}:
        side_key = "l"
    elif side in {"right", "r"}:
        side_key = "r"
    else:
        return None
    if command not in {"open", "close"}:
        return None
    return f"web:gripper:{side_key}:{command}:0"


def _normalize_button_event(
    button: str,
    runtime: RuntimeState,
    session: SessionState,
) -> OperatorEvent:
    value = button.strip().lower()
    if value == "y":
        return OperatorEvent(intent="accept")
    if value == "x":
        if _collection_recording(runtime, session) or runtime.rollout_intervention_active:
            return OperatorEvent(intent="cancel")
        return OperatorEvent(intent="start")
    for side in ("left", "right", "l", "r"):
        for command in ("open", "close"):
            if value == f"{side}_gripper_{command}":
                return OperatorEvent(gripper_side=side, gripper_command=command)
    return OperatorEvent()


def _resolve_event_command(
    runtime: RuntimeState,
    session: SessionState,
    event: OperatorEvent,
) -> str | None:
    command = _gripper_command(event)
    if command is not None:
        return command

    resolved_intent = event.intent
    if resolved_intent not in {"start", "accept", "cancel"}:
        return None

    if resolved_intent == "start":
        if session.mode is SessionMode.COLLECT and session.status is not SessionStatus.RUNNING:
            return "web:collect_start"
        if session.status is SessionStatus.RUNNING and session.mode in (
            SessionMode.REAL,
            SessionMode.SIM,
        ):
            return "web:halt"
        if runtime.rollout_intervention_active:
            return "web:run"
    if resolved_intent == "accept":
        if _collection_recording(runtime, session):
            return "web:collect_stop"
        if runtime.rollout_intervention_active:
            return "web:run"
        if runtime.rollout_save_ready:
            return "web:rollout_save"
    if resolved_intent == "cancel":
        if _collection_recording(runtime, session):
            return "web:collect_cancel"
        if runtime.rollout_intervention_active:
            return "web:rollout_intervention_abandon"
    return None


def resolve_operator_event(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    *,
    intent: str | None = None,
    button: str | None = None,
    source: str = "ui",
) -> str | None:
    """Resolve a UI/CAT operator event into one existing web command.

    Args:
        config: Active config.
        runtime: Current runtime state.
        session: Current session state.
        intent: High-level UI intent: start, accept, or cancel.
        button: Physical CAT button: x or y.
        source: Human-readable event source used only for logs.

    Returns:
        A concrete ``web:*`` command for ``handle_command`` to execute, or None
        when the event is invalid in the current state.
    """
    event = OperatorEvent(intent=intent)
    if button is not None:
        event = _normalize_button_event(button, runtime, session)

    if event.intent not in {"start", "accept", "cancel", None}:
        session.last_error = f"Unsupported operator event: intent={intent!r} button={button!r}"
        return None

    command = _resolve_event_command(runtime, session, event)
    if command is None and event.intent is None:
        session.last_error = f"Unsupported operator event: intent={intent!r} button={button!r}"
        return None

    if command is None:
        resolved_intent = event.intent
        session.last_error = (
            f"Operator action {resolved_intent!r} is not valid in "
            f"mode={session.mode.value} status={session.status.value}"
        )
        logger.info(
            "[OPERATOR] source=%s intent=%s button=%s ignored mode=%s status=%s",
            source,
            resolved_intent,
            button,
            session.mode.value,
            session.status.value,
        )
        return None

    session.last_error = ""
    logger.info(
        "[OPERATOR] source=%s intent=%s button=%s resolved=%s",
        source,
        event.intent,
        button,
        command,
    )
    return command
