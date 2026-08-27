from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from core.app.handlers.teleop import TeleopExecutionState
from core.app.operator_control import handle_teleop_operator_event
from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict
from teleop_client.base import TeleopOperatorEvent, TeleopStatus


class _Logger:
    def __init__(self) -> None:
        self.active = False

    @property
    def has_active_episode(self) -> bool:
        return self.active


class _Client:
    def __init__(self) -> None:
        self.resets = 0
        self.acks: list[tuple[str, bool, str]] = []

    def reset(self, *, require_neutral: bool = False) -> None:
        self.resets += 1

    def status(self) -> TeleopStatus:
        return TeleopStatus(source_type="test", connected=True, neutral=True)

    def acknowledge_event(self, event, *, accepted: bool, message: str) -> None:
        self.acks.append((event.intent, accepted, message))


def _runtime() -> tuple[RuntimeState, SessionState]:
    runtime = SimpleNamespace(
        teleop_client=_Client(),
        teleop_execution=TeleopExecutionState(control_source="client", active=True),
        collection_teleop_armed=True,
        collection_teleop_active=True,
        episode_logger=_Logger(),
    )
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.READY)
    return cast(RuntimeState, runtime), session


def _event(event_id: int, intent: str) -> TeleopOperatorEvent:
    return TeleopOperatorEvent("test", event_id, intent, float(event_id))


def _dispatch(command, _config, runtime, session) -> None:
    if command == "web:collect_start":
        runtime.episode_logger.active = True
        session.status = SessionStatus.RUNNING
    elif command in {"web:collect_stop", "web:collect_cancel"}:
        runtime.episode_logger.active = False
        session.status = SessionStatus.READY
    elif command == "web:collect_arm:off":
        runtime.collection_teleop_armed = False
        runtime.collection_teleop_active = False
        runtime.teleop_execution.active = False
        session.mode = SessionMode.SELECT
    elif command == "web:collect_arm:on":
        runtime.collection_teleop_armed = True
        runtime.collection_teleop_active = True
        runtime.teleop_execution.active = True
        session.mode = SessionMode.COLLECT
    elif command == "web:collect_home":
        session.last_error = ""


def test_operator_events_toggle_recording() -> None:
    runtime, session = _runtime()
    config = ConfigDict(collection=ConfigDict(teleop=ConfigDict()))

    handle_teleop_operator_event(
        _event(0, "record_toggle"), config, runtime, session, dispatch=_dispatch
    )
    handle_teleop_operator_event(
        _event(1, "record_toggle"), config, runtime, session, dispatch=_dispatch
    )

    assert runtime.teleop_client.acks[0][:2] == ("record_toggle", True)
    assert runtime.teleop_client.acks[1][:2] == ("record_toggle", True)


def test_dispatch_failure_is_rejected_and_acknowledged_once() -> None:
    runtime, session = _runtime()
    config = ConfigDict(collection=ConfigDict(teleop=ConfigDict()))

    def fail_dispatch(*_args) -> None:
        raise RuntimeError("queue unavailable")

    handle_teleop_operator_event(
        _event(3, "record_toggle"), config, runtime, session, dispatch=fail_dispatch
    )

    assert len(runtime.teleop_client.acks) == 1
    intent, accepted, message = runtime.teleop_client.acks[0]
    assert intent == "record_toggle"
    assert accepted is False
    assert "queue unavailable" in message
    assert session.last_error == message


def test_arm_toggle_uses_the_shared_collect_arm_command_while_inactive() -> None:
    runtime, session = _runtime()
    config = ConfigDict(collection=ConfigDict(teleop=ConfigDict()))

    handle_teleop_operator_event(
        _event(4, "arm_toggle"), config, runtime, session, dispatch=_dispatch
    )
    assert runtime.teleop_client.acks[-1][:2] == ("arm_toggle", True)
    assert not runtime.collection_teleop_armed

    handle_teleop_operator_event(
        _event(5, "arm_toggle"), config, runtime, session, dispatch=_dispatch
    )
    assert runtime.teleop_client.acks[-1][:2] == ("arm_toggle", True)
    assert runtime.collection_teleop_armed


def test_home_uses_the_shared_collect_home_command_when_disarmed() -> None:
    runtime, session = _runtime()
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    runtime.teleop_execution.active = False
    session.mode = SessionMode.SELECT
    config = ConfigDict(collection=ConfigDict(teleop=ConfigDict()))
    commands: list[str] = []

    def dispatch(command, config_arg, runtime_arg, session_arg):
        commands.append(command)
        _dispatch(command, config_arg, runtime_arg, session_arg)

    handle_teleop_operator_event(_event(6, "home"), config, runtime, session, dispatch=dispatch)

    assert commands == ["web:collect_home"]
    assert runtime.teleop_client.acks[-1][:2] == ("home", True)
