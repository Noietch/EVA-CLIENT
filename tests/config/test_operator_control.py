from __future__ import annotations

import queue

import numpy as np

from core.app.operator_control import resolve_operator_event
from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot


class _Transport:
    def supports_collection(self) -> bool:
        return True


class _Logger:
    def __init__(self, *, collecting: bool = False, active_frames: int = 0) -> None:
        self._collecting = collecting
        self.active_frame_count = active_frames

    @property
    def has_active_episode(self) -> bool:
        return self._collecting

    @property
    def is_collection_enabled(self) -> bool:
        return True

    def is_queue_full(self) -> bool:
        return False


def _robot() -> Robot:
    return Robot(
        name="fake",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam"),),
            state_composition=("arm",),
        ),
    )


def _runtime() -> RuntimeState:
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    runtime.command_queue = queue.Queue()
    return runtime


def _config() -> ConfigDict:
    return ConfigDict(
        operator_control=ConfigDict(enabled=True, button_topic="/eva/operator_button"),
        collection=ConfigDict(schema=ConfigDict(columns={"qpos": "state"})),
        rollout=ConfigDict(intervention=ConfigDict(enabled=True)),
    )


def test_operator_start_in_collect_ready_starts_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.READY)
    runtime.episode_logger = _Logger(collecting=False)

    command = resolve_operator_event(_config(), runtime, session, intent="start", source="ui")

    assert command == "web:collect_start"
    assert session.last_error == ""


def test_operator_accept_in_collect_recording_stops_and_saves_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.RUNNING)
    runtime.episode_logger = _Logger(collecting=True)

    command = resolve_operator_event(_config(), runtime, session, intent="accept", source="ui")

    assert command == "web:collect_stop"
    assert session.last_error == ""


def test_operator_cancel_in_collect_recording_cancels_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.RUNNING)
    runtime.episode_logger = _Logger(collecting=True)

    command = resolve_operator_event(_config(), runtime, session, intent="cancel", source="cat")

    assert command == "web:collect_cancel"
    assert session.last_error == ""


def test_operator_start_while_policy_running_enters_intervention():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING)

    command = resolve_operator_event(_config(), runtime, session, intent="start", source="cat")

    assert command == "web:halt"


def test_operator_accept_in_intervention_resumes_policy():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(_config(), runtime, session, intent="accept", source="cat")

    assert command == "web:run"


def test_operator_cancel_in_intervention_abandons_segment():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(_config(), runtime, session, intent="cancel", source="cat")

    assert command == "web:rollout_intervention_abandon"


def test_operator_accept_when_rollout_ready_saves_rollout():
    runtime = _runtime()
    runtime.rollout_save_ready = True
    runtime.rollout_episode_logger = _Logger(active_frames=3)
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(_config(), runtime, session, intent="accept", source="ui")

    assert command == "web:rollout_save"


def test_x_button_maps_to_cancel_when_collect_recording():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.RUNNING)
    runtime.episode_logger = _Logger(collecting=True)

    command = resolve_operator_event(_config(), runtime, session, button="x", source="cat")

    assert command == "web:collect_cancel"


def test_operator_gripper_button_uses_standard_semantic_event():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(
        _config(), runtime, session, button="right_gripper_close", source="hardware"
    )

    assert command == "web:gripper:r:close:0"
    assert session.last_error == ""


def test_x_button_maps_to_start_when_collect_ready():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.READY)
    runtime.episode_logger = _Logger(collecting=False)

    command = resolve_operator_event(_config(), runtime, session, button="x", source="cat")

    assert command == "web:collect_start"


def test_y_button_maps_to_accept():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(_config(), runtime, session, button="y", source="cat")

    assert command == "web:run"


def test_x_button_maps_to_halt_while_policy_running():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING)

    command = resolve_operator_event(_config(), runtime, session, button="x", source="cat")

    assert command == "web:halt"
    assert session.last_error == ""


def test_x_button_maps_to_abandon_when_intervention_active():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(_config(), runtime, session, button="x", source="cat")

    assert command == "web:rollout_intervention_abandon"
    assert session.last_error == ""


def test_y_button_maps_to_rollout_save_when_save_ready():
    runtime = _runtime()
    runtime.rollout_save_ready = True
    runtime.rollout_episode_logger = _Logger(active_frames=3)
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(_config(), runtime, session, button="y", source="cat")

    assert command == "web:rollout_save"
    assert session.last_error == ""


def test_gripper_buttons_map_to_immediate_unlocked_gripper_commands():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(
        _config(), runtime, session, button="left_gripper_open", source="cat"
    )

    assert command == "web:gripper:l:open:0"
    assert session.last_error == ""


def test_right_gripper_close_button_maps_to_right_close_command():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_event(
        _config(), runtime, session, button="right_gripper_close", source="cat"
    )

    assert command == "web:gripper:r:close:0"
    assert session.last_error == ""


def test_invalid_operator_event_sets_explicit_error():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.SELECT, status=SessionStatus.UNSET)

    command = resolve_operator_event(_config(), runtime, session, intent="accept", source="ui")

    assert command is None
    assert session.last_error == "Operator action 'accept' is not valid in mode=select status=unset"
