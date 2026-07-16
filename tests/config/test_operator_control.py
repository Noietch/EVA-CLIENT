from __future__ import annotations

import queue
from pathlib import Path

import numpy as np

from core.app.operator_control import resolve_operator_action
from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
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


def test_core_operator_control_contains_only_semantic_actions():
    source = (
        Path(__file__).resolve().parents[2] / "src" / "core" / "app" / "operator_control.py"
    ).read_text()

    assert "operator_button" not in source
    assert "button_topic" not in source
    assert "/eva/operator_button" not in source
    assert 'button_value == "a"' not in source
    assert 'button_value == "b"' not in source
    assert 'button_value in {"x", "y"}' not in source


def test_operator_start_in_collect_ready_starts_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.READY)
    runtime.episode_logger = _Logger(collecting=False)

    command = resolve_operator_action(runtime, session, "start", source="ui")

    assert command == "web:collect_start"
    assert session.last_error == ""


def test_operator_accept_in_collect_recording_stops_and_saves_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.RUNNING)
    runtime.episode_logger = _Logger(collecting=True)

    command = resolve_operator_action(runtime, session, "accept", source="ui")

    assert command == "web:collect_stop"
    assert session.last_error == ""


def test_operator_cancel_in_collect_recording_cancels_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.RUNNING)
    runtime.episode_logger = _Logger(collecting=True)

    command = resolve_operator_action(runtime, session, "cancel", source="ui")

    assert command == "web:collect_cancel"
    assert session.last_error == ""


def test_operator_start_while_policy_running_enters_intervention():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING)

    command = resolve_operator_action(runtime, session, "start", source="ui")

    assert command == "web:halt"


def test_operator_accept_in_intervention_resumes_policy():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "accept", source="ui")

    assert command == "web:run"


def test_operator_cancel_in_intervention_abandons_segment():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "cancel", source="ui")

    assert command == "web:rollout_intervention_abandon"


def test_operator_accept_when_rollout_ready_saves_rollout():
    runtime = _runtime()
    runtime.rollout_save_ready = True
    runtime.rollout_episode_logger = _Logger(active_frames=3)
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "accept", source="ui")

    assert command == "web:rollout_save"


def test_intervention_toggle_does_not_control_collection():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.RUNNING)
    runtime.episode_logger = _Logger(collecting=True)

    command = resolve_operator_action(runtime, session, "intervention_toggle", source="external")

    assert command is None
    assert "is not valid" in session.last_error


def test_gripper_action_maps_to_immediate_command():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "gripper_right_close", source="external")

    assert command == "web:gripper:r:close:0"
    assert session.last_error == ""


def test_intervention_accept_resumes_policy():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "intervention_accept", source="external")

    assert command == "web:run"


def test_intervention_toggle_halts_running_policy():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING)

    command = resolve_operator_action(runtime, session, "intervention_toggle", source="external")

    assert command == "web:halt"
    assert session.last_error == ""


def test_intervention_toggle_abandons_active_intervention():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "intervention_toggle", source="external")

    assert command == "web:rollout_intervention_abandon"
    assert session.last_error == ""


def test_intervention_accept_does_not_save_complete_rollout():
    runtime = _runtime()
    runtime.rollout_save_ready = True
    runtime.rollout_episode_logger = _Logger(active_frames=3)
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "intervention_accept", source="external")

    assert command is None
    assert "is not valid" in session.last_error


def test_rollout_toggle_starts_ready_rollout():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)
    session.is_setup_done = True

    command = resolve_operator_action(runtime, session, "rollout_toggle", source="external")

    assert command == "web:run"


def test_rollout_toggle_saves_running_rollout():
    runtime = _runtime()
    runtime.rollout_episode_logger = _Logger(collecting=True, active_frames=3)
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING)

    command = resolve_operator_action(runtime, session, "rollout_toggle", source="external")

    assert command == "web:rollout_save"


def test_rollout_toggle_does_not_decide_active_intervention():
    runtime = _runtime()
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "rollout_toggle", source="external")

    assert command is None
    assert "is not valid" in session.last_error


def test_rollout_reset_resets_and_discards_rollout():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING)

    command = resolve_operator_action(runtime, session, "rollout_reset", source="external")

    assert command == "web:console_reset"


def test_left_gripper_action_maps_to_immediate_unlocked_command():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "gripper_left_open", source="external")

    assert command == "web:gripper:l:open:0"
    assert session.last_error == ""


def test_right_gripper_action_maps_to_right_close_command():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.READY)

    command = resolve_operator_action(runtime, session, "gripper_right_close", source="external")

    assert command == "web:gripper:r:close:0"
    assert session.last_error == ""


def test_invalid_operator_event_sets_explicit_error():
    runtime = _runtime()
    session = SessionState(mode=SessionMode.SELECT, status=SessionStatus.UNSET)

    command = resolve_operator_action(runtime, session, "accept", source="ui")

    assert command is None
    assert session.last_error == "Operator action 'accept' is not valid in mode=select status=unset"
