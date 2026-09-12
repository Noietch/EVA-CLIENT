"""Review navigation stays separate from robot motion and uses reliable intents."""
from unittest.mock import Mock

import pytest

from core.app.operator_control import handle_teleop_operator_event
from core.app.state import RuntimeState, SessionState
from examples.input_sources.vr_webxr.node import OperatorEventMapper, normalize_controller
from teleop_client.base import TeleopOperatorEvent
from teleop_client.vr.client import parse_operator_event

pytestmark = pytest.mark.integration


def test_stick_deadzone_repeat_and_y_release():
    mapper = OperatorEventMapper()

    def update(at, axes=(0.0, 0.0), y=False):
        return mapper.update(
            session_id="review", client_time_ms=at,
            controllers={"left": {"thumbstick": axes}},
            face_buttons={"left": {"secondary": y}, "right": {}},
        )

    assert update(0, (0.2, -0.3)) == ()
    assert update(10, (0.8, 0))[0]["intent"] == "review_right"
    assert update(409, (0.8, 0)) == ()
    assert update(410, (0.8, 0))[0]["intent"] == "review_right"
    assert update(500, (0, -0.9))[0]["intent"] == "review_up"
    assert update(510) == ()
    assert update(520, y=True) == ()
    events = update(530)
    assert events[0]["intent"] == "intervention_toggle"
    assert update(540) == ()
    assert parse_operator_event(events[0]).intent == "intervention_toggle"
    mapper.reset()
    event = update(550, (-1, 0))[0]
    assert event["event_id"] == 0
    assert parse_operator_event(event).intent == "review_left"


def test_normalized_stick_uses_xr_axes_and_ignores_lost_tracking():
    raw = {"valid": True, "position": [0, 0, 0],
           "orientation_xyzw": [0, 0, 0, 1], "axes": [0, 0, 0.8, -0.9]}
    controller, _ = normalize_controller("left", raw)
    assert controller["thumbstick"] == [0.8, -0.9]
    controller, _ = normalize_controller("left", {**raw, "valid": False})
    assert controller["thumbstick"] == [0, 0]
    controller, _ = normalize_controller("left", {**raw, "axes": [-1, 0]})
    assert controller["thumbstick"] == [-1, 0]


def test_review_routes_without_arming_and_preserves_rl_boundary():
    runtime = RuntimeState(robot=Mock(), transport=Mock())
    runtime.teleop_client = Mock()
    runtime.teleop_execution = Mock()
    dispatch = Mock()
    for index, intent in enumerate(("review_down", "intervention_toggle", "review_select")):
        handle_teleop_operator_event(
            TeleopOperatorEvent("review", index, intent, 0), {}, runtime, SessionState(),
            dispatch=dispatch,
        )
    assert [item["action"] for item in runtime.collection_review_events] == [
        "down", "toggle_qc", "select"
    ]
    assert runtime.collection_teleop_armed is False
    dispatch.assert_not_called()
    runtime.rl_active = True
    handle_teleop_operator_event(
        TeleopOperatorEvent("review", 2, "review_up", 0), {}, runtime, SessionState(),
        dispatch=dispatch,
    )
    assert len(runtime.collection_review_events) == 3
    assert runtime.teleop_client.acknowledge_event.call_args.kwargs["accepted"] is False


def test_thumbstick_click_selects_once_and_suppresses_direction():
    mapper = OperatorEventMapper()
    raw = {"valid": True, "position": [0, 0, 0],
           "orientation_xyzw": [0, 0, 0, 1], "axes": [0, 0, 1, 0],
           "buttons": [{}, {}, {}, {"pressed": True}]}
    controller, face = normalize_controller("left", raw)
    assert face["thumbstick"] is True

    def update(at, pressed):
        return mapper.update(
            session_id="click", client_time_ms=at,
            controllers={"left": controller},
            face_buttons={"left": {**face, "thumbstick": pressed}, "right": {}},
        )

    events = update(0, True)
    assert [event["intent"] for event in events] == ["review_select"]
    assert parse_operator_event(events[0]).intent == "review_select"
    assert update(1000, True) == ()
    update(1010, False)
    assert update(1020, True)[0]["intent"] == "review_select"
