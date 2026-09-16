from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.app.control_channel import _handle_message
from core.app.handlers.teleop import acknowledge_teleop_event

pytestmark = pytest.mark.unit


def _runtime(*, teleop_client: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        console_ctx=SimpleNamespace(
            active_tab="debug",
            session=SimpleNamespace(
                selected_collect_task=None,
                selected_collect_set=None,
                selected_collect_task_index=None,
                last_error="",
            ),
        ),
        command_queue=queue.Queue(),
        collection_teleop_armed=False,
        current_clip_id=None,
        current_cell=None,
        teleop_client=teleop_client,
    )


def test_channel_rejects_unknown_commands_without_queueing() -> None:
    runtime = _runtime()
    reply = _handle_message(runtime, {"cmd": "web:not_a_command"})
    assert reply == {"ok": False, "error": "command not allowed: 'not_a_command'"}
    assert runtime.command_queue.empty()


def test_channel_sends_haptic_directly_to_vr_client() -> None:
    client = SimpleNamespace(send_haptic=Mock(return_value=True))
    runtime = _runtime(teleop_client=client)

    reply = _handle_message(
        runtime,
        {
            "cmd": "web:haptic",
            "hand": "right",
            "intensity": 0.35,
            "duration_ms": 70,
        },
    )

    assert reply == {
        "ok": True,
        "cmd": "web:haptic",
        "hands": ["right"],
        "intensity": 0.35,
        "duration_ms": 70.0,
    }
    client.send_haptic.assert_called_once_with(hand="right", intensity=0.35, duration_ms=70.0)
    assert runtime.command_queue.empty()


def test_channel_haptic_supports_both_hands_and_clamps_values() -> None:
    client = SimpleNamespace(send_haptic=Mock(return_value=True))
    runtime = _runtime(teleop_client=client)

    reply = _handle_message(runtime, {"cmd": "web:haptic", "hand": "all", "intensity": 2})

    assert reply["ok"] is True
    assert reply["hands"] == ["left", "right"]
    assert reply["intensity"] == 1.0
    assert reply["duration_ms"] == 80.0
    assert client.send_haptic.call_count == 2
    assert client.send_haptic.call_args_list[0].kwargs == {
        "hand": "left",
        "intensity": 1.0,
        "duration_ms": 80.0,
    }
    assert client.send_haptic.call_args_list[1].kwargs == {
        "hand": "right",
        "intensity": 1.0,
        "duration_ms": 80.0,
    }


def test_channel_haptic_accepts_positional_command_arguments() -> None:
    client = SimpleNamespace(send_haptic=Mock(return_value=True))
    runtime = _runtime(teleop_client=client)

    reply = _handle_message(runtime, {"cmd": "web:haptic:left:0.4:120"})

    assert reply["ok"] is True
    assert reply["hands"] == ["left"]
    client.send_haptic.assert_called_once_with(hand="left", intensity=0.4, duration_ms=120.0)


def test_channel_haptic_rejects_invalid_request_or_disconnected_headset() -> None:
    runtime = _runtime(teleop_client=SimpleNamespace(send_haptic=Mock(return_value=False)))
    assert _handle_message(runtime, {"cmd": "web:haptic", "hand": "middle"}) == {
        "ok": False,
        "error": "haptic hand must be 'left', 'right', or 'both'",
    }
    assert _handle_message(runtime, {"cmd": "web:haptic", "hand": "right"}) == {
        "ok": False,
        "error": "VR headset is not connected",
    }


def test_operator_acknowledgement_does_not_vibrate_without_a_haptic_request() -> None:
    client = SimpleNamespace(acknowledge_event=Mock(), send_haptic=Mock())
    runtime = _runtime(teleop_client=client)
    event = SimpleNamespace(intent="home")

    acknowledge_teleop_event(runtime, event, accepted=True, message="accepted")

    client.acknowledge_event.assert_called_once_with(event, accepted=True, message="accepted")
    client.send_haptic.assert_not_called()
