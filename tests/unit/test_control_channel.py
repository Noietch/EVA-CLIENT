from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.app.control_channel import _handle_message

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
