from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from core.app.control_channel import _handle_message

pytestmark = pytest.mark.unit


def test_channel_rejects_unknown_commands_without_queueing() -> None:
    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(
            active_tab="debug",
            session=SimpleNamespace(selected_collect_task=None, last_error=""),
        ),
        command_queue=queue.Queue(),
        collection_teleop_armed=False,
        current_clip_id=None,
        current_cell=None,
    )
    reply = _handle_message(runtime, {"cmd": "web:not_a_command"})
    assert reply == {"ok": False, "error": "command not allowed: 'not_a_command'"}
    assert runtime.command_queue.empty()
