from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

import core.app.console.server as console_server
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


def test_channel_status_includes_recorder_history(monkeypatch) -> None:
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
    calls = []

    def _serialize_status(ctx, *, include_history=False):
        calls.append((ctx, include_history))
        return {"eval_recorder": {"episodes": [{"clip_id": "trial-0001"}]}}

    monkeypatch.setattr(console_server, "_serialize_status", _serialize_status)

    reply = _handle_message(runtime, {"query": "status"})

    assert reply["data"]["eval_recorder"]["episodes"] == [
        {"clip_id": "trial-0001"}
    ]
    assert calls == [(runtime.console_ctx, True)]


def test_channel_start_binds_eval_sim_metadata_before_enqueueing_start() -> None:
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
    sim_metadata = {
        "schema_version": 3,
        "scene": {"scene_id": "scene-01"},
        "task_item": {"task_item_id": "task-01"},
    }

    reply = _handle_message(
        runtime,
        {
            "cmd": "web:start",
            "clip_id": "trial-0001",
            "prompt": "pick the block",
            "trial": 1,
            "sim_metadata": sim_metadata,
        },
    )

    assert reply == {"ok": True, "cmd": "web:start"}
    assert runtime.current_clip_id == "trial-0001"
    assert runtime.current_cell == {
        "prompt": "pick the block",
        "trial": 1,
        "sim_metadata": sim_metadata,
    }
    assert runtime.command_queue.get_nowait() == "web:start"


def test_channel_collect_start_binds_sim_metadata_before_enqueueing_collect() -> None:
    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(
            active_tab="collect",
            session=SimpleNamespace(
                selected_collect_task="pick the block",
                collection_sim_metadata=None,
            ),
        ),
        command_queue=queue.Queue(),
    )
    sim_metadata = {
        "schema_version": 4,
        "scene": {"scene_id": "scene-01"},
        "task_item": {"task_item_id": "task-01"},
    }

    reply = _handle_message(
        runtime,
        {"cmd": "web:collect_start", "sim_metadata": sim_metadata},
    )

    assert reply == {"ok": True, "cmd": "web:collect_start"}
    assert runtime.console_ctx.session.collection_sim_metadata == sim_metadata
    assert runtime.command_queue.get_nowait() == "web:collect_start"


def test_channel_collect_start_preserves_slot_bound_sim_metadata() -> None:
    sim_metadata = {"schema_version": 4, "scene": {"scene_id": "scene-01"}}
    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(
            active_tab="collect",
            session=SimpleNamespace(
                selected_collect_task="pick the block",
                collection_sim_metadata=sim_metadata,
            ),
        ),
        command_queue=queue.Queue(),
    )

    reply = _handle_message(runtime, {"cmd": "web:collect_start"})

    assert reply == {"ok": True, "cmd": "web:collect_start"}
    assert runtime.console_ctx.session.collection_sim_metadata is sim_metadata
    assert runtime.command_queue.get_nowait() == "web:collect_start"
