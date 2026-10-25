from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.app.console.server as console_server
from core.app.control_channel import _handle_message

pytestmark = pytest.mark.unit


def _runtime(*, teleop_client: object | None = None) -> SimpleNamespace:
    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(
            active_tab="debug",
            eval_start_lock=threading.Lock(),
            active_model_name=lambda: "model",
            session=SimpleNamespace(
                selected_collect_task=None,
                selected_collect_set=None,
                selected_collect_task_index=None,
                collection_sim_metadata=None,
                last_error="",
            ),
        ),
        command_queue=queue.Queue(),
        collection_teleop_armed=False,
        current_clip_id=None,
        current_cell=None,
        web_phase="ready",
        teleop_client=teleop_client,
    )
    runtime.console_ctx.runtime = runtime
    return runtime


def test_channel_rejects_unknown_commands_without_queueing() -> None:
    runtime = _runtime()
    reply = _handle_message(runtime, {"cmd": "web:not_a_command"})
    assert reply == {"ok": False, "error": "command not allowed: 'not_a_command'"}
    assert runtime.command_queue.empty()


def test_channel_status_includes_recorder_history(monkeypatch) -> None:
    runtime = _runtime()
    serialize = Mock(return_value={"eval_recorder": {"episodes": [{"clip_id": "trial-0001"}]}})
    monkeypatch.setattr(console_server, "_serialize_status", serialize)

    reply = _handle_message(runtime, {"query": "status"})

    assert reply["data"]["eval_recorder"]["episodes"] == [{"clip_id": "trial-0001"}]
    serialize.assert_called_once_with(runtime.console_ctx, include_history=True)


def test_channel_start_binds_eval_sim_metadata_before_queueing() -> None:
    runtime = _runtime()
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
    assert runtime.current_cell == {
        "prompt": "pick the block",
        "trial": 1,
        "sim_metadata": sim_metadata,
    }
    assert runtime.command_queue.get_nowait() == "web:start"


def test_channel_collect_start_binds_sim_metadata_before_queueing() -> None:
    runtime = _runtime()
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
    runtime = _runtime()
    runtime.console_ctx.active_tab = "collect"
    sim_metadata = {"schema_version": 4, "scene": {"scene_id": "scene-01"}}
    runtime.console_ctx.session.collection_sim_metadata = sim_metadata

    reply = _handle_message(runtime, {"cmd": "web:collect_start"})

    assert reply == {"ok": True, "cmd": "web:collect_start"}
    assert runtime.console_ctx.session.collection_sim_metadata is sim_metadata
    assert runtime.command_queue.get_nowait() == "web:collect_start"


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
