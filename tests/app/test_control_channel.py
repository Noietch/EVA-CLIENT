from __future__ import annotations

import inspect
import queue
from types import SimpleNamespace

from core.app import run
from core.app.command_catalog import WEB_COMMAND_VERBS, control_command_catalog
from core.app.control_channel import _handle_message


def test_catalog_matches_web_command_dispatch_branches() -> None:
    source = inspect.getsource(run._handle_web_command)
    branches = {
        line.split('"', 2)[1]
        for line in source.splitlines()
        if line.lstrip().startswith('if verb == "')
    }
    assert WEB_COMMAND_VERBS <= branches | {"select_collect_task"}


def test_catalog_is_json_ready_and_exposes_rl_templates() -> None:
    entries = control_command_catalog()
    assert all(set(entry) == {"verb", "command", "controls"} for entry in entries)
    assert any(
        entry["verb"] == "rl_select_critic"
        and entry["command"] == "web:rl_select_critic:{slot}"
        for entry in entries
    )
    assert any(
        entry["verb"] == "gripper"
        and entry["command"] == "web:gripper:{side}:{state}:{lock}"
        for entry in entries
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        console_ctx=SimpleNamespace(
            active_tab="debug",
            session=SimpleNamespace(selected_collect_task=None, last_error=""),
        ),
        command_queue=queue.Queue(),
        collection_teleop_armed=False,
    )


def test_channel_rejects_unknown_commands_without_queueing() -> None:
    runtime = _runtime()
    reply = _handle_message(runtime, {"cmd": "web:not_a_command"})
    assert reply == {"ok": False, "error": "command not allowed: 'not_a_command'"}
    assert runtime.command_queue.empty()


def test_channel_handles_rl_and_collect_commands() -> None:
    runtime = _runtime()

    assert _handle_message(runtime, {"cmd": "web:rl_setup"}) == {
        "ok": True,
        "cmd": "web:rl_setup",
    }
    assert runtime.command_queue.get_nowait() == "web:rl_setup"

    reply = _handle_message(
        runtime,
        {"cmd": "web:select_collect_task:pack the phone", "task": "pack the phone"},
    )
    assert reply == {"ok": True, "selected_collect_task": "pack the phone"}
    assert runtime.console_ctx.session.selected_collect_task == "pack the phone"
    assert runtime.command_queue.empty()

    reply = _handle_message(runtime, {"cmd": "web:tab_switch:collect", "armed": True})
    assert reply == {"ok": True, "tab": "collect", "armed": True}
    assert runtime.console_ctx.active_tab == "collect"
    assert runtime.collection_teleop_armed is True
    assert runtime.command_queue.get_nowait() == "web:tab_switch:collect"
