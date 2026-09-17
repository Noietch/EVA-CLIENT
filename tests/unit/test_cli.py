"""Headless CLI dispatch and shared terminal rendering."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from core.app import cli

pytestmark = pytest.mark.unit


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        inference_cfg=SimpleNamespace(debug_tasks=["pour soybean", "put cup"]),
    )


def _ctx() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    runtime = SimpleNamespace(command_queue=queue.Queue())
    session = SimpleNamespace()
    return _config(), runtime, session


def _dispatch(line: str, context: tuple) -> list[str]:
    config, runtime, session = context
    cli._dispatch(line, config, runtime, session)
    result = []
    while not runtime.command_queue.empty():
        result.append(runtime.command_queue.get_nowait())
    return result


def test_command_table_has_help_and_unique_names() -> None:
    names = [name for name, help_text in cli._COMMANDS if help_text]
    assert len(names) == len(cli._COMMANDS)
    assert len(names) == len(set(names))
    assert "setup" in cli._help_text()


def test_run_flow_commands_queue_expected_actions() -> None:
    ctx = _ctx()
    assert _dispatch("setup", ctx) == ["web:setup"]
    assert _dispatch("run", ctx) == ["web:run"]
    assert _dispatch("reset", ctx) == ["web:console_reset"]
    assert _dispatch("stop", ctx) == ["c"]
    assert _dispatch("c", ctx) == ["c"]


def test_task_selects_by_index_or_free_text(capsys) -> None:
    ctx = _ctx()
    assert _dispatch("task 1", ctx) == ["web:switch_task:put cup"]
    assert _dispatch("task pick up the pen", ctx) == ["web:switch_task:pick up the pen"]
    assert _dispatch("task 99", ctx) == []
    assert "out of range" in capsys.readouterr().out
    assert _dispatch("task", ctx) == []
