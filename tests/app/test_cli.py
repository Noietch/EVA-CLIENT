"""Headless CLI dispatch and shared terminal rendering."""

from __future__ import annotations

import logging
import queue
from types import SimpleNamespace

import pytest

from core.app import cli
from core.app.tui import CliLogFilter, LogPane, boot_summary_lines, status_rows, status_view

pytestmark = pytest.mark.unit


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(type="arx_r5"),
        transport=SimpleNamespace(type="zmq"),
        policy=SimpleNamespace(type="openpi", host="127.0.0.1", port=9000),
        inference_cfg=SimpleNamespace(
            debug_tasks=["pour soybean", "put cup"],
            publish_rate=40,
            inference_rate=15,
        ),
        inference_strategies={"sync": {}, "async": {}},
        get=lambda key, default=None: {
            "control_channel": {"enabled": True, "host": "0.0.0.0", "port": 5757}
        }.get(key, default),
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


def test_unknown_command_is_reported_not_queued(capsys) -> None:
    assert _dispatch("nonsense", _ctx()) == []
    assert "unknown command" in capsys.readouterr().out


def _status_payload(**overrides: object) -> dict:
    data = {
        "cli_mode": "real",
        "session_status": "running",
        "web_phase": "running",
        "step_index": 142,
        "chunk_index": 9,
        "last_infer_ms": 23.4,
        "policy_connected": True,
        "transport_connected": True,
        "transport_type": "zmq",
        "selected_strategy": "async",
        "selected_task": "pour soybean",
        "setup_stage": "",
        "run_elapsed_ms": 61200,
        "last_error": "",
    }
    data.update(overrides)
    return data


def test_status_rows_surface_state_and_health() -> None:
    rows = dict(status_rows(_status_payload()))
    assert rows["mode"] == "REAL"
    assert rows["task"] == "pour soybean"
    assert rows["strategy"] == "async"
    assert str(rows["status"]) == "running"
    assert "policy" in str(rows["health"])


def test_status_rows_show_errors_and_hide_empty_optionals() -> None:
    rows = dict(status_rows(_status_payload(last_error="policy refused", setup_stage="")))
    assert str(rows["error"]) == "policy refused"
    assert "setup" not in rows

    rows = dict(status_rows(_status_payload(last_error="", setup_stage="resetting")))
    assert "error" not in rows
    assert rows["setup"] == "resetting"


def test_cli_help_and_info_are_safe_with_stub_context(capsys) -> None:
    ctx = _ctx()
    for command in ("help", "info", "tasks"):
        assert _dispatch(command, ctx) == []
    text = capsys.readouterr().out
    assert "setup" in text and "arx_r5" in text and "pour soybean" in text


def _record(message: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord("eva", level, "p", 1, message, (), None)


def test_log_pane_buffers_the_tail_not_the_transcript() -> None:
    pane = LogPane(capacity=3)
    pane.setFormatter(logging.Formatter("%(message)s"))
    for i in range(6):
        pane.emit(_record(f"line {i}"))
    assert [text for _, text in pane.tail(10)] == ["line 3", "line 4", "line 5"]
    # A pane taller than the buffer must not pad or crash.
    assert len(pane.tail(2)) == 2
    assert pane.tail(0) == []


def test_log_pane_splits_multiline_records() -> None:
    # A traceback arrives as one record; rendered as one fragment it would overflow the
    # pane's fixed height and push the prompt off screen.
    pane = LogPane()
    pane.setFormatter(logging.Formatter("%(message)s"))
    pane.emit(_record("first\nsecond\nthird"))
    assert [text for _, text in pane.tail(10)] == ["first", "second", "third"]


def test_log_pane_retains_record_levels() -> None:
    pane = LogPane()
    pane.setFormatter(logging.Formatter("%(message)s"))
    pane.emit(_record("boom", logging.ERROR))
    pane.emit(_record("hmm", logging.WARNING))
    assert pane.tail(5) == [(logging.ERROR, "boom"), (logging.WARNING, "hmm")]


def test_cli_filter_mutes_the_heartbeat_only_while_attached() -> None:
    flt = CliLogFilter()
    beat = _record("[HB] mode=real status=running")
    other = _record("[CMD] run")

    # Detached (piped run, or after `quit`): everything gets through.
    assert flt.filter(beat) is True
    assert flt.filter(other) is True

    flt.cli_active = True
    assert flt.filter(beat) is False
    assert flt.filter(other) is True

    quiet = _record("chatty")
    quiet.cli_quiet = True  # type: ignore[attr-defined]
    assert flt.filter(quiet) is False


def test_boot_summary_lines_are_plain_text() -> None:
    # The rich UI routes the boot block through logging into the pane; prompt_toolkit
    # draws escape codes literally, so the lines must carry none.
    lines = boot_summary_lines(
        robot="arx_r5",
        transport="zmq",
        obs_mode="JointState",
        policy="openpi @ 127.0.0.1:9000",
        rates="publish 40 Hz",
        entries=[("control (ZMQ)", "tcp://127.0.0.1:5757")],
        mode_label="headless",
    )
    body = "\n".join(lines)
    assert "\x1b" not in body
    assert "arx_r5" in body and "tcp://127.0.0.1:5757" in body


def test_cli_status_reads_live_state(capsys) -> None:
    config, runtime, _ = _ctx()
    runtime.transport = SimpleNamespace(seconds_since_last_recv=lambda: 0.1)
    runtime.policy = None
    runtime.web_phase = "idle"
    session = SimpleNamespace(
        status=SimpleNamespace(value="unset"), mode=SimpleNamespace(value="real"),
        step_index=0, last_infer_ms=0.0, selected_task=None, last_error="",
    )
    assert _dispatch("status", (config, runtime, session)) == []
    assert "transport=ok" in capsys.readouterr().out


def test_status_view_builds_a_table() -> None:
    assert status_view(_status_payload()).row_count == len(status_rows(_status_payload()))
