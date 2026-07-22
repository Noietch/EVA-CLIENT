"""In-process interactive CLI for headless inference.

A tiny stdin REPL that runs inside the eva process (no browser, no network hop).
It drives the same state machine as everything else by putting ``web:*`` strings
onto ``runtime.command_queue`` — identical to what the ZMQ control channel and the
web console do — and reads ``session`` / ``runtime`` directly for status display.

Scope is deliberately small: the real-robot inference flow, as step commands
(task / setup / run / stop / reset / status). The mode is fixed to REAL — SIM only
previews to the 3D canvas, which headless disables, so it has no meaning here. It
coexists with the ZMQ channel: both feed the one queue, so a simulator and a human
can drive the same session. Only started when stdin is a TTY, so a backgrounded /
piped process keeps running headless with just the ZMQ entry.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

from core.app.state import RuntimeState, SessionState
from core.config import ConfigDict

logger = logging.getLogger(__name__)

# Color is enabled only for a real terminal without NO_COLOR set (so pipes / log files
# stay clean). Flipped on in maybe_start_inference_cli.
_USE_COLOR = False

_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "orange": "38;5;208",
}


def _c(text: str, *styles: str) -> str:
    """Wrap ``text`` in ANSI styles when color is on, else return it unchanged."""
    if not _USE_COLOR or not styles:
        return text
    codes = ";".join(_CODES[s] for s in styles)
    return f"\x1b[{codes}m{text}\x1b[0m"


# The visible prompt is always "eva> " (5 cols). Two forms: the input() form wraps the
# color codes in \001/\002 so readline computes width correctly; the echo form (used by
# the log handler's redraw) uses raw codes since it writes straight to the stream.
def _input_prompt() -> str:
    if not _USE_COLOR:
        return "eva> "
    return "\001\x1b[1;38;5;208m\002eva> \001\x1b[0m\002"


def _echo_prompt() -> str:
    return _c("eva> ", "bold", "orange")


_COMMANDS = [
    ("info", "endpoints / ports / robot (from the config)"),
    ("tasks", "list configured debug tasks (with indices)"),
    ("task <text|N>", "select task by text or by index from `tasks`"),
    ("setup", "connect + reset + validate"),
    ("run", "start continuous inference"),
    ("stop / c", "stop publishing"),
    ("reset", "reset the arm to home"),
    ("status", "current state + policy/transport health"),
    ("help", "this help"),
    ("quit / exit", "leave the CLI (inference keeps running)"),
]


def _help_text() -> str:
    header = _c("eva inference CLI", "bold") + _c("  (real-robot)", "dim")
    rows = "\n".join(
        f"  {_c(name.ljust(14), 'bold', 'orange')}  {_c(desc, 'dim')}" for name, desc in _COMMANDS
    )
    return f"{header}\n{rows}"


def _debug_tasks(config: ConfigDict) -> list[str]:
    return list(config.inference_cfg.debug_tasks or [])


def _transport_online(runtime: RuntimeState) -> bool:
    # Mirrors _serialize_status: a live backend is "online" only once a message has
    # arrived recently; None (never received) counts as offline.
    since = runtime.transport.seconds_since_last_recv()
    return since is not None and since <= 2.0


def _info_lines(config: ConfigDict, runtime: RuntimeState) -> str:
    ch = config.get("control_channel") or {}
    rows = [
        ("robot", config.robot.type),
        ("transport", config.transport.type),
        ("policy", f"{config.policy.type} @ {config.policy.host}:{config.policy.port}"),
        (
            "control (ZMQ)",
            f"{'on' if ch.get('enabled') else 'off'}  "
            f"{ch.get('host', '127.0.0.1')}:{ch.get('port', 5757)}",
        ),
        ("tasks", f"{len(_debug_tasks(config))} configured (type 'tasks')"),
    ]
    return "\n".join(f"  {_c(key.ljust(13), 'dim')}  {val}" for key, val in rows)


def _tasks_lines(config: ConfigDict) -> str:
    tasks = _debug_tasks(config)
    if not tasks:
        return _c("no debug tasks configured", "dim") + " — use 'task <text>' with a free prompt"
    return "\n".join(f"  {_c(f'[{i}]', 'orange')} {t}" for i, t in enumerate(tasks))


def _flag(value: str, good: bool) -> str:
    return _c(value, "green" if good else "red")


def _status_line(runtime: RuntimeState, session: SessionState) -> str:
    online = _transport_online(runtime)
    policy_ok = runtime.policy is not None
    running = session.status.value == "running"
    return (
        f"mode={session.mode.value} "
        f"status={_c(session.status.value, 'green') if running else session.status.value} "
        f"phase={runtime.web_phase} "
        f"step={_c(str(session.step_index), 'bold')} "
        f"infer={session.last_infer_ms:.1f}ms "
        f"policy={_flag('ok' if policy_ok else 'none', policy_ok)} "
        f"transport={_flag('ok' if online else 'offline', online)} "
        f"task={session.selected_task or '-'} "
        f"error={_c(session.last_error, 'red') if session.last_error else '-'}"
    )


def _resolve_task(arg: str, config: ConfigDict) -> str | None:
    # A bare integer selects by index into the configured debug tasks; anything else is
    # taken as a free-form prompt (headless testing shouldn't be limited to the presets).
    if arg.isdigit():
        tasks = _debug_tasks(config)
        idx = int(arg)
        if 0 <= idx < len(tasks):
            return tasks[idx]
        print(f"task index {idx} out of range (0..{len(tasks) - 1}); see 'tasks'")
        return None
    return arg


def _dispatch(
    line: str, config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> None:
    """Map one CLI line to a queued command or an inline print."""
    parts = line.split(maxsplit=1)
    verb = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    queue = runtime.command_queue
    assert queue is not None

    if verb in {"help", "?"}:
        print(_help_text())
    elif verb == "info":
        print(_info_lines(config, runtime))
    elif verb == "tasks":
        print(_tasks_lines(config))
    elif verb == "status":
        print(_status_line(runtime, session))
    elif verb == "task":
        if not arg:
            print(_c("usage:", "dim") + " task <text|N>   (N = index from 'tasks')")
        else:
            resolved = _resolve_task(arg, config)
            if resolved is not None:
                queue.put(f"web:switch_task:{resolved}")
    elif verb == "setup":
        queue.put("web:setup")
    elif verb == "run":
        queue.put("web:run")
    elif verb in {"stop", "c"}:
        queue.put("c")
    elif verb == "reset":
        queue.put("web:console_reset")
    else:
        print(_c(f"unknown command: {verb!r}", "red") + " (try 'help')")


def _repl(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    try:
        import readline  # noqa: F401  (line editing/history when available)
    except Exception:
        pass
    # REAL mode is already selected synchronously by the caller (before this banner),
    # so its logs don't land on top of the prompt.
    print()
    print(_c("  EVA", "bold", "orange") + _c("  inference CLI · real-robot", "dim"))
    hint = (
        _c("  type ", "dim")
        + _c("help", "bold")
        + _c(" for commands, ", "dim")
        + _c("info", "bold")
        + _c(" for endpoints", "dim")
    )
    print(hint)
    print()
    while True:
        try:
            line = input(_input_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in {"quit", "exit"}:
            return
        try:
            _dispatch(line, config, runtime, session)
        except Exception as error:  # never let a typo kill the CLI thread
            print(f"error: {error}")


class _PromptPreservingHandler(logging.StreamHandler):
    """A stream handler that keeps the ``eva> `` line intact under concurrent logging.

    Logs (heartbeat, [CMD], [STATUS]) are emitted from other threads while the REPL
    sits at an ``input()`` prompt. Without coordination each log line lands right after
    whatever the user has half-typed. This handler, when the CLI is active, first erases
    the current terminal line, writes the record, then redraws the prompt plus the
    in-progress input (from readline's line buffer) so typing is never clobbered.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)
        self.cli_active = False

    def emit(self, record: logging.LogRecord) -> None:
        if not self.cli_active:
            super().emit(record)
            return
        try:
            msg = self.format(record)
            # \r + clear-to-end-of-line, print the log, then repaint prompt + buffer.
            pending = ""
            try:
                import readline

                pending = readline.get_line_buffer()
            except Exception:
                pending = ""
            self.stream.write(f"\r\x1b[K{msg}\n{_echo_prompt()}{pending}")
            self.flush()
        except Exception:
            self.handleError(record)


_prompt_handler: _PromptPreservingHandler | None = None


def _install_prompt_handler() -> None:
    """Route root logging through the prompt-preserving handler (idempotent)."""
    global _prompt_handler
    root = logging.getLogger()
    if _prompt_handler is not None:
        _prompt_handler.cli_active = True
        return
    handler = _PromptPreservingHandler()
    # Match the app's existing format (main.py basicConfig) so lines look identical.
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    handler.cli_active = True
    # Replace the stream handlers basicConfig installed so logs don't double-print.
    for existing in list(root.handlers):
        if isinstance(existing, logging.StreamHandler):
            root.removeHandler(existing)
    root.addHandler(handler)
    _prompt_handler = handler


def maybe_start_inference_cli(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> None:
    """Start the interactive CLI on a daemon thread when stdin is a real terminal.

    Backgrounded / piped runs (stdin not a TTY) skip it and stay driven by ZMQ only.
    """
    if runtime.command_queue is None:
        raise RuntimeError("inference CLI requires runtime.command_queue")
    if not sys.stdin or not sys.stdin.isatty():
        logger.info("stdin is not a TTY; interactive CLI disabled (ZMQ control only)")
        return
    # Enable ANSI color only for an interactive terminal without NO_COLOR set.
    global _USE_COLOR
    _USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    _install_prompt_handler()
    thread = threading.Thread(
        target=_repl,
        args=(config, runtime, session),
        name="eva-inference-cli",
        daemon=True,
    )
    thread.start()
    logger.info("Interactive inference CLI started (type 'help')")
