"""Shared terminal rendering for the CLI — theme, boot summary, status view, log handler.

One place decides how eva looks in a terminal, so the argparse entry (``main.py``),
the boot path (``core.app.run``) and the headless REPL (``core.app.cli``) can't drift
apart. ``rich`` does the drawing; it is already on the import path unconditionally
(``core.cfg.registry`` / ``core.cfg.config`` both use it), so this adds no new cost.

Everything degrades to plain text: the console is built with ``no_color`` /
``force_terminal`` derived from the real stream, so a piped run or ``NO_COLOR=1``
produces clean ASCII-ish output that greps and reads fine in a log file. Panels and
tables still render (as box-drawing characters), which is what you want in a captured
deploy log — only the escape codes go away.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import threading
from collections import deque
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# One palette for every surface. "brand" is the orange the prompt already used.
EVA_THEME = Theme(
    {
        "brand": "bold color(208)",
        "key": "dim",
        "val": "default",
        "ok": "bold green",
        "bad": "bold red",
        "warn": "bold yellow",
        "muted": "dim",
        "accent": "cyan",
    }
)

_console: Console | None = None
_err_console: Console | None = None

# Strips SGR sequences (color, bold, dim) from rendered text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _color_disabled() -> bool:
    return bool(os.environ.get("NO_COLOR"))


def console() -> Console:
    """Rich console on stdout, built once. Colors off for pipes / NO_COLOR."""
    global _console
    if _console is None:
        _console = Console(
            theme=EVA_THEME,
            no_color=_color_disabled(),
            soft_wrap=False,
            highlight=False,
        )
    return _console


def err_console() -> Console:
    """Rich console on stderr — for argparse-style fatal errors."""
    global _err_console
    if _err_console is None:
        _err_console = Console(
            theme=EVA_THEME,
            stderr=True,
            no_color=_color_disabled(),
            highlight=False,
        )
    return _err_console


def supports_color() -> bool:
    """True when stdout is a real terminal and NO_COLOR is unset."""
    return bool(sys.stdout and sys.stdout.isatty()) and not _color_disabled()


def flag(value: str, good: bool) -> Text:
    """A health value styled green/red, with a leading dot when it's a status word."""
    return Text(f"● {value}", style="ok" if good else "bad")


def kv_table(rows: list[tuple[str, Any]], *, key_width: int = 13) -> Table:
    """Two-column key/value grid — the house style for info blocks."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="key", width=key_width, no_wrap=True)
    table.add_column(style="val", overflow="fold")
    for key, value in rows:
        table.add_row(key, value if isinstance(value, Text) else str(value))
    return table


# --- boot summary -----------------------------------------------------------------


def boot_summary(
    *,
    robot: str,
    transport: str,
    obs_mode: str,
    policy: str,
    rates: str,
    entries: list[tuple[str, str]],
    mode_label: str,
) -> Table:
    """The launch block: what was built, and where to reach it.

    ``entries`` are the control entry points ("Console", "Control (ZMQ)", …) as
    (label, target) pairs — whatever the caller actually started.
    """
    rows: list[tuple[str, Any]] = [
        ("robot", robot),
        ("transport", f"{transport}  [muted]obs={obs_mode}[/muted]"),
        ("policy", policy),
        ("rates", rates),
    ]
    for label, target in entries:
        rows.append((label, Text(target, style="accent")))
    grid = kv_table(rows)

    outer = Table.grid(padding=(0, 0))
    outer.add_column()
    outer.add_row(Text.assemble(("  EVA", "brand"), (f"  {mode_label}", "muted")))
    outer.add_row("")
    outer.add_row(grid)
    return outer


def boot_summary_lines(**kwargs: Any) -> list[str]:
    """The boot block as plain text lines, for a log pane that owns the screen.

    The rich UI clears the terminal, so anything printed before it starts is lost. The
    same content goes through logging instead, which the pane buffers and renders.
    ``no_color`` still emits bold/dim codes, which prompt_toolkit would draw literally,
    so the output is stripped down to plain text.
    """
    con = Console(theme=EVA_THEME, no_color=True, width=100, file=io.StringIO())
    con.print(boot_summary(**kwargs))
    assert isinstance(con.file, io.StringIO)
    return [_ANSI_RE.sub("", line).rstrip() for line in con.file.getvalue().splitlines()]


def print_boot_summary(**kwargs: Any) -> None:
    """Render the boot block with breathing room around it.

    Goes to stderr, alongside the logs — it replaces log lines an operator used to
    capture with ``eva … 2> deploy.log``, and printing it to stdout instead would
    make the console URL and control endpoint vanish from those captures.
    """
    con = err_console()
    con.print()
    con.print(boot_summary(**kwargs))
    con.print()


# --- status ------------------------------------------------------------------------


def status_rows(data: dict) -> list[tuple[str, Any]]:
    """Flatten a ``_serialize_status`` payload into display rows.

    Takes the console serializer's dict rather than reaching into session/runtime, so
    the REPL, the ZMQ channel and the web console all describe state identically.
    """
    running = data.get("session_status") == "running"
    policy_ok = bool(data.get("policy_connected"))
    transport_ok = bool(data.get("transport_connected"))

    status_text = Text(
        str(data.get("session_status", "?")),
        style="ok" if running else "muted",
    )
    step = Text.assemble(
        (str(data.get("step_index", 0)), "bold"),
        ("  chunk ", "muted"),
        (str(data.get("chunk_index", 0)), "default"),
    )
    health = Text.assemble(
        flag("policy" if policy_ok else "policy", policy_ok),
        ("   ", ""),
        flag(f"{data.get('transport_type', 'transport')}", transport_ok),
    )
    rows: list[tuple[str, Any]] = [
        ("mode", str(data.get("cli_mode", "?")).upper()),
        ("status", status_text),
        ("phase", str(data.get("web_phase", "-"))),
        ("step", step),
        ("infer", f"{data.get('last_infer_ms', 0.0)} ms"),
        ("health", health),
        ("strategy", str(data.get("selected_strategy") or "-")),
        ("task", str(data.get("selected_task") or "-")),
    ]
    if data.get("setup_stage"):
        rows.append(("setup", str(data["setup_stage"])))
    elapsed = data.get("run_elapsed_ms") or 0
    if running and elapsed:
        rows.append(("elapsed", f"{elapsed / 1000.0:.1f} s"))
    if data.get("last_error"):
        rows.append(("error", Text(str(data["last_error"]), style="bad")))
    return rows


def status_view(data: dict) -> Table:
    """The ``status`` command's panel."""
    return kv_table(status_rows(data), key_width=10)


# --- logging ------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s"
LOG_DATEFMT = "%H:%M:%S"

_LEVEL_COLORS = {
    "DEBUG": "\x1b[2m",
    "INFO": "\x1b[38;5;39m",
    "WARN": "\x1b[33m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;31m",
}
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


class EvaLogFormatter(logging.Formatter):
    """The app log format, with the timestamp dimmed and the level colored.

    Only the two fixed-width prefix fields are styled — the message itself is left
    untouched so existing ``[CMD]`` / ``[HB]`` greps keep working. Color is applied
    only when ``use_color`` is set, so log files stay clean.
    """

    def __init__(self, use_color: bool) -> None:
        super().__init__(LOG_FORMAT, datefmt=LOG_DATEFMT)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        if not self.use_color:
            return line
        stamp, _, rest = line.partition(" ")
        level, _, message = rest.partition(" ")
        color = _LEVEL_COLORS.get(record.levelname, "")
        return f"{_DIM}{stamp}{_RESET} {color}{level}{_RESET} {message}"


class LogPane(logging.Handler):
    """Keeps the last N formatted log lines so a UI can render them as a pane.

    The rich CLI shows logs in a fixed region above the prompt rather than letting them
    scroll the screen. That needs the recent lines addressable, not just written once to
    a stream — so this handler buffers them and lets the renderer pull the tail.

    Records are formatted on arrival (cheap, bounded) because the formatter needs the
    live record, but the *rendering* happens on the UI's own repaint schedule.
    """

    def __init__(self, capacity: int = 400) -> None:
        super().__init__()
        self._lines: deque[tuple[int, str]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        # Bumped on every record so a poller can tell "anything new?" without copying.
        self.revision = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with self._lock:
            for part in line.splitlines() or [""]:
                self._lines.append((record.levelno, part))
            self.revision += 1

    def tail(self, count: int) -> list[tuple[int, str]]:
        """The most recent ``count`` lines as ``(levelno, text)``, oldest first."""
        with self._lock:
            if count <= 0:
                return []
            return list(self._lines)[-count:]


class CliLogFilter(logging.Filter):
    """Drops log records that would be noise while the interactive prompt is attached.

    Keeping the prompt readable under concurrent logging is the UI's job, so this only
    has to decide *what* to show, not *how*. Two things get dropped while ``cli_active``:

      - ``[HB]`` heartbeats — the status bar shows that data continuously, so an idle
        beat every 30s would just push the screen around for no new information.
      - anything a caller marks ``extra={"cli_quiet": True}``.

    Both come back the moment the CLI detaches, so a piped or post-``quit`` run keeps
    the full log.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cli_active = False
        self.suppress_heartbeat = True

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.cli_active:
            return True
        if getattr(record, "cli_quiet", False):
            return False
        if self.suppress_heartbeat and record.getMessage().startswith("[HB]"):
            return False
        return True


def configure_logging(use_color: bool | None = None) -> None:
    """Install the app's root log handler. Called once from ``main``."""
    logging.addLevelName(logging.WARNING, "WARN")
    if use_color is None:
        use_color = bool(sys.stderr and sys.stderr.isatty()) and not _color_disabled()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(EvaLogFormatter(use_color))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
