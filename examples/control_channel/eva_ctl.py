#!/usr/bin/env python3
"""eva_ctl — thin ZMQ client for the EVA control channel.

Drives every console button over the ZMQ REP socket exposed by
``core.app.control_channel`` (enable it with ``control_channel.enabled=True`` in
the config). Intended for a simulator to run automated evaluation without a browser.

Usage:
    python examples/control_channel/eva_ctl.py cmd web:select_mode:sim
    python examples/control_channel/eva_ctl.py cmd web:run
    python examples/control_channel/eva_ctl.py query status
    python examples/control_channel/eva_ctl.py wait-idle       # block until the run finishes
    python examples/control_channel/eva_ctl.py cmd web:tab_switch:collect --json '{"armed": true}'

The ``send()`` helper is import-friendly, so a simulator can drive the loop directly
(add this directory to sys.path, or copy this file next to your script):

    from eva_ctl import send
    send(cmd="web:select_mode:sim")
    send(cmd="web:setup")
    send(cmd="web:run")
    while send(query="status")["data"]["session_status"] == "running":
        time.sleep(0.5)
    send(cmd="web:halt")
    send(cmd="web:console_reset")
"""

from __future__ import annotations

import argparse
import json
import sys
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5757


def send(
    *,
    cmd: str | None = None,
    query: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    extra: dict | None = None,
    timeout_ms: int = 5000,
) -> dict:
    """Send one request to the control channel and return the JSON reply.

    Exactly one of ``cmd`` / ``query`` should be given. ``extra`` merges extra keys
    into the request (e.g. ``{"armed": True}`` for tab_switch, ``{"task": "..."}``
    for select_collect_task).
    """
    import zmq

    message: dict = {}
    if cmd is not None:
        message["cmd"] = cmd
    if query is not None:
        message["query"] = query
    if extra:
        message.update(extra)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://{host}:{port}")
    try:
        socket.send_json(message)
        reply = socket.recv_json()
    finally:
        socket.close()
    return reply  # type: ignore[return-value]


def wait_idle(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    poll_s: float = 0.5,
    timeout_s: float = 600.0,
) -> dict:
    """Poll status until session_status leaves 'running', then return the final status.

    Doubles as the run-completion signal for an evaluation loop: after web:run,
    wait_idle() blocks until the robot stops publishing (run ended / halted).
    """
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        reply = send(query="status", host=host, port=port)
        if not reply.get("ok"):
            return reply
        last = reply["data"]
        if last.get("session_status") != "running":
            return last
        time.sleep(poll_s)
    return last


_QUERY_WORDS = {"status", "config", "frame"}
_STATUS_LINE = (
    "mode={cli_mode} status={session_status} step={step_index} "
    "infer={last_infer_ms}ms policy={policy}"
)


def _fmt_status(data: dict) -> str:
    return _STATUS_LINE.format(
        cli_mode=data.get("cli_mode", "?"),
        session_status=data.get("session_status", "?"),
        step_index=data.get("step_index", 0),
        last_infer_ms=data.get("last_infer_ms", 0),
        policy="ok" if data.get("policy_connected") else "none",
    )


def _repl_dispatch(line: str, host: str, port: int) -> dict:
    """Map one REPL line to a send() call. Bare verbs get the ``web:`` prefix; the
    read-only words (status/config/frame) go out as queries."""
    parts = line.split()
    head = parts[0]
    if head in _QUERY_WORDS:
        return send(query=head, host=host, port=port)
    # e.g. "select_mode sim" -> web:select_mode:sim ; "gripper l open 0" -> web:gripper:l:open:0
    verb = ":".join(parts)
    cmd = verb if verb.startswith("web:") else f"web:{verb}"
    return send(cmd=cmd, host=host, port=port)


def _watch(host: str, port: int) -> None:
    """Single-line live status refresh until Ctrl-C (REPL-side status display)."""
    try:
        while True:
            reply = send(query="status", host=host, port=port)
            if reply.get("ok"):
                sys.stdout.write("\r" + _fmt_status(reply["data"]) + "   ")
            else:
                sys.stdout.write("\r" + str(reply.get("error", "error")) + "   ")
            sys.stdout.flush()
            time.sleep(1.0)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


_REPL_HELP = """commands:
  <verb> [args]      send web:<verb>[:arg...]  (e.g. run, halt, setup, select_mode sim)
  status|config|frame   read-only query
  watch              live single-line status until Ctrl-C
  help               this help
  quit|exit          leave the REPL
"""


def repl(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Interactive prompt over the control channel. Pure client; no backend deps."""
    try:
        import readline  # noqa: F401  (enables line editing/history if available)
    except Exception:
        pass
    print(f"eva control REPL — tcp://{host}:{port}. 'help' for commands, 'quit' to exit.")
    while True:
        try:
            line = input("eva> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"quit", "exit"}:
            return 0
        if line == "help":
            print(_REPL_HELP)
            continue
        if line == "watch":
            _watch(host, port)
            continue
        try:
            reply = _repl_dispatch(line, host, port)
        except Exception as error:  # keep the REPL alive on any client-side error
            print(f"error: {error}")
            continue
        if reply.get("ok") and "data" in reply and isinstance(reply["data"], dict):
            data = reply["data"]
            if "session_status" in data:
                print(_fmt_status(data))
            else:
                print(json.dumps(data, ensure_ascii=False))
        else:
            print(json.dumps(reply, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="EVA control-channel CLI client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="action", required=True)

    p_cmd = sub.add_parser("cmd", help="send a web:* command")
    p_cmd.add_argument("command", help="e.g. web:run")
    p_cmd.add_argument("--json", dest="extra", default="", help="extra keys as JSON object")

    p_query = sub.add_parser("query", help="read-only query")
    p_query.add_argument("name", choices=["status", "config", "frame"])

    sub.add_parser("wait-idle", help="block until the active run finishes")
    sub.add_parser("repl", help="interactive prompt over the control channel")

    args = parser.parse_args()

    if args.action == "cmd":
        extra = json.loads(args.extra) if args.extra else None
        reply = send(cmd=args.command, host=args.host, port=args.port, extra=extra)
    elif args.action == "query":
        reply = send(query=args.name, host=args.host, port=args.port)
    elif args.action == "wait-idle":
        reply = {"ok": True, "data": wait_idle(host=args.host, port=args.port)}
    elif args.action == "repl":
        return repl(host=args.host, port=args.port)
    else:  # pragma: no cover - argparse guards this
        parser.error(f"unknown action {args.action!r}")
        return 2

    print(json.dumps(reply, ensure_ascii=False, indent=2))
    return 0 if reply.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main())
