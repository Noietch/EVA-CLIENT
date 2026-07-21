"""ZMQ control channel — exposes every console button to external callers (e.g. a
simulator driving automated evaluation).

Every console button collapses, on the backend, to one thing: putting a ``web:*``
string onto ``runtime.command_queue`` for ``core.app.run.handle_command`` to
dispatch. This module opens a ZMQ REP socket that forwards any allow-listed
``web:*`` command onto that same queue, so a single transport exposes the full
button surface without mirroring each route. Read-only queries reuse the console
serializers (``_serialize_status`` / ``_serialize_config`` / ``_serialize_frame``).

Mirrors the ``operator_control`` bridge pattern (external event -> command_queue).
``zmq`` is imported lazily so hosts without pyzmq still import the package.
"""

from __future__ import annotations

import logging
import threading

from core.app.state import RuntimeState
from core.config import ConfigDict

logger = logging.getLogger(__name__)

# Allow-listed ``web:*`` verbs — the authoritative set matches the branches in
# core.app.run._handle_web_command. Keep this in sync when adding a verb there;
# an unlisted verb is rejected so the queue can't be an arbitrary-string sink.
_ALLOWED_VERBS = frozenset(
    {
        "bootstrap",
        "warmup",
        "switch_task",
        "start",
        "stop",
        "reset",
        "next_trial",
        "init_move",
        "init_gripper",
        "init_done",
        "switch_ckpt",
        "operator_action",
        "operator_button",
        "connect",
        "disconnect",
        "select_mode",
        "select_episode",
        "load_replay_dataset",
        "clear_replay",
        "set_replay_fps",
        "replay_seek",
        "select_strategy",
        "update_infer_params",
        "setup",
        "run",
        "halt",
        "rollout_intervention_abandon",
        "rollout_intervention_mode",
        "console_reset",
        "rollout_save",
        "rollout_stop",
        "tab_switch",
        "step_infer",
        "step_commit",
        "step_cancel",
        "manual_qpos",
        "manual_dispatch",
        "manual_send",
        "manual_home",
        "collect_start",
        "collect_stop",
        "collect_cancel",
        "gripper",
        # select_collect_task has no run.py verb — it is a pure ctx/session mutation
        # handled inline here (mirrors console/server.py::_post_select_collect_task).
        "select_collect_task",
    }
)

_READ_ONLY_QUERIES = frozenset({"status", "config", "frame"})


def _reject(reason: str) -> dict:
    return {"ok": False, "error": reason}


def _handle_query(runtime: RuntimeState, query: str) -> dict:
    """Serve a read-only snapshot by reusing the console serializers."""
    from core.app.console.server import (
        _serialize_config,
        _serialize_frame,
        _serialize_status,
    )

    ctx = runtime.console_ctx
    if ctx is None:
        return _reject("console context not ready")
    if query == "status":
        return {"ok": True, "data": _serialize_status(ctx)}
    if query == "config":
        return {"ok": True, "data": _serialize_config(ctx)}
    if query == "frame":
        return {"ok": True, "data": _serialize_frame(ctx)}
    return _reject(f"unknown query: {query!r}")


def _handle_command(runtime: RuntimeState, message: dict) -> dict:
    """Validate and dispatch one ``web:*`` command onto the command queue.

    Two verbs need the console-context mutations the HTTP layer applies before
    enqueueing (they don't reach through the queue otherwise):
      - select_collect_task: sets session.selected_collect_task (no enqueue).
      - tab_switch: when entering COLLECT, arms teleop via runtime fields.
    Everything else is a straight passthrough onto command_queue.
    """
    command = str(message.get("cmd", "")).strip()
    if not command.startswith("web:"):
        return _reject(f"command must start with 'web:', got {command!r}")
    payload = command[len("web:") :]
    verb, _, arg = payload.partition(":")
    verb = verb.strip()
    if verb not in _ALLOWED_VERBS:
        return _reject(f"command not allowed: {verb!r}")

    ctx = runtime.console_ctx
    if ctx is None:
        return _reject("console context not ready")

    if verb == "start" and message.get("clip_id"):
        runtime.current_clip_id = str(message["clip_id"])
        runtime.current_cell = {
            "prompt": message.get("prompt"),
            "trial": message.get("trial"),
        }

    # select_collect_task: pure session mutation, mirror console/server.py exactly.
    if verb == "select_collect_task":
        task = str(message.get("task", arg))
        ctx.session.selected_collect_task = task
        return {"ok": True, "selected_collect_task": task}

    # tab_switch into COLLECT: arm teleop the same way _post_tab_switch does, so a
    # later collect_start passes its ctx.active_tab + collection_teleop_armed gate.
    if verb == "tab_switch":
        tab = arg or "debug"
        armed = tab == "collect" and bool(message.get("armed", False))
        ctx.active_tab = tab
        runtime.collection_teleop_armed = armed
        ctx.session.last_error = ""
        command_tab = "collect" if armed else ("debug" if tab == "collect" else tab)
        assert runtime.command_queue is not None
        runtime.command_queue.put(f"web:tab_switch:{command_tab}")
        return {"ok": True, "tab": tab, "armed": armed}

    assert runtime.command_queue is not None
    runtime.command_queue.put(command)
    return {"ok": True, "cmd": command}


def _handle_message(runtime: RuntimeState, message: object) -> dict:
    if not isinstance(message, dict):
        return _reject("message must be a JSON object")
    if "query" in message:
        return _handle_query(runtime, str(message.get("query", "")).strip())
    if "cmd" in message:
        return _handle_command(runtime, message)
    return _reject("message must carry 'cmd' or 'query'")


def maybe_start_control_channel(config: ConfigDict, runtime: RuntimeState) -> None:
    """Start the ZMQ REP control channel if config.control_channel.enabled.

    Must be called AFTER start_console_server so runtime.console_ctx is set.
    """
    channel_cfg = config.get("control_channel") or {}
    if not channel_cfg.get("enabled", False):
        return
    if runtime.command_queue is None:
        raise RuntimeError("control_channel requires runtime.command_queue")

    import zmq

    host = str(channel_cfg.get("host", "127.0.0.1"))
    port = int(channel_cfg.get("port", 5757))

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")

    def serve() -> None:
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        logger.info("[CONTROL] ZMQ control channel: tcp://%s:%d", display_host, port)
        while True:
            try:
                message = socket.recv_json()
            except Exception as error:  # malformed frame / decode failure
                logger.warning("[CONTROL] recv failed: %s", error)
                try:
                    socket.send_json(_reject(f"recv failed: {error}"))
                except Exception:
                    pass
                continue
            try:
                reply = _handle_message(runtime, message)
            except Exception as error:
                logger.exception("[CONTROL] handler error")
                reply = _reject(f"handler error: {error}")
            try:
                socket.send_json(reply)
            except Exception:
                logger.exception("[CONTROL] send failed")

    thread = threading.Thread(target=serve, name="eva-control-channel", daemon=True)
    thread.start()
