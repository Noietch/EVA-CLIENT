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
import math
import threading
from typing import cast

from core.app.command_catalog import WEB_COMMAND_VERBS
from core.app.handlers.teleop import send_teleop_haptic
from core.app.state import RuntimeState
from core.config import ConfigDict

logger = logging.getLogger(__name__)

# Allow-listed ``web:*`` verbs — shared with the web metadata registry. Keep the
# registry in sync with core.app.run._handle_web_command; an unlisted verb is
# rejected so the queue can't be an arbitrary-string sink.
_ALLOWED_VERBS = WEB_COMMAND_VERBS

_READ_ONLY_QUERIES = frozenset({"status", "config", "frame"})
_HAPTIC_HANDS = frozenset({"left", "right", "both", "all"})
_HAPTIC_DEFAULT_INTENSITY = 0.6
_HAPTIC_DEFAULT_DURATION_MS = 80.0


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
        return {"ok": True, "data": _serialize_status(ctx, include_history=True)}
    if query == "config":
        return {"ok": True, "data": _serialize_config(ctx)}
    if query == "frame":
        return {"ok": True, "data": _serialize_frame(ctx)}
    return _reject(f"unknown query: {query!r}")


def _finite_haptic_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _haptic_request(message: dict, arg: str) -> tuple[tuple[str, ...], float, float]:
    """Parse JSON fields or ``web:haptic:hand:intensity:duration_ms`` arguments."""
    positional = [part.strip() for part in arg.split(":")] if arg else []
    if len(positional) > 3 or any(not part for part in positional):
        raise ValueError("haptic format is web:haptic[:hand[:intensity[:duration_ms]]]")

    hand = str(message.get("hand", positional[0] if positional else "both")).strip().lower()
    if hand == "all":
        hand = "both"
    if hand not in _HAPTIC_HANDS:
        raise ValueError("haptic hand must be 'left', 'right', or 'both'")

    intensity_value = message.get(
        "intensity",
        positional[1] if len(positional) >= 2 else _HAPTIC_DEFAULT_INTENSITY,
    )
    duration_value = message.get(
        "duration_ms",
        positional[2] if len(positional) >= 3 else _HAPTIC_DEFAULT_DURATION_MS,
    )
    intensity = max(0.0, min(1.0, _finite_haptic_number(intensity_value, "intensity")))
    duration_ms = max(
        1.0,
        min(1000.0, _finite_haptic_number(duration_value, "duration_ms")),
    )
    hands = ("left", "right") if hand == "both" else (hand,)
    return hands, intensity, duration_ms


def _handle_haptic(runtime: RuntimeState, message: dict, arg: str) -> dict:
    try:
        hands, intensity, duration_ms = _haptic_request(message, arg)
    except ValueError as error:
        return _reject(str(error))

    sent = [
        hand
        for hand in hands
        if send_teleop_haptic(
            runtime,
            hand=hand,
            intensity=intensity,
            duration_ms=duration_ms,
        )
    ]
    if not sent:
        return _reject("VR headset is not connected")
    return {
        "ok": True,
        "cmd": "web:haptic",
        "hands": sent,
        "intensity": intensity,
        "duration_ms": duration_ms,
    }


def _handle_command(runtime: RuntimeState, message: dict) -> dict:
    """Validate and dispatch one ``web:*`` command or immediate haptic request.

    Two verbs need the console-context mutations the HTTP layer applies before
    enqueueing (they don't reach through the queue otherwise):
      - select_collect_task: sets session.selected_collect_task (no enqueue).
      - tab_switch: updates the active Console tab.
    Haptic requests are sent directly to the VR client; everything else is a
    straight passthrough onto command_queue.
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

    # Haptics are latency-sensitive and do not need the main command queue. Send
    # directly to the VR client's bounded outbound queue from this ZMQ worker.
    if verb == "haptic":
        return _handle_haptic(runtime, message, arg)

    # select_collect_task: pure session mutation, mirror console/server.py exactly.
    if verb == "select_collect_task":
        task = str(message.get("task", arg))
        dataset = str(message.get("dataset", "")).strip() or None
        task_index_value = message.get("task_index")
        task_index = int(task_index_value) if task_index_value is not None else None
        ctx.session.selected_collect_task = task
        ctx.session.selected_collect_set = dataset
        ctx.session.selected_collect_task_index = task_index
        response = {"ok": True, "selected_collect_task": task}
        if dataset is not None:
            response.update(dataset=dataset, task_index=task_index)
        return response

    if verb == "tab_switch":
        tab = arg or "debug"
        ctx.active_tab = tab
        ctx.session.last_error = ""
        assert runtime.command_queue is not None
        runtime.command_queue.put(f"web:tab_switch:{tab}")
        return {"ok": True, "tab": tab}

    # Bind trial metadata before starting evaluation
    if verb == "start" and message.get("clip_id"):
        clip_id = str(message["clip_id"])
        runtime.current_clip_id = clip_id
        runtime.current_cell = {
            "prompt": message.get("prompt"),
            "trial": message.get("trial"),
        }

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
