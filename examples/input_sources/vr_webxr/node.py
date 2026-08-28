#!/usr/bin/env python3
"""External WebXR input node publishing normalized VR frames to EVA Client."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import queue
import secrets
import ssl
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.parse import parse_qs, urlencode, urlsplit

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve

PROTOCOL = "eva.teleop.vr"
PROTOCOL_VERSION = 3
HANDS = ("left", "right")
DEFAULT_EVENT_RETRY_TTL_S = 2.0
DEFAULT_EVENT_QUEUE_MAXSIZE = 64
DEFAULT_ACK_QUEUE_MAXSIZE = 64
MAX_PENDING_EVENTS = 64
_BRIDGE_START_TIMEOUT_S = 5.0
_BRIDGE_STOP_TIMEOUT_S = 5.0
STATIC_ROOT = Path(__file__).with_name("static")
THREE_JS = (
    Path(__file__).parents[3] / "src/core/app/console/static/vendor/three/three.module.min.js"
)

logger = logging.getLogger("eva.vr_webxr_node")


class BrowserProtocolError(ValueError):
    pass


class FrameConsoleMonitor:
    """Rate-limited, best-effort terminal diagnostics for received WebXR frames."""

    def __init__(self, *, interval_s: float = 0.25, stream: TextIO = sys.stdout) -> None:
        if not 0.0 <= float(interval_s) < float("inf"):
            raise ValueError("stats interval must be non-negative and finite")
        self._interval_s = float(interval_s)
        self._stream = stream
        self._enabled = self._interval_s > 0.0
        self._last_seq: int | None = None
        self._last_arrival_ms: float | None = None
        self._last_report_ms = -float("inf")
        self._minimum_clock_offset_ms = float("inf")
        self._smoothed_period_ms: float | None = None
        self._dropped_frames = 0
        self._rtt_ms: float | None = None
        self._line_width = 0
        self._has_output = False

    def set_rtt(self, rtt_s: float) -> None:
        self._rtt_ms = max(0.0, float(rtt_s) * 1000.0)

    @staticmethod
    def _controller_summary(label: str, value: object) -> str:
        if not isinstance(value, Mapping) or not value.get("valid", False):
            return f"{label}=invalid"
        position = value.get("position")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)):
            return f"{label}=invalid"
        try:
            x, y, z = (float(position[index]) for index in range(3))
            trigger = float(value.get("trigger", 0.0))
            grip_engaged = "on" if value.get("grip_engaged", False) else "off"
        except (IndexError, TypeError, ValueError):
            return f"{label}=invalid"
        return f"{label}=({x:+.2f},{y:+.2f},{z:+.2f}) t={trigger:.2f} {grip_engaged}"

    def observe(self, frame: Mapping[str, object], *, arrival_ms: float | None = None) -> None:
        """Record one frame without allowing diagnostics to disrupt frame handling."""
        if not self._enabled:
            return
        try:
            self._observe(frame, arrival_ms=arrival_ms)
        except Exception:
            self._enabled = False
            logger.debug("Disabled VR frame diagnostics after output failure", exc_info=True)

    def _observe(self, frame: Mapping[str, object], *, arrival_ms: float | None) -> None:
        now_ms = time.monotonic() * 1000.0 if arrival_ms is None else float(arrival_ms)
        source_ms = float(cast(float, frame["client_time_ms"]))
        seq = int(cast(int, frame["seq"]))

        period_ms = None if self._last_arrival_ms is None else now_ms - self._last_arrival_ms
        if period_ms is not None and period_ms >= 0.0:
            if self._smoothed_period_ms is None:
                self._smoothed_period_ms = period_ms
            else:
                self._smoothed_period_ms = 0.9 * self._smoothed_period_ms + 0.1 * period_ms
        if self._last_seq is not None and seq > self._last_seq + 1:
            self._dropped_frames += seq - self._last_seq - 1
        self._last_seq = seq
        self._last_arrival_ms = now_ms

        clock_offset_ms = now_ms - source_ms
        self._minimum_clock_offset_ms = min(self._minimum_clock_offset_ms, clock_offset_ms)
        relative_delay_ms = max(0.0, clock_offset_ms - self._minimum_clock_offset_ms)

        if now_ms - self._last_report_ms < self._interval_s * 1000.0:
            return
        self._last_report_ms = now_ms
        rate_hz = (
            1000.0 / self._smoothed_period_ms
            if self._smoothed_period_ms is not None and self._smoothed_period_ms > 0.0
            else 0.0
        )
        interval_text = "n/a" if period_ms is None else f"{period_ms:.1f}ms"
        controllers = cast(Mapping[str, object], frame.get("controllers", {}))
        line = " | ".join(
            (
                f"VR RX seq={seq}",
                f"rate={rate_hz:.1f}Hz",
                f"interval={interval_text}",
                f"rtt={'n/a' if self._rtt_ms is None else f'{self._rtt_ms:.1f}ms'}",
                f"relative_delay={relative_delay_ms:.1f}ms",
                f"dropped={self._dropped_frames}",
                self._controller_summary("L", controllers.get("left")),
                self._controller_summary("R", controllers.get("right")),
            )
        )
        self._line_width = max(self._line_width, len(line))
        self._stream.write("\r" + line.ljust(self._line_width))
        self._stream.flush()
        self._has_output = True

    def finish(self) -> None:
        if not self._has_output:
            return
        try:
            self._stream.write("\n")
            self._stream.flush()
        except Exception:
            pass
        self._has_output = False


@dataclasses.dataclass(frozen=True)
class ControllerLayout:
    trigger: int = 0
    grip: int = 1
    primary: int = 4
    secondary: int = 5


_DEFAULT_LAYOUT = ControllerLayout()
_PROFILE_LAYOUTS = {
    "pico-4": ControllerLayout(),
    "pico-4-ultra": ControllerLayout(),
    "oculus-touch": ControllerLayout(),
    "oculus-touch-v2": ControllerLayout(),
    "oculus-touch-v3": ControllerLayout(),
    "meta-quest-touch-plus": ControllerLayout(),
}


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BrowserProtocolError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BrowserProtocolError(f"{field} must be a list")
    return cast(Sequence[Any], value)


def _finite(value: object, field: str) -> float:
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise BrowserProtocolError(f"{field} must be numeric") from error
    if not -float("inf") < result < float("inf"):
        raise BrowserProtocolError(f"{field} must be finite")
    return result


def _button_value(buttons: Sequence[Any], index: int) -> float:
    if index >= len(buttons):
        return 0.0
    button = _mapping(buttons[index], f"buttons[{index}]")
    value = _finite(button.get("value", 0.0), f"buttons[{index}].value")
    # Some PICO WebXR runtimes expose the trigger as a digital button: the
    # pressed flag changes, but GamepadButton.value remains zero. Preserve the
    # press for both binary and toggle gripper mappings.
    if bool(button.get("pressed", False)):
        value = max(value, 1.0)
    return max(0.0, min(1.0, value))


def _button_pressed(buttons: Sequence[Any], index: int) -> bool:
    if index >= len(buttons):
        return False
    button = _mapping(buttons[index], f"buttons[{index}]")
    return bool(button.get("pressed", False)) or _button_value(buttons, index) >= 0.5


def _layout(profiles: Sequence[Any]) -> ControllerLayout:
    normalized = tuple(str(profile).lower() for profile in profiles)
    for profile in normalized:
        for key, layout in _PROFILE_LAYOUTS.items():
            if key in profile:
                return layout
    return _DEFAULT_LAYOUT


def _pose(raw: Mapping[str, Any], field: str) -> dict[str, list[float]]:
    position = [
        _finite(value, f"{field}.position") for value in _sequence(raw.get("position"), field)
    ]
    orientation = [
        _finite(value, f"{field}.orientation_xyzw")
        for value in _sequence(raw.get("orientation_xyzw"), field)
    ]
    if len(position) != 3 or len(orientation) != 4:
        raise BrowserProtocolError(f"{field} pose dimensions are invalid")
    orientation_norm = float(np.linalg.norm(np.asarray(orientation, dtype=np.float64)))
    if orientation_norm <= 1e-8:
        raise BrowserProtocolError(f"{field}.orientation_xyzw must be non-zero")
    orientation = [value / orientation_norm for value in orientation]
    return {"position": position, "orientation_xyzw": orientation}


def normalize_controller(hand: str, value: object) -> tuple[dict[str, object], dict[str, bool]]:
    raw = _mapping(value, f"controllers.{hand}")
    valid = bool(raw.get("valid", False))
    profiles = _sequence(raw.get("profiles", ()), f"controllers.{hand}.profiles")
    buttons = _sequence(raw.get("buttons", ()), f"controllers.{hand}.buttons")
    layout = _layout(profiles)
    normalized: dict[str, object] = {
        "valid": valid,
        "trigger": _button_value(buttons, layout.trigger),
        "profiles": [str(profile) for profile in profiles],
    }
    if valid:
        normalized.update(_pose(raw, f"controllers.{hand}"))
    face = {
        "primary": _button_pressed(buttons, layout.primary),
        "secondary": _button_pressed(buttons, layout.secondary),
        "grip": _button_pressed(buttons, layout.grip),
    }
    return normalized, face


class OperatorEventMapper:
    """Translate controller face-button gestures into idempotent operator intents.

    A short press emits ``record_toggle`` on release. Holding A for the configured
    duration emits exactly one ``record_cancel`` and suppresses the later release.
    A short B press emits ``arm_toggle`` on release. A short left X press emits
    ``home`` on release, and a short left Y press emits ``intervention_toggle``.
    Grip is consumed here only for its
    long-press toggle state; the client receives the resulting state alongside
    the absolute controller pose.
    """

    def __init__(self, *, long_press_ms: float = 1000.0) -> None:
        if not 0.0 < float(long_press_ms) < float("inf"):
            raise ValueError("long_press_ms must be positive and finite")
        self._long_press_ms = float(long_press_ms)
        self._event_id = 0
        self._previous_primary = False
        self._previous_secondary = False
        self._previous_home = False
        self._previous_intervention = False
        self._press_started_ms: float | None = None
        self._secondary_started_ms: float | None = None
        self._long_press_fired = False

    def reset(self) -> None:
        self._event_id = 0
        self._previous_primary = False
        self._previous_secondary = False
        self._previous_home = False
        self._previous_intervention = False
        self._press_started_ms = None
        self._secondary_started_ms = None
        self._long_press_fired = False

    def update(
        self,
        *,
        session_id: str,
        client_time_ms: float,
        controllers: Mapping[str, Mapping[str, object]],
        face_buttons: Mapping[str, Mapping[str, bool]],
    ) -> tuple[dict[str, object], ...]:
        _ = controllers
        pressed = bool(face_buttons["right"].get("primary", False))
        secondary = bool(face_buttons["right"].get("secondary", False))
        home_pressed = bool(face_buttons["left"].get("primary", False))
        intervention_pressed = bool(face_buttons["left"].get("secondary", False))
        intents: list[str] = []
        if pressed and not self._previous_primary:
            self._press_started_ms = float(client_time_ms)
            self._long_press_fired = False
        if self._press_started_ms is not None and pressed and not self._long_press_fired:
            if client_time_ms - self._press_started_ms >= self._long_press_ms:
                intents.append("record_cancel")
                self._long_press_fired = True
        if self._previous_primary and not pressed:
            if self._press_started_ms is not None and not self._long_press_fired:
                intents.append("record_toggle")
            self._press_started_ms = None
            self._long_press_fired = False
        if secondary and not self._previous_secondary:
            self._secondary_started_ms = float(client_time_ms)
        if self._previous_secondary and not secondary:
            if (
                self._secondary_started_ms is not None
                and client_time_ms - self._secondary_started_ms < self._long_press_ms
            ):
                intents.append("arm_toggle")
            self._secondary_started_ms = None
        if self._previous_home and not home_pressed:
            intents.append("home")
        if self._previous_intervention and not intervention_pressed:
            intents.append("intervention_toggle")
        self._previous_primary = pressed
        self._previous_secondary = secondary
        self._previous_home = home_pressed
        self._previous_intervention = intervention_pressed
        events = []
        for intent in intents:
            events.append(
                {
                    "protocol": PROTOCOL,
                    "version": PROTOCOL_VERSION,
                    "type": "event",
                    "session_id": session_id,
                    "event_id": self._event_id,
                    "client_time_ms": client_time_ms,
                    "intent": intent,
                }
            )
            self._event_id += 1
        return tuple(events)

    def input_feedback(self, client_time_ms: float) -> dict[str, object]:
        pressed = []
        if self._previous_primary:
            pressed.append("right.primary")
        if self._previous_secondary:
            pressed.append("right.secondary")
        if self._previous_home:
            pressed.append("left.primary")
        if self._previous_intervention:
            pressed.append("left.secondary")
        progress: dict[str, float] = {}
        if self._previous_primary and self._press_started_ms is not None:
            progress["right.primary"] = min(
                1.0,
                max(0.0, (float(client_time_ms) - self._press_started_ms) / self._long_press_ms),
            )
        return {"pressed": pressed, "hold_progress": progress}


@dataclasses.dataclass
class _PendingEvent:
    payload: dict[str, object]
    created_at: float
    last_sent_at: float = 0.0

    def should_retry(
        self,
        *,
        browser_connected: bool,
        session_id: str,
        now: float,
        retry_ttl_s: float,
    ) -> bool:
        return bool(
            browser_connected
            and session_id
            and str(self.payload.get("session_id", "")) == session_id
            and now - self.created_at <= retry_ttl_s
        )


def normalize_browser_frame(
    payload: str | bytes | Mapping[str, Any],
    *,
    session_id: str,
) -> tuple[dict[str, object], dict[str, dict[str, bool]]]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            decoded: object = json.loads(payload)
        except json.JSONDecodeError as error:
            raise BrowserProtocolError("browser payload must be valid JSON") from error
    else:
        decoded = payload
    raw = _mapping(decoded, "payload")
    if raw.get("type") != "frame" or raw.get("version") != 1:
        raise BrowserProtocolError("browser frame version/type is unsupported")
    seq = raw.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise BrowserProtocolError("seq must be a non-negative integer")
    reference_space = str(raw.get("reference_space", ""))
    if reference_space not in {"local", "local-floor", "bounded-floor"}:
        raise BrowserProtocolError(f"unsupported reference space: {reference_space!r}")
    raw_controllers = _mapping(raw.get("controllers"), "controllers")
    controllers: dict[str, dict[str, object]] = {}
    face: dict[str, dict[str, bool]] = {}
    for hand in HANDS:
        controllers[hand], face[hand] = normalize_controller(
            hand, raw_controllers.get(hand, {"valid": False, "buttons": [], "profiles": []})
        )
    return (
        {
            "protocol": PROTOCOL,
            "version": PROTOCOL_VERSION,
            "type": "frame",
            "session_id": session_id,
            "seq": seq,
            "client_time_ms": _finite(raw.get("client_time_ms"), "client_time_ms"),
            "reference_space": reference_space,
            "controllers": controllers,
        },
        face,
    )


@dataclasses.dataclass
class _HandGripState:
    grip_engaged: bool = False
    press_started_ms: float | None = None
    long_press_fired: bool = False

    def update_permission(
        self, *, grip_pressed: bool, client_time_ms: float, long_press_ms: float
    ) -> bool:
        if grip_pressed and self.press_started_ms is None:
            self.press_started_ms = float(client_time_ms)
            self.long_press_fired = False
        if (
            grip_pressed
            and self.press_started_ms is not None
            and not self.long_press_fired
            and client_time_ms - self.press_started_ms >= long_press_ms
        ):
            self.grip_engaged = not self.grip_engaged
            self.long_press_fired = True
            toggled = True
        else:
            toggled = False
        if not grip_pressed:
            self.press_started_ms = None
            self.long_press_fired = False
        return toggled


class GripToggleEncoder:
    """Add debounced per-hand grip-toggle state to absolute controller poses."""

    def __init__(self, *, long_press_ms: float = 1000.0) -> None:
        if not 0.0 < float(long_press_ms) < float("inf"):
            raise ValueError("long_press_ms must be positive and finite")
        self._long_press_ms = float(long_press_ms)
        self._hands = {hand: _HandGripState() for hand in HANDS}
        self._toggled_hands: tuple[str, ...] = ()

    def reset_permissions(self) -> None:
        """Clear per-arm grip permissions when the global ARM gate is toggled."""
        for state in self._hands.values():
            state.grip_engaged = False
            state.press_started_ms = None
            state.long_press_fired = False
        self._toggled_hands = ()

    def _encode_controller(
        self,
        hand: str,
        controller: Mapping[str, object],
        face: Mapping[str, bool],
        client_time_ms: float,
    ) -> dict[str, object]:
        state = self._hands[hand]
        valid = bool(controller.get("valid", False))
        output: dict[str, object] = {
            "valid": valid,
            "grip_engaged": state.grip_engaged,
            "trigger": float(cast(float, controller.get("trigger", 0.0))),
            "profiles": [
                str(profile) for profile in cast(Sequence[object], controller.get("profiles", ()))
            ],
        }
        if not valid:
            return output
        output["position"] = [
            float(cast(float, value)) for value in cast(Sequence[object], controller["position"])
        ]
        output["orientation_xyzw"] = [
            float(cast(float, value))
            for value in cast(Sequence[object], controller["orientation_xyzw"])
        ]
        return output

    def encode(
        self,
        frame: Mapping[str, object],
        face_buttons: Mapping[str, Mapping[str, bool]],
    ) -> dict[str, object]:
        client_time_ms = float(cast(float, frame["client_time_ms"]))
        raw_controllers = cast(Mapping[str, Mapping[str, object]], frame["controllers"])
        toggled_hands: list[str] = []
        encoded = dict(frame)
        controllers: dict[str, dict[str, object]] = {}
        for hand in HANDS:
            if self._hands[hand].update_permission(
                grip_pressed=bool(face_buttons[hand].get("grip", False)),
                client_time_ms=client_time_ms,
                long_press_ms=self._long_press_ms,
            ):
                toggled_hands.append(hand)
            controllers[hand] = self._encode_controller(
                hand,
                raw_controllers[hand],
                face_buttons[hand],
                client_time_ms,
            )
        encoded["controllers"] = controllers
        self._toggled_hands = tuple(toggled_hands)
        return encoded

    def consume_toggles(self) -> tuple[str, ...]:
        toggled_hands = self._toggled_hands
        self._toggled_hands = ()
        return toggled_hands

    def input_feedback(self, client_time_ms: float) -> dict[str, object]:
        pressed: list[str] = []
        progress: dict[str, float] = {}
        for hand, state in self._hands.items():
            if state.press_started_ms is None:
                continue
            control = f"{hand}.grip"
            pressed.append(control)
            progress[control] = min(
                1.0,
                max(0.0, (float(client_time_ms) - state.press_started_ms) / self._long_press_ms),
            )
        return {"pressed": pressed, "hold_progress": progress}


def _merge_input_feedback(*items: Mapping[str, object]) -> dict[str, object]:
    pressed: list[str] = []
    hold_progress: dict[str, float] = {}
    for item in items:
        for control in cast(Sequence[object], item.get("pressed", ())):
            name = str(control)
            if name and name not in pressed:
                pressed.append(name)
        hold_progress.update(
            {
                str(control): float(cast(float, value))
                for control, value in cast(
                    Mapping[str, object], item.get("hold_progress", {})
                ).items()
            }
        )
    return {"pressed": pressed, "hold_progress": hold_progress}


class ZmqBridge:
    """Own ZMQ sockets, latest-frame publishing, event retries, and acknowledgements."""

    def __init__(
        self,
        endpoint: str,
        ack_endpoint: str,
        *,
        event_retry_ttl_s: float = DEFAULT_EVENT_RETRY_TTL_S,
        event_queue_maxsize: int = DEFAULT_EVENT_QUEUE_MAXSIZE,
        acknowledgement_queue_maxsize: int = DEFAULT_ACK_QUEUE_MAXSIZE,
    ) -> None:
        if not 0.0 < event_retry_ttl_s < float("inf"):
            raise ValueError("event_retry_ttl_s must be positive and finite")
        if event_queue_maxsize <= 0:
            raise ValueError("event_queue_maxsize must be positive")
        if acknowledgement_queue_maxsize <= 0:
            raise ValueError("acknowledgement_queue_maxsize must be positive")
        self.endpoint = endpoint
        self.ack_endpoint = ack_endpoint
        self._event_retry_ttl_s = float(event_retry_ttl_s)
        self._frames: queue.Queue[dict[str, object]] = queue.Queue(maxsize=2)
        self._new_events: queue.Queue[_PendingEvent] = queue.Queue(maxsize=event_queue_maxsize)
        self.acknowledgements: queue.Queue[dict[str, object]] = queue.Queue(
            maxsize=acknowledgement_queue_maxsize
        )
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None
        self._browser_lock = threading.Lock()
        self._browser_connected = False
        self._session_id = ""
        self._source_error = ""

    @staticmethod
    def _drain(queue_obj: queue.Queue[object]) -> None:
        while True:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                return

    def _clear_protocol_state(self) -> None:
        """Discard all state that belongs to the previous bridge worker/session."""
        self._drain(self._frames)  # type: ignore[arg-type]
        self._drain(self._new_events)  # type: ignore[arg-type]
        self._drain(self.acknowledgements)  # type: ignore[arg-type]
        with self._browser_lock:
            self._browser_connected = False
            self._session_id = ""
            self._source_error = ""

    def start(self) -> None:
        existing = self._thread
        if existing is not None:
            if existing.is_alive():
                if self._stop.is_set():
                    raise RuntimeError(
                        "cannot start VR ZMQ bridge while the previous worker is stopping"
                    )
                return
            self._thread = None
        # The old worker has exited, so no concurrent consumer can observe this reset.
        self._clear_protocol_state()
        self._stop.clear()
        self._ready.clear()
        self._startup_error = None
        thread = threading.Thread(target=self._run, name="vr-zmq-bridge", daemon=True)
        self._thread = thread
        thread.start()
        if not self._ready.wait(timeout=_BRIDGE_START_TIMEOUT_S):
            stopped = self._stop_worker(thread)
            suffix = "" if stopped else "; worker did not stop"
            raise RuntimeError(f"timed out starting the VR ZMQ bridge{suffix}")
        if self._startup_error is not None:
            error = self._startup_error
            stopped = self._stop_worker(thread)
            suffix = "" if stopped else "; worker did not stop"
            raise RuntimeError(f"failed to start the VR ZMQ bridge: {error}{suffix}") from error
        if not thread.is_alive():
            self._thread = None
            raise RuntimeError("VR ZMQ bridge worker stopped during startup")

    def _stop_worker(self, thread: threading.Thread) -> bool:
        self._stop.set()
        if thread.is_alive():
            thread.join(timeout=_BRIDGE_STOP_TIMEOUT_S)
        if thread.is_alive():
            return False
        if self._thread is thread:
            self._thread = None
        return True

    def set_browser(self, connected: bool, session_id: str = "", error: str = "") -> None:
        with self._browser_lock:
            self._browser_connected = bool(connected)
            self._session_id = str(session_id)
            self._source_error = str(error)

    def submit_frame(self, frame: dict[str, object]) -> None:
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def submit_event(self, event: dict[str, object]) -> bool:
        """Queue one operator event, rejecting it when the bounded queue is full."""
        try:
            self._new_events.put_nowait(
                _PendingEvent(
                    payload=dict(event),
                    created_at=time.monotonic(),
                )
            )
        except queue.Full:
            with self._browser_lock:
                self._source_error = "operator event queue full; event dropped"
            logger.warning("Dropped VR operator event because the event queue is full")
            return False
        return True

    def submit_browser_message(self, message: dict[str, object]) -> None:
        """Queue one node-generated message for the connected WebXR browser."""
        self._store_acknowledgement(message)

    def _store_acknowledgement(self, ack: dict[str, object]) -> None:
        """Keep the newest browser feedback when the presentation queue is full."""
        try:
            self.acknowledgements.put_nowait(ack)
        except queue.Full:
            try:
                self.acknowledgements.get_nowait()
            except queue.Empty:
                pass
            try:
                self.acknowledgements.put_nowait(ack)
            except queue.Full:
                # A concurrent consumer can race the replacement. Dropping this ACK is
                # still bounded and preferable to blocking the ZMQ worker.
                logger.warning("Dropped VR acknowledgement while replacing a full queue")

    def _run(self) -> None:
        import zmq

        context = zmq.Context.instance()
        publisher = context.socket(zmq.PUB)
        acknowledgements = context.socket(zmq.PULL)
        try:
            publisher.setsockopt(zmq.SNDHWM, 64)
            publisher.bind(self.endpoint)
            acknowledgements.setsockopt(zmq.RCVHWM, 64)
            acknowledgements.bind(self.ack_endpoint)
        except Exception as error:
            self._startup_error = error
            publisher.close(linger=0)
            acknowledgements.close(linger=0)
            self._ready.set()
            return
        self._ready.set()
        pending: dict[tuple[str, int], _PendingEvent] = {}
        poller = zmq.Poller()
        poller.register(acknowledgements, zmq.POLLIN)
        next_heartbeat = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                while True:
                    try:
                        pending_event = self._new_events.get_nowait()
                    except queue.Empty:
                        break
                    event = pending_event.payload
                    key = (
                        str(event["session_id"]),
                        int(cast(int, event["event_id"])),
                    )
                    if key not in pending and len(pending) >= MAX_PENDING_EVENTS:
                        pending.pop(next(iter(pending)))
                        logger.warning("Dropped oldest pending VR event after retry queue overflow")
                    pending[key] = pending_event
                latest = None
                while True:
                    try:
                        latest = self._frames.get_nowait()
                    except queue.Empty:
                        break
                if latest is not None:
                    publisher.send_json(latest)
                with self._browser_lock:
                    browser_connected = self._browser_connected
                    session_id = self._session_id
                    source_error = self._source_error
                for key, pending_event in tuple(pending.items()):
                    if not pending_event.should_retry(
                        browser_connected=browser_connected,
                        session_id=session_id,
                        now=now,
                        retry_ttl_s=self._event_retry_ttl_s,
                    ):
                        pending.pop(key, None)
                        continue
                    if now - pending_event.last_sent_at >= 0.2:
                        publisher.send_json(pending_event.payload)
                        pending_event.last_sent_at = now
                if now >= next_heartbeat:
                    heartbeat = {
                        "protocol": PROTOCOL,
                        "version": PROTOCOL_VERSION,
                        "type": "heartbeat",
                        "session_id": session_id,
                        "browser_connected": browser_connected,
                        "source_error": source_error,
                    }
                    publisher.send_json(heartbeat)
                    next_heartbeat = now + 0.25
                if dict(poller.poll(timeout=10)).get(acknowledgements, 0) & zmq.POLLIN:
                    ack = acknowledgements.recv_json()
                    if isinstance(ack, dict):
                        key = (str(ack.get("session_id", "")), int(ack.get("event_id", -1)))
                        pending.pop(key, None)
                        self._store_acknowledgement(ack)
        except Exception as error:
            message = f"VR ZMQ bridge worker failed: {error}"
            with self._browser_lock:
                self._source_error = message
            self._stop.set()
            logger.exception(message)
        finally:
            publisher.close(linger=0)
            acknowledgements.close(linger=0)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_BRIDGE_STOP_TIMEOUT_S)
            if thread.is_alive():
                raise RuntimeError("timed out stopping VR ZMQ bridge worker")
        if thread is not None and self._thread is thread:
            self._thread = None
        self._clear_protocol_state()


class WebXrNode:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str,
        public_host: str,
        tls_cert: str,
        tls_key: str,
        bridge: ZmqBridge,
        long_press_ms: float = 1000.0,
        stats_interval_s: float = 0.25,
    ) -> None:
        if bool(tls_cert) != bool(tls_key):
            raise ValueError("TLS certificate and key must be configured together")
        self.host = host
        self.port = int(port)
        self.token = token or secrets.token_urlsafe(24)
        self.public_host = public_host
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.bridge = bridge
        self.long_press_ms = float(long_press_ms)
        self.stats_interval_s = float(stats_interval_s)
        self._socket: WebSocketServerProtocol | None = None
        self._session_id = ""

    @property
    def public_url(self) -> str:
        scheme = "https" if self.tls_cert else "http"
        host = self.public_host or self.host
        if host in {"", "0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"{scheme}://{host}:{self.port}/?token={self.token}"

    @property
    def pico_commands(self) -> tuple[str, str]:
        scheme = "https" if self.tls_cert else "http"
        query = urlencode({"token": self.token, "mode": "ar"})
        url = f"{scheme}://127.0.0.1:{self.port}/?{query}"
        return (
            f'adb -s "$PICO_SERIAL" reverse tcp:{self.port} tcp:{self.port}',
            f'adb -s "$PICO_SERIAL" shell "am start -a android.intent.action.VIEW -d \'{url}\'"',
        )

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self.tls_cert:
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.tls_cert, self.tls_key)
        return context

    def _warn_if_remote_without_tls(self) -> None:
        public_host = self.public_host.strip().lower()
        if (
            public_host
            and public_host not in {"localhost", "127.0.0.1", "::1"}
            and not self.tls_cert
        ):
            logger.warning(
                "Remote WebXR public_host=%s has no TLS certificate; use HTTPS/WSS "
                "or a TLS-terminating reverse proxy (localhost is the development exception)",
                self.public_host,
            )

    def _authorized(self, path: str) -> bool:
        value = parse_qs(urlsplit(path).query).get("token", [""])[0]
        return secrets.compare_digest(value, self.token)

    @staticmethod
    async def _probe_rtt(
        socket: WebSocketServerProtocol,
        monitor: FrameConsoleMonitor,
    ) -> None:
        while True:
            try:
                pong = await socket.ping()
                monitor.set_rtt(await asyncio.wait_for(pong, timeout=2.0))
            except ConnectionClosed:
                return
            except TimeoutError:
                logger.debug("Timed out measuring WebXR WebSocket RTT")
            except Exception:
                logger.debug("Stopped WebXR WebSocket RTT probe after failure", exc_info=True)
                return
            await asyncio.sleep(1.0)

    async def _process_request(self, path: str, _headers: object):
        route = urlsplit(path).path
        if route == "/ws":
            if self._authorized(path):
                return None
            return HTTPStatus.UNAUTHORIZED, [("Content-Type", "text/plain")], b"Unauthorized\n"
        resources = {
            "/": (STATIC_ROOT / "index.html", "text/html; charset=utf-8"),
            "/app.js": (STATIC_ROOT / "app.js", "text/javascript; charset=utf-8"),
            "/three.module.min.js": (THREE_JS, "text/javascript; charset=utf-8"),
        }
        resource = resources.get(route)
        if resource is None:
            return HTTPStatus.NOT_FOUND, [("Content-Type", "text/plain")], b"Not found\n"
        path_obj, content_type = resource
        try:
            body = path_obj.read_bytes()
        except OSError:
            logger.exception("Failed reading WebXR resource %s", path_obj)
            return HTTPStatus.INTERNAL_SERVER_ERROR, [], b"Server error\n"
        return HTTPStatus.OK, [("Content-Type", content_type), ("Cache-Control", "no-store")], body

    async def _handle_socket(self, socket: WebSocketServerProtocol) -> None:
        if self._socket is not None:
            await socket.close(code=1013, reason="another headset is connected")
            return
        self._socket = socket
        self._session_id = uuid.uuid4().hex
        event_mapper = OperatorEventMapper(long_press_ms=self.long_press_ms)
        event_mapper.reset()
        grip_encoder = GripToggleEncoder(long_press_ms=self.long_press_ms)
        monitor = FrameConsoleMonitor(interval_s=self.stats_interval_s)
        rtt_task = asyncio.create_task(self._probe_rtt(socket, monitor))
        self.bridge.set_browser(True, self._session_id)
        logger.info("WebXR connected session=%s remote=%s", self._session_id, socket.remote_address)
        error = ""
        try:
            async for payload in socket:
                browser_frame, face = normalize_browser_frame(payload, session_id=self._session_id)
                frame = grip_encoder.encode(browser_frame, face)
                events = event_mapper.update(
                    session_id=self._session_id,
                    client_time_ms=float(cast(float, frame["client_time_ms"])),
                    controllers=cast(
                        Mapping[str, Mapping[str, object]], browser_frame["controllers"]
                    ),
                    face_buttons=face,
                )
                frame["input_feedback"] = _merge_input_feedback(
                    event_mapper.input_feedback(float(cast(float, frame["client_time_ms"]))),
                    grip_encoder.input_feedback(float(cast(float, frame["client_time_ms"]))),
                )
                self.bridge.submit_frame(frame)
                for hand in grip_encoder.consume_toggles():
                    self.bridge.submit_browser_message(
                        {
                            "protocol": PROTOCOL,
                            "version": PROTOCOL_VERSION,
                            "type": "haptic",
                            "session_id": self._session_id,
                            "hand": hand,
                            "intensity": 0.6,
                            "duration_ms": 80,
                        }
                    )
                for event in events:
                    self.bridge.submit_event(event)
                if any(event.get("intent") == "arm_toggle" for event in events):
                    # B controls the global ARM gate. A subsequent B press starts
                    # from a fully disabled per-arm state instead of restoring the
                    # grip permissions that were active before the gate was closed.
                    grip_encoder.reset_permissions()
                monitor.observe(frame)
        except (BrowserProtocolError, UnicodeDecodeError) as protocol_error:
            error = str(protocol_error)
            await socket.close(code=1008, reason="invalid WebXR frame")
        except ConnectionClosed as close_error:
            error = str(close_error)
        except Exception as socket_error:
            error = str(socket_error)
            logger.exception("WebXR connection failed")
        finally:
            rtt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rtt_task
            monitor.finish()
            self.bridge.set_browser(False, self._session_id, error)
            self._socket = None
            logger.info("WebXR disconnected session=%s error=%r", self._session_id, error)

    async def _forward_acks(self) -> None:
        while True:
            try:
                ack = await asyncio.to_thread(
                    self.bridge.acknowledgements.get,
                    True,
                    0.25,
                )
            except queue.Empty:
                continue
            socket = self._socket
            if socket is None or str(ack.get("session_id", "")) != self._session_id:
                continue
            try:
                await socket.send(json.dumps(ack))
            except ConnectionClosed:
                pass

    async def serve_forever(self) -> None:
        self._warn_if_remote_without_tls()
        self.bridge.start()
        async with serve(
            self._handle_socket,
            self.host,
            self.port,
            ssl=self._ssl_context(),
            process_request=self._process_request,
            compression=None,
            max_size=128 * 1024,
            max_queue=2,
            ping_interval=10,
            ping_timeout=10,
        ) as server:
            bound = next(iter(server.sockets), None)
            if bound is not None:
                self.port = int(bound.getsockname()[1])
            logger.info("WebXR node ready: %s", self.public_url)
            reverse, launch = self.pico_commands
            logger.info("PICO commands:\n%s\n%s", reverse, launch)
            await self._forward_acks()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--public-host", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--tls-cert", default="")
    parser.add_argument("--tls-key", default="")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8765")
    parser.add_argument("--ack-endpoint", default="tcp://127.0.0.1:8766")
    parser.add_argument(
        "--event-retry-ttl-s",
        type=float,
        default=DEFAULT_EVENT_RETRY_TTL_S,
    )
    parser.add_argument("--long-press-ms", type=float, default=1000.0)
    parser.add_argument(
        "--stats-interval-s",
        type=float,
        default=0.25,
        help="single-line VR receive diagnostics interval; use 0 to disable",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bridge = ZmqBridge(
        args.endpoint,
        args.ack_endpoint,
        event_retry_ttl_s=args.event_retry_ttl_s,
    )
    node = WebXrNode(
        host=args.host,
        port=args.port,
        public_host=args.public_host,
        token=args.token,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        bridge=bridge,
        long_press_ms=args.long_press_ms,
        stats_interval_s=args.stats_interval_s,
    )
    try:
        asyncio.run(node.serve_forever())
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
