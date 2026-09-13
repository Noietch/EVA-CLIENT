"""ZMQ VR client that converts normalized controller frames to canonical EEF targets."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from teleop_client.base import (
    CanonicalEefCommand,
    TeleopClientError,
    TeleopContext,
    TeleopInputToken,
    TeleopOperatorEvent,
    TeleopResult,
    TeleopStatus,
)
from teleop_client.vr.retarget import (
    ArmRetargeter,
    GripperMapping,
    VrControllerState,
    VrPose,
    WorkspaceBounds,
)

logger = logging.getLogger(__name__)

VR_NODE_PROTOCOL = "eva.teleop.vr"
VR_NODE_PROTOCOL_VERSION = 3
_START_TIMEOUT_S = 5.0
_STOP_TIMEOUT_S = 5.0
_ACK_SEND_TIMEOUT_MS = 50
_ACK_QUEUE_MAXSIZE = 64
_EVENT_BUFFER_MAXSIZE = 64
_NEUTRAL_TRIGGER_EPSILON = 0.05
_HANDS = ("left", "right")
_EVENT_INTENTS = frozenset(
    {
        "record_toggle",
        "record_cancel",
        "arm_toggle",
        "home",
        "intervention_toggle",
        "review_left",
        "review_right",
        "review_up",
        "review_down",
        "review_select",
    }
)


class VrProtocolError(ValueError):
    """Raised when the external input node violates the normalized wire contract."""


@dataclasses.dataclass(frozen=True)
class VrPoseFrame:
    session_id: str
    seq: int
    client_time_ms: float
    received_at: float
    reference_space: str
    controllers: dict[str, VrControllerState]
    pressed_controls: tuple[str, ...] = ()
    hold_progress: tuple[tuple[str, float], ...] = ()

    def age(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.received_at)


@dataclasses.dataclass(frozen=True)
class _ArmBinding:
    group_name: str
    hand: str
    gripper_threshold: float
    retargeter: ArmRetargeter


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VrProtocolError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _finite_float(value: object, field: str) -> float:
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise VrProtocolError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise VrProtocolError(f"{field} must be finite")
    return result


def _finite_vector(value: object, size: int, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise VrProtocolError(f"{field} must be a numeric vector") from error
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise VrProtocolError(f"{field} must be a finite vector of length {size}")
    return vector


def _unit_interval(value: object, field: str) -> float:
    result = _finite_float(value, field)
    if not 0.0 <= result <= 1.0:
        raise VrProtocolError(f"{field} must be in [0, 1]")
    return result


def _parse_pose(value: object, field: str) -> VrPose:
    raw = _mapping(value, field)
    position = _finite_vector(raw.get("position"), 3, f"{field}.position")
    quaternion = _finite_vector(raw.get("orientation_xyzw"), 4, f"{field}.orientation_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise VrProtocolError(f"{field}.orientation_xyzw must be non-zero")
    return VrPose(
        position=position.astype(np.float32),
        orientation_xyzw=(quaternion / norm).astype(np.float32),
    )


def _parse_controller(hand: str, value: object) -> VrControllerState:
    raw = _mapping(value, f"controllers.{hand}")
    valid = bool(raw.get("valid", False))
    grip_engaged = bool(raw.get("grip_engaged", False))
    profiles = raw.get("profiles", ())
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise VrProtocolError(f"controllers.{hand}.profiles must be a list")
    return VrControllerState(
        hand=hand,
        valid=valid,
        grip_engaged=grip_engaged,
        pose=_parse_pose(raw, f"controllers.{hand}") if valid else None,
        trigger=_unit_interval(raw.get("trigger", 0.0), f"controllers.{hand}.trigger"),
        profiles=tuple(str(item) for item in profiles),
    )


def _parse_input_feedback(
    value: object,
) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
    if value is None:
        return (), ()
    raw = _mapping(value, "input_feedback")
    pressed_raw = raw.get("pressed", ())
    if not isinstance(pressed_raw, Sequence) or isinstance(pressed_raw, (str, bytes)):
        raise VrProtocolError("input_feedback.pressed must be a list")
    pressed = tuple(dict.fromkeys(str(item) for item in pressed_raw if str(item)))
    progress_raw = _mapping(raw.get("hold_progress", {}), "input_feedback.hold_progress")
    progress = tuple(
        (str(control), _unit_interval(value, f"input_feedback.hold_progress.{control}"))
        for control, value in progress_raw.items()
        if str(control)
    )
    return pressed, progress


def _parse_envelope(payload: bytes | str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, bytes):
        try:
            decoded: object = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VrProtocolError("VR node payload must be UTF-8 JSON") from error
    elif isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise VrProtocolError("VR node payload must be valid JSON") from error
    else:
        decoded = payload
    raw = _mapping(decoded, "payload")
    if raw.get("protocol") != VR_NODE_PROTOCOL:
        raise VrProtocolError(f"unsupported VR node protocol: {raw.get('protocol')!r}")
    if raw.get("version") != VR_NODE_PROTOCOL_VERSION:
        raise VrProtocolError(f"unsupported VR node version: {raw.get('version')!r}")
    return raw


def parse_vr_pose_frame(
    payload: bytes | str | Mapping[str, Any], *, received_at: float | None = None
) -> VrPoseFrame:
    raw = _parse_envelope(payload)
    if raw.get("type") != "frame":
        raise VrProtocolError(f"expected frame, got {raw.get('type')!r}")
    session_id = str(raw.get("session_id", "")).strip()
    if not session_id:
        raise VrProtocolError("session_id is required")
    seq = raw.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise VrProtocolError("seq must be a non-negative integer")
    reference_space = str(raw.get("reference_space", ""))
    if reference_space not in {"local", "local-floor", "bounded-floor"}:
        raise VrProtocolError(f"unsupported reference_space: {reference_space!r}")
    controllers = _mapping(raw.get("controllers"), "controllers")
    pressed_controls, hold_progress = _parse_input_feedback(raw.get("input_feedback"))
    return VrPoseFrame(
        session_id=session_id,
        seq=seq,
        client_time_ms=_finite_float(raw.get("client_time_ms"), "client_time_ms"),
        received_at=time.monotonic() if received_at is None else float(received_at),
        reference_space=reference_space,
        controllers={
            hand: _parse_controller(hand, controllers.get(hand, {"valid": False}))
            for hand in _HANDS
        },
        pressed_controls=pressed_controls,
        hold_progress=hold_progress,
    )


def parse_operator_event(
    payload: bytes | str | Mapping[str, Any], *, received_at: float | None = None
) -> TeleopOperatorEvent:
    raw = _parse_envelope(payload)
    if raw.get("type") != "event":
        raise VrProtocolError(f"expected event, got {raw.get('type')!r}")
    session_id = str(raw.get("session_id", "")).strip()
    event_id = raw.get("event_id")
    intent = str(raw.get("intent", "")).strip().lower()
    if not session_id:
        raise VrProtocolError("session_id is required")
    if not isinstance(event_id, int) or isinstance(event_id, bool) or event_id < 0:
        raise VrProtocolError("event_id must be a non-negative integer")
    if intent not in _EVENT_INTENTS:
        raise VrProtocolError(f"unsupported operator intent: {intent!r}")
    return TeleopOperatorEvent(
        session_id=session_id,
        event_id=event_id,
        intent=intent,
        received_at=time.monotonic() if received_at is None else float(received_at),
    )


def _event_ack(
    event: TeleopOperatorEvent,
    *,
    accepted: bool,
    message: str,
) -> dict[str, object]:
    return {
        "protocol": VR_NODE_PROTOCOL,
        "version": VR_NODE_PROTOCOL_VERSION,
        "type": "event_ack",
        "session_id": event.session_id,
        "event_id": event.event_id,
        "intent": event.intent,
        "accepted": bool(accepted),
        "message": str(message),
    }


def _workspace(value: object) -> WorkspaceBounds | None:
    if value in (None, {}):
        return None
    raw = _mapping(value, "workspace")
    return WorkspaceBounds(
        minimum=np.asarray(raw.get("min"), dtype=np.float64),
        maximum=np.asarray(raw.get("max"), dtype=np.float64),
    )


def _binding(value: Mapping[str, object]) -> _ArmBinding:
    group_name = str(value.get("group_name", "")).strip()
    hand = str(value.get("controller", "")).strip()
    if not group_name:
        raise ValueError("VR arm binding requires group_name")
    if hand not in _HANDS:
        raise ValueError(f"VR arm {group_name!r} must bind left or right controller")
    gripper_raw = _mapping(value.get("gripper") or {}, f"arms.{group_name}.gripper")
    return _ArmBinding(
        group_name=group_name,
        hand=hand,
        gripper_threshold=float(cast(float, gripper_raw.get("threshold", 0.5))),
        retargeter=ArmRetargeter(
            base_from_xr_rotation=np.asarray(value.get("base_from_xr_rotation"), dtype=np.float64),
            position_scale=float(cast(float, value.get("position_scale", 1.0))),
            eef_filter_alpha=float(cast(float, value.get("eef_filter_alpha", 1.0))),
            workspace=_workspace(value.get("workspace")),
            gripper=GripperMapping(
                mode=str(gripper_raw.get("mode", "binary")),
                threshold=float(cast(float, gripper_raw.get("threshold", 0.5))),
                open_value=float(cast(float, gripper_raw.get("open_value", 1.0))),
                close_value=float(cast(float, gripper_raw.get("close_value", 0.0))),
            ),
        ),
    )


class VrTeleopClient:
    """Receive normalized VR messages and produce canonical robot EEF commands."""

    def __init__(
        self,
        *,
        endpoint: str,
        ack_endpoint: str,
        input_timeout_s: float,
        arms: Sequence[Mapping[str, object]],
        heartbeat_timeout_s: float = 2.0,
    ) -> None:
        if not endpoint.startswith("tcp://") or not ack_endpoint.startswith("tcp://"):
            raise ValueError("VR node endpoints must use tcp://")
        if endpoint == ack_endpoint:
            raise ValueError("VR frame and acknowledgement endpoints must differ")
        if input_timeout_s <= 0.0 or heartbeat_timeout_s <= 0.0:
            raise ValueError("VR timeouts must be positive")
        bindings = tuple(_binding(arm) for arm in arms)
        if not bindings:
            raise ValueError("VR client requires at least one arm")
        if len({item.group_name for item in bindings}) != len(bindings):
            raise ValueError("VR arm group bindings must be unique")
        if len({item.hand for item in bindings}) != len(bindings):
            raise ValueError("VR controller bindings must be unique")
        self._endpoint = endpoint
        self._ack_endpoint = ack_endpoint
        self._input_timeout_s = float(input_timeout_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._bindings = bindings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        # Internal parser tests may inject frames before starting a network worker. Once
        # the lifecycle has been entered, poll() always requires a live worker.
        self._worker_ever_started = False
        self._startup_error: Exception | None = None
        self._ack_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=_ACK_QUEUE_MAXSIZE)
        self._events: list[TeleopOperatorEvent] = []
        self._event_cursor: dict[str, int] = {}
        self._frame: VrPoseFrame | None = None
        self._session_id = ""
        self._last_seq = -1
        self._node_seen_at: float | None = None
        self._browser_connected = False
        self._source_error = ""
        self._generation = 0
        self._rejection_reason = ""
        self._engaged_groups: tuple[str, ...] = ()
        self._held_groups = tuple(item.group_name for item in bindings)
        self._awaiting_neutral = False

    def start(self) -> None:
        existing = self._thread
        if existing is not None:
            if existing.is_alive():
                if self._stop.is_set():
                    raise RuntimeError(
                        "cannot start VR teleop client while the previous worker is stopping"
                    )
                return
            self._thread = None
        # A close timeout leaves the stop flag set while the old thread is still alive;
        # the branch above rejects that case. Once it has really exited, discard every
        # protocol artifact before creating a new session.
        self._stop.clear()
        self._ready.clear()
        with self._lock:
            self._startup_error = None
            self._clear_protocol_state_locked()
        self._clear_ack_queue()
        self._thread = threading.Thread(
            target=self._receive_loop,
            name="vr-teleop-client",
            daemon=True,
        )
        thread = self._thread
        self._worker_ever_started = True
        thread.start()
        if not self._ready.wait(timeout=_START_TIMEOUT_S):
            stopped = self._stop_worker(thread)
            suffix = "" if stopped else "; worker did not stop"
            raise RuntimeError(f"timed out starting VR teleop client{suffix}")
        with self._lock:
            startup_error = self._startup_error
        if startup_error is not None:
            self._stop_worker(thread)
            raise RuntimeError(
                f"failed to start VR teleop client: {startup_error}"
            ) from startup_error
        if not thread.is_alive():
            self._thread = None
            raise RuntimeError("VR teleop client worker stopped during startup")

    def _stop_worker(self, thread: threading.Thread) -> bool:
        self._stop.set()
        if thread.is_alive():
            thread.join(timeout=_STOP_TIMEOUT_S)
        if thread.is_alive():
            return False
        if self._thread is thread:
            self._thread = None
        return True

    def _clear_ack_queue(self) -> None:
        while True:
            try:
                self._ack_queue.get_nowait()
            except queue.Empty:
                return

    def _clear_protocol_state_locked(
        self, *, source_error: str = "", require_neutral: bool = True
    ) -> None:
        """Clear input/session state without touching the bounded ACK queue."""
        self._generation += 1
        for binding in self._bindings:
            binding.retargeter.reset()
        self._frame = None
        self._session_id = ""
        self._last_seq = -1
        self._node_seen_at = None
        self._browser_connected = False
        self._source_error = source_error
        self._rejection_reason = ""
        self._events.clear()
        self._event_cursor.clear()
        self._engaged_groups = ()
        self._held_groups = tuple(item.group_name for item in self._bindings)
        self._awaiting_neutral = bool(require_neutral)

    def _fail_closed_locked(self, source_error: str) -> None:
        """Drop all motion/input state after a worker or protocol safety failure."""
        self._clear_protocol_state_locked(source_error=source_error)
        self._stop.set()

    def _enqueue_ack_locked(self, ack: dict[str, object]) -> None:
        """Enqueue feedback without ever blocking the receive worker.

        ACK presentation is lossy by design: when full, discard the oldest item and
        retain the newest feedback. The source error records that feedback was lost.
        """
        try:
            self._ack_queue.put_nowait(ack)
            return
        except queue.Full:
            pass
        try:
            self._ack_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._ack_queue.put_nowait(ack)
        except queue.Full:
            logger.warning("Dropped VR acknowledgement while replacing a full queue")
        self._source_error = "VR acknowledgement queue full; dropped oldest feedback"

    def _receive_loop(self) -> None:
        receiver: Any | None = None
        acknowledgements: Any | None = None
        started = False
        try:
            import zmq

            context = zmq.Context.instance()
            receiver_socket = context.socket(zmq.SUB)
            receiver = receiver_socket
            receiver_socket.setsockopt(zmq.SUBSCRIBE, b"")
            receiver_socket.setsockopt(zmq.RCVHWM, 256)
            receiver_socket.connect(self._endpoint)
            acknowledgement_socket = context.socket(zmq.PUSH)
            acknowledgements = acknowledgement_socket
            acknowledgement_socket.setsockopt(zmq.SNDHWM, 64)
            acknowledgement_socket.setsockopt(zmq.SNDTIMEO, _ACK_SEND_TIMEOUT_MS)
            acknowledgement_socket.connect(self._ack_endpoint)
            poller = zmq.Poller()
            poller.register(receiver_socket, zmq.POLLIN)
            started = True
            self._ready.set()
            pending_ack: dict[str, object] | None = None
            while not self._stop.is_set():
                while True:
                    if pending_ack is None:
                        try:
                            pending_ack = self._ack_queue.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        acknowledgement_socket.send_json(pending_ack)
                    except zmq.Again:
                        break
                    pending_ack = None
                if not dict(poller.poll(timeout=50)).get(receiver_socket, 0) & zmq.POLLIN:
                    continue
                try:
                    self._ingest(receiver_socket.recv())
                except VrProtocolError as error:
                    with self._lock:
                        self._generation += 1
                        self._source_error = str(error)
                    logger.warning("Rejected VR node message: %s", error)
        except Exception as error:
            message = f"VR teleop client worker failed: {error}"
            with self._lock:
                if not started:
                    self._startup_error = error
                self._fail_closed_locked(message)
            if started:
                logger.exception(message)
        finally:
            if started and not self._stop.is_set():
                with self._lock:
                    self._fail_closed_locked(
                        self._source_error or "VR teleop client worker stopped unexpectedly"
                    )
            self._ready.set()
            if receiver is not None:
                receiver.close(linger=0)
            if acknowledgements is not None:
                acknowledgements.close(linger=0)

    def _ingest(self, payload: bytes) -> None:
        raw = _parse_envelope(payload)
        message_type = raw.get("type")
        now = time.monotonic()
        if message_type == "heartbeat":
            with self._lock:
                previous_connected = self._browser_connected
                previous_error = self._source_error
                self._node_seen_at = now
                self._browser_connected = bool(raw.get("browser_connected", False))
                self._source_error = str(raw.get("source_error", ""))
                if (
                    self._browser_connected != previous_connected
                    or self._source_error != previous_error
                ):
                    self._generation += 1
                if not self._browser_connected:
                    source_error = self._source_error
                    self._clear_protocol_state_locked(source_error=source_error)
                    self._node_seen_at = now
                    self._browser_connected = False
            return
        if message_type == "frame":
            frame = parse_vr_pose_frame(raw, received_at=now)
            with self._lock:
                previous_connected = self._browser_connected
                previous_error = self._source_error
                if frame.session_id != self._session_id:
                    previous_session_id = self._session_id
                    self._generation += 1
                    for binding in self._bindings:
                        binding.retargeter.reset()
                    self._session_id = frame.session_id
                    self._last_seq = -1
                    self._frame = None
                    self._rejection_reason = ""
                    self._events.clear()
                    self._event_cursor.clear()
                    self._engaged_groups = ()
                    self._held_groups = tuple(item.group_name for item in self._bindings)
                    if previous_session_id:
                        self._awaiting_neutral = True
                if frame.seq <= self._last_seq:
                    return
                self._last_seq = frame.seq
                self._frame = frame
                if not previous_connected or previous_error:
                    self._generation += 1
                self._node_seen_at = now
                self._browser_connected = True
                self._source_error = ""
            return
        if message_type == "event":
            event = parse_operator_event(raw, received_at=now)
            with self._lock:
                frame = self._frame
                if (
                    not self._browser_connected
                    or frame is None
                    or event.session_id != frame.session_id
                    or frame.age(now) > self._input_timeout_s
                ):
                    self._enqueue_ack_locked(
                        _event_ack(
                            event,
                            accepted=False,
                            message="VR event does not belong to the active input session",
                        )
                    )
                    return
                cursor = self._event_cursor.get(event.session_id, -1)
                if event.event_id <= cursor:
                    return
                if len(self._events) >= _EVENT_BUFFER_MAXSIZE:
                    message = "VR event buffer full; event rejected and input failed closed"
                    self._enqueue_ack_locked(_event_ack(event, accepted=False, message=message))
                    self._fail_closed_locked(message)
                    return
                self._event_cursor[event.session_id] = event.event_id
                self._events.append(event)
                self._node_seen_at = now
            return
        raise VrProtocolError(f"unsupported VR node message type: {message_type!r}")

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and not self._stop_worker(thread):
            raise RuntimeError("timed out stopping VR teleop client worker")
        self.reset()
        with self._lock:
            self._startup_error = None
            self._clear_protocol_state_locked(require_neutral=False)
        self._clear_ack_queue()

    def _frame_is_neutral(self, frame: VrPoseFrame) -> bool:
        if frame.pressed_controls:
            return False
        return all(
            frame.controllers[binding.hand].valid
            and frame.controllers[binding.hand].pose is not None
            and not frame.controllers[binding.hand].grip_engaged
            and frame.controllers[binding.hand].trigger
            <= min(binding.gripper_threshold, _NEUTRAL_TRIGGER_EPSILON)
            for binding in self._bindings
        )

    def reset(self, *, require_neutral: bool = False) -> None:
        with self._lock:
            self._generation += 1
            for binding in self._bindings:
                binding.retargeter.reset()
            self._rejection_reason = ""
            self._engaged_groups = ()
            self._held_groups = tuple(item.group_name for item in self._bindings)
            self._awaiting_neutral = bool(require_neutral)

    def poll(self, context: TeleopContext) -> TeleopResult:
        with self._lock:
            worker = self._thread
            if self._worker_ever_started and (
                worker is None or not worker.is_alive() or self._stop.is_set()
            ):
                raise TeleopClientError("VR input worker is not running")
            frame = self._frame
            browser_connected = self._browser_connected
            source_token = TeleopInputToken(
                self._generation,
                frame_received_at=frame.received_at if frame is not None else None,
            )
        if frame is None or not browser_connected:
            raise TeleopClientError("VR input is not connected")
        age = frame.age(context.now)
        if age > self._input_timeout_s:
            raise TeleopClientError(f"VR input frame stale for {age * 1000.0:.0f} ms")
        measured = np.asarray(context.measured_eef, dtype=np.float32).reshape(-1)
        expected = 8 * len(self._bindings)
        if measured.shape != (expected,) or not np.all(np.isfinite(measured)):
            raise TeleopClientError(
                f"VR measured EEF must be finite shape ({expected},), got {measured.shape}"
            )
        targets: list[np.ndarray] = []
        active_arms: list[bool] = []
        with self._lock:
            if self._awaiting_neutral:
                self._engaged_groups = ()
                self._held_groups = tuple(item.group_name for item in self._bindings)
                if not self._frame_is_neutral(frame):
                    return TeleopResult.idle(source_token=source_token)
                self._awaiting_neutral = False
                return TeleopResult.idle(source_token=source_token)

            # Convert one fresh frame into canonical EEF targets
            for index, binding in enumerate(self._bindings):
                arm_measured = measured[index * 8 : (index + 1) * 8]
                try:
                    result = binding.retargeter.update(
                        frame.controllers[binding.hand], arm_measured
                    )
                except ValueError as error:
                    self._rejection_reason = str(error)
                    self._engaged_groups = ()
                    self._held_groups = tuple(item.group_name for item in self._bindings)
                    return TeleopResult.rejected(self._rejection_reason, source_token=source_token)
                target = np.asarray(result.target_eef, dtype=np.float32).copy()
                targets.append(target)
                active_arms.append(bool(result.active))
            self._engaged_groups = tuple(
                binding.group_name
                for binding, active in zip(self._bindings, active_arms, strict=True)
                if active
            )
            self._held_groups = tuple(
                binding.group_name
                for binding, active in zip(self._bindings, active_arms, strict=True)
                if not active
            )
            self._rejection_reason = ""
        return TeleopResult.from_command(
            CanonicalEefCommand(
                value=np.concatenate(targets).astype(np.float32),
                active_arms=tuple(active_arms),
            ),
            source_token=source_token,
        )

    def validate_result(self, result: TeleopResult) -> bool:
        """Return whether a polled result still belongs to the live input generation."""
        token = result.source_token
        if token is None:
            return False
        now = time.monotonic()
        with self._lock:
            worker = self._thread
            source_is_fresh = bool(
                token.frame_received_at is not None
                and now - token.frame_received_at <= self._input_timeout_s
            )
            return bool(
                token.generation == self._generation
                and (
                    not self._worker_ever_started
                    or (worker is not None and worker.is_alive() and not self._stop.is_set())
                )
                and self._browser_connected
                and self._frame is not None
                and source_is_fresh
            )

    def drain_events(self) -> tuple[TeleopOperatorEvent, ...]:
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            return events

    def acknowledge_event(
        self,
        event: TeleopOperatorEvent,
        *,
        accepted: bool,
        message: str,
    ) -> None:
        with self._lock:
            self._enqueue_ack_locked(_event_ack(event, accepted=accepted, message=message))

    def status(self, now: float | None = None) -> TeleopStatus:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            frame = self._frame
            node_seen_at = self._node_seen_at
            browser_connected = self._browser_connected
            worker = self._thread
            worker_healthy = not self._worker_ever_started or bool(
                worker is not None and worker.is_alive() and not self._stop.is_set()
            )
            frame_fresh = bool(
                frame is not None and frame.age(current) <= self._input_timeout_s
            )
            connected = bool(
                worker_healthy
                and browser_connected
                and node_seen_at is not None
                and current - node_seen_at <= self._heartbeat_timeout_s
                and frame_fresh
            )
            input_age_ms = None if frame is None else frame.age(current) * 1000.0
            neutral = bool(
                connected
                and frame is not None
                and frame.age(current) <= self._input_timeout_s
                and self._frame_is_neutral(frame)
            )
            authorized_groups = tuple(
                binding.group_name
                for binding in self._bindings
                if frame is not None and frame.controllers[binding.hand].grip_engaged
            )
            return TeleopStatus(
                source_type="vr_webxr",
                connected=connected,
                input_age_ms=input_age_ms,
                engaged_groups=self._engaged_groups,
                held_groups=self._held_groups,
                authorized_groups=authorized_groups,
                pressed_controls=() if frame is None else frame.pressed_controls,
                hold_progress=() if frame is None else frame.hold_progress,
                neutral=neutral,
                source_error=self._rejection_reason or self._source_error,
            )


__all__ = [
    "VR_NODE_PROTOCOL",
    "VR_NODE_PROTOCOL_VERSION",
    "VrPoseFrame",
    "VrProtocolError",
    "VrTeleopClient",
    "parse_operator_event",
    "parse_vr_pose_frame",
]
