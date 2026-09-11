# pyright: reportAttributeAccessIssue=false
"""ZeroMQ transport + its wire protocol.

The client SUBscribes to an execution-layer node's observation stream (each message
is a complete, source-aligned snapshot stamped with a capture time ``t``) and
PUBlishes actions back. Because each observation message already bundles every
camera and both arms at one timestamp, alignment is just "take the freshest
complete frame" — drain the socket and keep the newest.

The wire format (observation / action structs + msgpack-numpy pack/unpack) lives
here too: it is part of the zmq middleware, not a standalone backend. Execution-
layer nodes import it from ``transport.zmq``. ``zmq`` itself is imported lazily so
this module never breaks the package import on hosts without pyzmq; msgpack-numpy
is already a project dependency (used by the policy clients), so the wire helpers
import cleanly everywhere.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np
from openpi_client import msgpack_numpy

from core.config import ConfigDict
from core.registry import TRANSPORT_REGISTRY
from core.types import CollectionRawBatch, CollectionRawSample, Observation, RawCollectionSnapshot
from transport.base import HilStatus, TransportBridge
from transport.utils import ImageRateTracker, StreamFreshness

if TYPE_CHECKING:
    from robots.base import Robot

logger = logging.getLogger(__name__)

_PACKER = msgpack_numpy.Packer()
COLLECTION_START_TARGET = "collect_start"
COLLECTION_STOP_TARGET = "collect_stop"
IMAGE_REQUEST_TARGET = "image_request"
HIL_START_TARGET = "hil_start"
HIL_STOP_TARGET = "hil_stop"
COLLECTION_CONTROL_REPEATS = 5
COLLECTION_CONTROL_INTERVAL_S = 0.02
COLLECTION_SOCKET_DRAIN_MAX = 64
COLLECTION_RECEIVE_HWM = 64
HIL_ACK_TIMEOUT_S = 1.0


# --- wire protocol -------------------------------------------------------------


@dataclasses.dataclass
class WireObservation:
    """One timestamped sensor frame published by an execution-layer node.

    Args:
        t: capture timestamp in seconds (monotonic at the source).
        images: observation_key -> [H, W, 3] uint8 array.
        state: actuator group_name -> joint position array.
        action: optional teleop action aligned with this frame (data-collection
            mode only), used as action_qpos by the collection API.
        eef: optional actuator group_name -> end-effector pose array.
        action_eef: optional teleop end-effector action aligned with this frame.
    """

    t: float
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    action: np.ndarray | None = None
    eef: dict[str, np.ndarray] | None = None
    action_eef: np.ndarray | None = None
    hil_supported: bool = False
    hil_active: bool = False
    hil_error: str = ""
    operator_event: str = ""
    operator_event_id: int = 0


@dataclasses.dataclass
class WireAction:
    """One action command sent from the client to an execution-layer node.

    Args:
        t: client-side send timestamp in seconds.
        action: flat action vector (joint space, full robot DOF).
        target: "real", "sim", "collect_start", or "collect_stop".
        mode: Optional control mode. Collection start uses the teleop control
            source; HIL start uses the intervention mode.
    """

    t: float
    action: np.ndarray
    target: str = "real"
    mode: str | None = None


def pack_observation(obs: WireObservation) -> bytes:
    """Serialize a WireObservation to msgpack-numpy bytes for the SUB/PUB wire.

    Casts every state/eef/action array to float32 and includes optional fields only
    when present. Image arrays are packed as-is ([H, W, 3] uint8).

    Args:
        obs: the observation frame to encode.

    Returns:
        msgpack-packed payload bytes.
    """
    payload: dict = {
        "t": float(obs.t),
        "images": obs.images,
        "state": {k: np.asarray(v, dtype=np.float32) for k, v in obs.state.items()},
    }
    if obs.action is not None:
        payload["action"] = np.asarray(obs.action, dtype=np.float32)
    if obs.eef is not None:
        payload["eef"] = {k: np.asarray(v, dtype=np.float32) for k, v in obs.eef.items()}
    if obs.action_eef is not None:
        payload["action_eef"] = np.asarray(obs.action_eef, dtype=np.float32)
    payload["hil_supported"] = bool(obs.hil_supported)
    payload["hil_active"] = bool(obs.hil_active)
    if obs.hil_error:
        payload["hil_error"] = str(obs.hil_error)
    if obs.operator_event:
        payload["operator_event"] = str(obs.operator_event)
        payload["operator_event_id"] = int(obs.operator_event_id)
    return _PACKER.pack(payload)


def unpack_observation(payload: bytes) -> WireObservation:
    """Decode msgpack-numpy bytes back into a WireObservation.

    Required fields (t, images, state) are always present; optional fields decode to
    None when absent. Images are restored as uint8 [H, W, 3]; numeric arrays float32.

    Args:
        payload: msgpack-packed observation bytes from pack_observation.

    Returns:
        The reconstructed WireObservation.
    """
    raw = msgpack_numpy.unpackb(payload)
    action = raw.get("action")
    eef = raw.get("eef")
    action_eef = raw.get("action_eef")
    return WireObservation(
        t=float(raw["t"]),
        images={k: np.asarray(v) for k, v in raw["images"].items()},
        state={k: np.asarray(v, dtype=np.float32) for k, v in raw["state"].items()},
        action=None if action is None else np.asarray(action, dtype=np.float32),
        eef=(None if eef is None else {k: np.asarray(v, dtype=np.float32) for k, v in eef.items()}),
        action_eef=None if action_eef is None else np.asarray(action_eef, dtype=np.float32),
        hil_supported=bool(raw.get("hil_supported", False)),
        hil_active=bool(raw.get("hil_active", False)),
        hil_error=str(raw.get("hil_error", "")),
        operator_event=str(raw.get("operator_event", "")),
        operator_event_id=int(raw.get("operator_event_id", 0)),
    )


def _inspect_observation_payload(payload: bytes) -> tuple[float, frozenset[str]]:
    """Read capture metadata without materializing packed NumPy arrays."""
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(payload)
    timestamp = None
    image_keys: frozenset[str] = frozenset()
    for _ in range(unpacker.read_map_header()):
        key = unpacker.unpack()
        if key == "t":
            timestamp = float(unpacker.unpack())
            continue
        if key != "images":
            unpacker.skip()
            continue
        keys = set()
        for _ in range(unpacker.read_map_header()):
            keys.add(str(unpacker.unpack()))
            unpacker.skip()
        image_keys = frozenset(keys)
    if timestamp is None:
        raise ValueError("wire observation is missing capture timestamp")
    return timestamp, image_keys


def _observation_timestamp(payload: bytes) -> float:
    """Read the first wire field (``t``) from a small payload prefix."""
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(payload[:64])
    unpacker.read_map_header()
    if unpacker.unpack() == "t":
        return float(unpacker.unpack())
    timestamp, _ = _inspect_observation_payload(payload)
    return timestamp


def pack_action(action: WireAction) -> bytes:
    """Serialize a WireAction (t, float32 action vector, target) to msgpack bytes."""
    payload = {
        "t": float(action.t),
        "action": np.asarray(action.action, dtype=np.float32),
        "target": action.target,
    }
    if action.mode is not None:
        payload["mode"] = action.mode
    return _PACKER.pack(payload)


def unpack_action(payload: bytes) -> WireAction:
    """Decode msgpack bytes into a WireAction (target defaults to "real")."""
    raw = msgpack_numpy.unpackb(payload)
    return WireAction(
        t=float(raw["t"]),
        action=np.asarray(raw["action"], dtype=np.float32),
        target=str(raw.get("target", "real")),
        mode=None if raw.get("mode") is None else str(raw["mode"]),
    )


# --- transport -----------------------------------------------------------------


class _WireCaptureJournal:
    """Byte-preserving journal whose file writes stay off the capture thread."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        self._file = tempfile.NamedTemporaryFile(
            prefix="eva-zmq-wire-",
            dir=directory,
            buffering=0,
        )
        self._state = threading.Condition()
        self._file_lock = threading.Lock()
        self._pending: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._size = 0
        self._written_size = 0
        self._error: BaseException | None = None
        self._finished = False
        self._writer = threading.Thread(
            target=self._write_pending,
            name="eva-zmq-wire-journal",
            daemon=True,
        )
        self._writer.start()

    def append(self, payload: bytes) -> _WireJournalEntry:
        with self._state:
            if self._finished:
                raise RuntimeError("ZMQ wire journal is already finished")
            if self._error is not None:
                raise RuntimeError("ZMQ wire journal writer failed") from self._error
            offset = self._size
            self._size += len(payload)
            self._pending.put(payload)
        return _WireJournalEntry(self, offset, len(payload))

    def finish(self) -> None:
        with self._state:
            if self._finished:
                return
            self._finished = True
            self._pending.put(None)

    def _write_pending(self) -> None:
        try:
            while True:
                payload = self._pending.get()
                if payload is None:
                    return
                with self._file_lock:
                    self._file.seek(self._written_size)
                    written = self._file.write(payload)
                if written != len(payload):
                    raise OSError(f"short write to ZMQ wire journal: {written}/{len(payload)}")
                with self._state:
                    self._written_size += written
                    self._state.notify_all()
        except BaseException as error:
            with self._state:
                self._error = error
                self._state.notify_all()

    def read(self, offset: int, size: int) -> bytes:
        with self._state:
            self._state.wait_for(
                lambda: self._error is not None or self._written_size >= offset + size
            )
            if self._error is not None:
                raise RuntimeError("ZMQ wire journal writer failed") from self._error
        with self._file_lock:
            self._file.seek(offset)
            payload = self._file.read(size)
        if len(payload) != size:
            raise OSError(f"short read from ZMQ wire journal: {len(payload)}/{size}")
        return payload


@dataclasses.dataclass(frozen=True)
class _WireJournalEntry:
    journal: _WireCaptureJournal
    offset: int
    size: int

    def read(self) -> bytes:
        return self.journal.read(self.offset, self.size)


class _ObservationReader:
    """One independent SUB socket + the WireObservation->Observation conversion.

    Each reader owns its own SUB socket connected to the same PUB endpoint, so the
    pub/sub fan-out delivers a full copy of every observation to each reader — a
    visualization reader and the control-loop reader never steal frames from one
    another. A lock guards drain/_latest because the console serves frames from
    multiple HTTP worker threads against a single reader.
    """

    def __init__(
        self,
        config: ConfigDict,
        robot: Robot,
        zmq_mod,
        preserve_collection_backlog: bool = False,
    ) -> None:
        self._robot = robot
        self._zmq = zmq_mod
        self._ctx = zmq_mod.Context.instance()
        self._sub_endpoint = config.transport.sub_endpoint
        self._latest: WireObservation | None = None
        self._latest_complete_image: WireObservation | None = None
        self._image_revision = 0
        self._latest_images: dict[str, np.ndarray] = {}
        self._preserve_collection_backlog = preserve_collection_backlog
        self._collection_queue: collections.deque[WireObservation] = collections.deque()
        self._raw_collection_queue: collections.deque[bytes] = collections.deque()
        self._freshness = StreamFreshness()
        self._image_rate = ImageRateTracker()
        self._lock = threading.Lock()
        self._closed = False
        self._collection_journal: _WireCaptureJournal | None = None
        self._operator_event_initialized = False
        self._last_operator_event_id = 0

        self._disabled_cameras = set(config.transport.disabled_cameras)
        self._disabled_groups = set(config.transport.disabled_groups)
        self._image_mode = str(getattr(config.transport, "image_mode", "stream"))
        # Per-group slice of the robot's initial qpos, used to fill state for
        # groups disabled in this deployment (e.g. an unused arm).
        self._group_initial_qpos: dict[str, np.ndarray] = {}
        offset = 0
        for group in robot.actuator_groups:
            self._group_initial_qpos[group.name] = np.asarray(
                robot.initial_qpos[offset : offset + group.dof], dtype=np.float32
            )
            offset += group.dof

        self._sub = self._create_subscriber()

    def _create_subscriber(self) -> Any:
        subscriber = self._ctx.socket(self._zmq.SUB)
        if self._preserve_collection_backlog:
            subscriber.setsockopt(self._zmq.RCVHWM, COLLECTION_RECEIVE_HWM)
        subscriber.setsockopt(self._zmq.SUBSCRIBE, b"")
        subscriber.setsockopt(self._zmq.RCVTIMEO, 0)
        subscriber.connect(self._sub_endpoint)
        return subscriber

    def _drain_latest(self) -> WireObservation | None:
        """Pop all queued SUB messages, keeping only the newest by timestamp."""
        with self._lock:
            if self._closed:
                return self._latest
            newest = self._latest
            got_message = False
            while True:
                try:
                    payload = self._sub.recv(self._zmq.NOBLOCK)
                except self._zmq.Again:
                    break
                except self._zmq.ZMQError:
                    # A request can race with close() during application shutdown.
                    if self._closed:
                        break
                    raise
                got_message = True
                obs = unpack_observation(payload)
                self._image_rate.mark_many(obs.images.keys(), obs.t)
                self._cache_images_locked(obs)
                if self._has_complete_images(obs):
                    self._latest_complete_image = obs
                    self._image_revision += 1
                if newest is None or obs.t >= newest.t:
                    newest = obs
            if got_message:
                self._freshness.mark()
            self._latest = newest
            return newest

    def _has_complete_images(self, obs: WireObservation) -> bool:
        disabled_cameras = getattr(self, "_disabled_cameras", set())
        required = {
            camera.observation_key
            for camera in self._robot.observation_schema.cameras
            if camera.observation_key not in disabled_cameras
        }
        return required.issubset(obs.images)

    def image_revision(self) -> int:
        """Return the number of complete image observations received by this reader."""
        with self._lock:
            return self._image_revision

    def get_frame_after_image_revision(self, revision: int) -> Observation | None:
        """Return a complete image observation newer than ``revision``."""
        self._drain_latest()
        with self._lock:
            if self._image_revision <= revision or self._latest_complete_image is None:
                return None
            wire_obs = self._latest_complete_image
        return self._wire_to_observation(wire_obs)

    def _drain_collection_queue(self) -> WireObservation | None:
        with self._lock:
            got_message = False
            for _ in range(COLLECTION_SOCKET_DRAIN_MAX):
                try:
                    payload = self._sub.recv(self._zmq.NOBLOCK)
                except self._zmq.Again:
                    break
                got_message = True
                obs = unpack_observation(payload)
                self._image_rate.mark_many(obs.images.keys(), obs.t)
                self._cache_images_locked(obs)
                self._collection_queue.append(obs)
                if self._latest is None or obs.t >= self._latest.t:
                    self._latest = obs
            if got_message:
                self._freshness.mark()
            if not self._collection_queue:
                return None
            return self._collection_queue.popleft()

    def seconds_since_last_recv(self) -> float | None:
        """Seconds since this reader's socket last received a message, or None."""
        return self._freshness.seconds_since()

    def image_min_hz(self) -> float | None:
        """Minimum image receive rate across enabled cameras."""
        keys = (
            camera.observation_key
            for camera in self._robot.observation_schema.cameras
            if camera.observation_key not in self._disabled_cameras
        )
        return self._image_rate.min_hz(keys)

    def hil_status(self) -> HilStatus:
        observation = self._drain_latest()
        if observation is None:
            return HilStatus(supported=False, error="No hardware status received")
        return HilStatus(
            supported=observation.hil_supported,
            active=observation.hil_active,
            error=observation.hil_error,
        )

    def poll_operator_event(self) -> str | None:
        """Return one new edge-triggered hardware event, never a stale replay."""
        observation = self._drain_latest()
        if observation is None:
            return None
        event_id = observation.operator_event_id
        if not self._operator_event_initialized:
            self._operator_event_initialized = True
            self._last_operator_event_id = event_id
            return None
        if event_id < self._last_operator_event_id:
            # Execution node restarted and reset its sequence. Establish a fresh
            # baseline so an event retained in the newest frame cannot fire late.
            self._last_operator_event_id = event_id
            return None
        if event_id == self._last_operator_event_id or not observation.operator_event:
            return None
        self._last_operator_event_id = event_id
        return observation.operator_event

    def clear_collection_backlog(self) -> float | None:
        """Discard pre-episode data, including messages still in the ZMQ/TCP pipe."""
        with self._lock:
            journal = getattr(self, "_collection_journal", None)
            self._collection_journal = None
            if journal is not None:
                journal.finish()
            if self._preserve_collection_backlog:
                # Draining recv(NOBLOCK) only empties messages currently visible to
                # Python. Old messages can arrive later from the upstream pipe.
                self._sub.close(linger=0)
                self._sub = self._create_subscriber()
                self._collection_queue.clear()
                self._raw_collection_queue.clear()
                self._latest = None
                logger.info("[COLLECTION_BACKLOG] reset subscriber before new episode")
                return None
            cutoff = None
            if self._collection_queue:
                cutoff = max(obs.t for obs in self._collection_queue)
            last_payload = self._raw_collection_queue[-1] if self._raw_collection_queue else None
            while True:
                try:
                    last_payload = self._sub.recv(self._zmq.NOBLOCK)
                except self._zmq.Again:
                    break
            if last_payload is not None:
                try:
                    payload_time = _observation_timestamp(last_payload)
                except Exception:
                    payload_time = None
                if payload_time is not None:
                    cutoff = payload_time if cutoff is None else max(cutoff, payload_time)
            self._collection_queue.clear()
            self._raw_collection_queue.clear()
            return cutoff

    def get_frame(self) -> Observation | None:
        """Return the freshest snapshot as a canonical Observation.

        Drains the SUB socket to the newest WireObservation, then keys images by
        observation_key and concatenates state in state_composition order (disabled
        groups filled from initial qpos, disabled cameras skipped). When the snapshot
        carries a teleop action (collection mode), the Observation's ``action_qpos``
        is set.

        Returns:
            Observation with images [H, W, 3] uint8 and state_qpos
            [state_dim] float32, or None if no frame is available or a required
            camera/group is missing.
        """
        wire_obs = self._drain_latest()
        if wire_obs is None:
            return None

        with self._lock:
            latest_images = self._latest_images.copy()
        images: dict[str, np.ndarray] = {}
        for camera in self._robot.observation_schema.cameras:
            if camera.observation_key in self._disabled_cameras:
                continue
            image = latest_images.get(camera.observation_key)
            if image is None:
                return None
            images[camera.observation_key] = np.asarray(image)

        state_parts: list[np.ndarray] = []
        for group_name in self._robot.observation_schema.state_composition:
            if group_name in self._disabled_groups:
                state_parts.append(self._group_initial_qpos[group_name])
                continue
            part = wire_obs.state.get(group_name)
            if part is None:
                return None
            state_parts.append(np.asarray(part, dtype=np.float32))
        state = np.concatenate(state_parts, axis=0)

        if wire_obs.action is not None:
            # Collection mode: the execution layer bundled the teleop action into
            # this snapshot (already time-aligned), so carry it through for the
            # recorder. Eval never sets action, so it stays None.
            return Observation(
                images=images,
                state_qpos=state,
                action_qpos=np.asarray(wire_obs.action, dtype=np.float32),
                timestamp=wire_obs.t,
            )
        return Observation(images=images, state_qpos=state, timestamp=wire_obs.t)

    def get_camera_frame(self, key: str) -> np.ndarray | None:
        """Single camera's image [H, W, 3] uint8 from the freshest snapshot, or None.

        Returns None if no frame is available, the camera is disabled, or that key
        has not appeared in the stream yet. Partial camera snapshots are cached per
        key so visualization can keep all camera panes alive when the execution
        layer publishes cameras independently.
        """
        wire_obs = self._drain_latest()
        if wire_obs is not None:
            self._cache_images(wire_obs)
        if key in self._disabled_cameras:
            return None
        with self._lock:
            image = self._latest_images.get(key)
        return None if image is None else np.asarray(image)

    def get_camera_keys(self) -> list[str]:
        """Enabled camera keys the visualization stream should expose."""
        wire_obs = self._drain_latest()
        if wire_obs is not None:
            self._cache_images(wire_obs)
        return [
            camera.observation_key
            for camera in self._robot.observation_schema.cameras
            if camera.observation_key not in self._disabled_cameras
        ]

    def _cache_images(self, wire_obs: WireObservation) -> None:
        with self._lock:
            self._cache_images_locked(wire_obs)

    def _cache_images_locked(self, wire_obs: WireObservation) -> None:
        for key, image in wire_obs.images.items():
            if key not in self._disabled_cameras:
                self._latest_images[key] = np.asarray(image)

    def get_latest_qpos(self) -> np.ndarray | None:
        """Joint state from the freshest snapshot concatenated across groups.

        Disabled groups are filled from their initial qpos. Returns None if no frame
        is available or an enabled group's state is missing.

        Returns:
            [qpos_dim] float32 joint position vector, or None.
        """
        wire_obs = self._drain_latest()
        if wire_obs is None:
            return None
        parts: list[np.ndarray] = []
        for group in self._robot.actuator_groups:
            if group.name in self._disabled_groups:
                parts.append(self._group_initial_qpos[group.name])
                continue
            part = wire_obs.state.get(group.name)
            if part is None:
                return None
            parts.append(np.asarray(part, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def _concat_group_fields(
        self, mapping: dict[str, np.ndarray] | None, groups: tuple | None = None
    ) -> np.ndarray | None:
        if mapping is None:
            return None
        parts = []
        for group in groups if groups is not None else self._robot.actuator_groups:
            part = mapping.get(group.name)
            if part is None:
                return None
            parts.append(np.asarray(part, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def _wire_to_observation(self, wire_obs: WireObservation) -> Observation:
        state_qpos = self._concat_group_fields(wire_obs.state)
        if state_qpos is None:
            raise ValueError(
                "wire observation is missing required joint state for an actuator group"
            )
        return Observation(
            images={k: np.asarray(v) for k, v in wire_obs.images.items()},
            state_qpos=state_qpos,
            state_eef=self._concat_group_fields(wire_obs.eef, self._robot.arm_groups),
            action_qpos=wire_obs.action,
            action_eef=wire_obs.action_eef,
            timestamp=wire_obs.t,
        )

    def get_collection_frame(self) -> Observation | None:
        """Next data-collection frame from the snapshot stream, or None.

        Pops the oldest queued frame when preserving the backlog, otherwise takes the
        freshest snapshot, and converts it to an Observation (images plus the
        state_qpos/state_eef and action_qpos/action_eef carried on the wire).
        """
        if getattr(self, "_preserve_collection_backlog", False):
            wire_obs = self._drain_collection_queue()
        else:
            wire_obs = self._drain_latest()
        if wire_obs is None:
            return None
        return self._wire_to_observation(wire_obs)

    def _drain_raw_collection(self) -> bytes | None:
        """Return one raw payload while honoring the collection backlog mode."""
        with self._lock:
            if getattr(self, "_preserve_collection_backlog", False):
                if self._raw_collection_queue:
                    payload = self._raw_collection_queue.popleft()
                else:
                    try:
                        payload = self._sub.recv(self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        return None
            else:
                payload = self._raw_collection_queue.pop() if self._raw_collection_queue else None
                self._raw_collection_queue.clear()
                got_message = payload is not None
                on_demand = self._image_mode == "on_demand"
                latest_complete = None
                while True:
                    try:
                        payload = self._sub.recv(self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
                    got_message = True
                    if on_demand:
                        try:
                            observation = unpack_observation(payload)
                        except Exception:
                            observation = None
                        if observation is not None and self._has_complete_images(observation):
                            latest_complete = payload
                if not got_message or payload is None:
                    return None
                if on_demand and latest_complete is not None:
                    payload = latest_complete
            self._freshness.mark()
            return payload

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        """Capture one raw collection payload and expose it as timestamped streams.

        Returns None when the backlog is empty.
        """
        payload = self._drain_raw_collection()
        if payload is None:
            return None
        timestamp = _observation_timestamp(payload)
        journal = getattr(self, "_collection_journal", None)
        if journal is None:
            journal = _WireCaptureJournal(getattr(self, "_collection_journal_dir", None))
            self._collection_journal = journal
        entry = journal.append(payload)

        def decode_raw(entry: _WireJournalEntry = entry) -> CollectionRawBatch:
            wire_obs = unpack_observation(entry.read())
            batch = CollectionRawBatch()
            for key, image in wire_obs.images.items():
                batch.images.setdefault(key, []).append(
                    CollectionRawSample(timestamp=wire_obs.t, value=np.asarray(image))
                )
            for group_name, state in wire_obs.state.items():
                batch.vectors.setdefault(f"state_qpos:{group_name}", []).append(
                    CollectionRawSample(
                        timestamp=wire_obs.t,
                        value=np.asarray(state, dtype=np.float32),
                    )
                )
            if wire_obs.action is not None:
                batch.vectors.setdefault("action_qpos", []).append(
                    CollectionRawSample(
                        timestamp=wire_obs.t,
                        value=np.asarray(wire_obs.action, dtype=np.float32),
                    )
                )
            if wire_obs.eef is not None:
                for group_name, eef in wire_obs.eef.items():
                    batch.vectors.setdefault(f"state_eef:{group_name}", []).append(
                        CollectionRawSample(
                            timestamp=wire_obs.t,
                            value=np.asarray(eef, dtype=np.float32),
                        )
                    )
            if wire_obs.action_eef is not None:
                batch.vectors.setdefault("action_eef", []).append(
                    CollectionRawSample(
                        timestamp=wire_obs.t,
                        value=np.asarray(wire_obs.action_eef, dtype=np.float32),
                    )
                )
            return batch

        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=decode_raw)

    def prepare_collection_capture(self, directory: str | None) -> None:
        """Place the next journal beside the active recorder's dataset."""
        if not directory:
            return
        with self._lock:
            self._collection_journal_dir = Path(directory)

    def finish_collection_capture(self) -> None:
        """Detach the active journal; queued snapshots retain it until save completes."""
        with self._lock:
            journal = self._collection_journal
            self._collection_journal = None
            if journal is not None:
                journal.finish()

    def close(self) -> None:
        """Close this reader's SUB socket (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._collection_journal is not None:
            self._collection_journal.finish()
        self._collection_journal = None
        self._sub.close(linger=0)


class ZmqTransport(TransportBridge):
    """ZeroMQ transport bridge.

    SUB socket receives WireObservation snapshots from the execution layer; PUB
    socket sends WireAction commands. get_frame() returns the freshest snapshot
    converted into the robot's canonical Observation (images keyed by
    observation_key, state concatenated in state_composition order).

    The SUB side lives in an _ObservationReader. The control loop reads through the
    transport's own reader; auxiliary consumers (e.g. the console visualization,
    polled from HTTP worker threads) get an independent reader with its own socket
    via create_observation_reader(), so they never contend for the same socket or
    steal each other's frames — pub/sub fans a full copy out to each subscriber.
    """

    def __init__(self, config: ConfigDict, robot: Robot) -> None:
        import zmq

        self._config = config
        self._robot = robot
        self._closed = False
        self._zmq = zmq

        self._reader = _ObservationReader(config, robot, zmq)
        self._collection_reader = _ObservationReader(
            config,
            robot,
            zmq,
            preserve_collection_backlog=True,
        )
        self._qpos_reader = _ObservationReader(config, robot, zmq)
        self._extra_readers: list[_ObservationReader] = []

        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(config.transport.pub_endpoint)

        logger.info(
            "ZMQ transport ready: sub=%s pub=%s robot=%s",
            config.transport.sub_endpoint,
            config.transport.pub_endpoint,
            robot.name,
        )

    def get_frame(self) -> Observation | None:
        """Freshest observation from the control-loop reader (see _ObservationReader)."""
        return self._reader.get_frame()

    def supports_collection(self) -> bool:
        """Always True — the ZMQ transport always exposes a collection reader."""
        return True

    def get_collection_frame(self) -> Observation | None:
        """Next collection frame from the dedicated collection reader."""
        return self._collection_reader.get_collection_frame()

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        """Deferred-decode collection snapshot from the dedicated collection reader."""
        return self._collection_reader.acquire_collection_raw()

    def prepare_collection_capture(self, directory: str | None) -> None:
        """Place the next raw wire journal on the active recorder's filesystem."""
        self._collection_reader.prepare_collection_capture(directory)

    def finish_collection_capture(self) -> None:
        """Release the reader's reference to the completed raw wire journal."""
        self._collection_reader.finish_collection_capture()

    def get_latest_qpos(self) -> np.ndarray | None:
        """Latest joint state [qpos_dim] float32 from the dedicated qpos reader."""
        return self._qpos_reader.get_latest_qpos()

    def hil_status(self) -> HilStatus:
        """Return the latest HIL capability reported by the execution node."""
        return self._qpos_reader.hil_status()

    def poll_operator_event(self) -> str | None:
        """Poll the dedicated qpos/status reader for a hardware button edge."""
        return self._qpos_reader.poll_operator_event()

    def _send_hil_control(self, target: str, mode: str | None = None) -> None:
        action = np.zeros(self._robot.total_action_dim, dtype=np.float32)
        for attempt in range(COLLECTION_CONTROL_REPEATS):
            self._pub.send(
                pack_action(
                    WireAction(
                        t=time.monotonic(),
                        action=action,
                        target=target,
                        mode=mode,
                    )
                )
            )
            if attempt + 1 < COLLECTION_CONTROL_REPEATS:
                time.sleep(COLLECTION_CONTROL_INTERVAL_S)

    def _wait_hil_state(self, active: bool) -> HilStatus:
        deadline = time.monotonic() + HIL_ACK_TIMEOUT_S
        last = self.hil_status()
        while time.monotonic() < deadline:
            last = self.hil_status()
            if last.error or (last.supported and last.active is active):
                return last
            time.sleep(0.01)
        return HilStatus(
            supported=last.supported,
            active=last.active,
            error=f"HIL {'start' if active else 'stop'} acknowledgement timed out",
        )

    def start_hil_control(self, mode: str) -> HilStatus:
        status = self.hil_status()
        if not status.supported:
            return status
        self._send_hil_control(HIL_START_TARGET, mode)
        return self._wait_hil_state(True)

    def stop_hil_control(self) -> HilStatus:
        status = self.hil_status()
        if not status.supported:
            return status
        self._send_hil_control(HIL_STOP_TARGET)
        return self._wait_hil_state(False)

    def get_hil_frame(self) -> Observation | None:
        """Return the next action-bearing observation produced during HIL."""
        return self._collection_reader.get_collection_frame()

    def seconds_since_last_recv(self) -> float | None:
        """Freshest receipt age across all readers, or None if none have received."""
        ages = [
            reader.seconds_since_last_recv()
            for reader in (
                self._reader,
                self._collection_reader,
                self._qpos_reader,
                *self._extra_readers,
            )
        ]
        known = [age for age in ages if age is not None]
        return min(known) if known else None

    def image_min_hz(self) -> float | None:
        """Minimum known image receive rate across all active readers."""
        rates = [
            reader.image_min_hz()
            for reader in (
                self._reader,
                self._collection_reader,
                self._qpos_reader,
                *self._extra_readers,
            )
        ]
        known = [rate for rate in rates if rate is not None]
        return min(known) if known else None

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        """Send the action to the execution layer as a timestamped WireAction.

        Args:
            action: [action_dim] float32 flat action vector.
            target: "real" or "sim" routing tag carried on the wire.
        """
        message = WireAction(t=time.monotonic(), action=np.asarray(action), target=target)
        self._pub.send(pack_action(message))

    def _publish_image_request(self) -> None:
        action = np.zeros(self._robot.total_action_dim, dtype=np.float32)
        self._pub.send(
            pack_action(
                WireAction(
                    t=time.monotonic(),
                    action=action,
                    target=IMAGE_REQUEST_TARGET,
                )
            )
        )

    def get_policy_frame(self) -> Observation | None:
        """Return a policy frame, requesting a fresh image in on-demand mode."""
        mode = str(getattr(self._config.transport, "image_mode", "stream"))
        if mode == "stream":
            return self._reader.get_frame()
        if mode != "on_demand":
            raise ValueError(f"unsupported transport.image_mode: {mode!r}")

        revision = self._reader.image_revision()
        self._publish_image_request()
        deadline = time.monotonic() + float(
            getattr(self._config.transport, "image_request_timeout_s", 2.0)
        )
        while not self.is_shutdown() and time.monotonic() < deadline:
            frame = self._reader.get_frame_after_image_revision(revision)
            if frame is not None:
                return frame
            time.sleep(0.001)
        logger.warning("Timed out waiting for on-demand image observation")
        return None

    def _send_collection_control(self, target: str, mode: str | None = None) -> None:
        action = np.zeros(self._robot.total_action_dim, dtype=np.float32)
        for attempt in range(COLLECTION_CONTROL_REPEATS):
            self._pub.send(
                pack_action(
                    WireAction(
                        t=time.monotonic(),
                        action=action,
                        target=target,
                        mode=mode,
                    )
                )
            )
            if attempt + 1 < COLLECTION_CONTROL_REPEATS:
                time.sleep(COLLECTION_CONTROL_INTERVAL_S)

    def start_collection(self) -> None:
        """Tell the execution layer to start recording (sends repeated start signals)."""
        teleop = (self._config.get("collection") or {}).get("teleop") or {}
        control_source = str(teleop.get("control_source", "transport"))
        self._send_collection_control(COLLECTION_START_TARGET, mode=control_source)

    def start_policy_collection(self) -> None:
        """Start a policy-driven rollout without connecting the teleop source."""
        self._send_collection_control(COLLECTION_START_TARGET, mode="client")

    def clear_collection_backlog(self) -> float | None:
        """Drop collection frames buffered before the active recording episode."""
        return self._collection_reader.clear_collection_backlog()

    def stop_collection(self) -> None:
        """Tell the execution layer to stop recording (sends repeated stop signals)."""
        self._send_collection_control(COLLECTION_STOP_TARGET)

    def create_observation_reader(self) -> _ObservationReader:
        """Create and track an independent reader (own SUB socket) for an aux consumer."""
        reader = _ObservationReader(self._config, self._robot, self._zmq)
        self._extra_readers.append(reader)
        return reader

    def close(self) -> None:
        """Close every reader and the PUB socket (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._reader.close()
        self._collection_reader.close()
        self._qpos_reader.close()
        for reader in self._extra_readers:
            reader.close()
        self._pub.close(linger=0)

    def is_shutdown(self) -> bool:
        """True once close() has been called."""
        return self._closed


@TRANSPORT_REGISTRY.register("zmq")
def _build_zmq(config: ConfigDict, robot: Robot) -> TransportBridge:
    return ZmqTransport(config, robot)
