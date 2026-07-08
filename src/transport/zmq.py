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
import threading
import time

import numpy as np
from openpi_client import msgpack_numpy

from core.config import ConfigDict
from core.registry import TRANSPORT_REGISTRY
from core.types import CollectionRawBatch, CollectionRawSample, Observation, RawCollectionSnapshot
from robots.base import Robot
from transport.base import TransportBridge
from transport.utils import ImageRateTracker, StreamFreshness

logger = logging.getLogger(__name__)

_PACKER = msgpack_numpy.Packer()
COLLECTION_START_TARGET = "collect_start"
COLLECTION_STOP_TARGET = "collect_stop"
COLLECTION_CONTROL_REPEATS = 5
COLLECTION_CONTROL_INTERVAL_S = 0.02
COLLECTION_SOCKET_DRAIN_MAX = 64


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


@dataclasses.dataclass
class WireAction:
    """One action command sent from the client to an execution-layer node.

    Args:
        t: client-side send timestamp in seconds.
        action: flat action vector (joint space, full robot DOF).
        target: "real", "sim", "collect_start", or "collect_stop".
    """

    t: float
    action: np.ndarray
    target: str = "real"


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
    )


def pack_action(action: WireAction) -> bytes:
    """Serialize a WireAction (t, float32 action vector, target) to msgpack bytes."""
    return _PACKER.pack(
        {
            "t": float(action.t),
            "action": np.asarray(action.action, dtype=np.float32),
            "target": action.target,
        }
    )


def unpack_action(payload: bytes) -> WireAction:
    """Decode msgpack bytes into a WireAction (target defaults to "real")."""
    raw = msgpack_numpy.unpackb(payload)
    return WireAction(
        t=float(raw["t"]),
        action=np.asarray(raw["action"], dtype=np.float32),
        target=str(raw.get("target", "real")),
    )


# --- transport -----------------------------------------------------------------


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
        self._latest: WireObservation | None = None
        self._latest_images: dict[str, np.ndarray] = {}
        self._preserve_collection_backlog = preserve_collection_backlog
        self._collection_queue: collections.deque[WireObservation] = collections.deque()
        self._raw_collection_queue: collections.deque[bytes] = collections.deque()
        self._freshness = StreamFreshness()
        self._image_rate = ImageRateTracker()
        self._lock = threading.Lock()
        self._closed = False

        self._disabled_cameras = set(config.transport.disabled_cameras)
        self._disabled_groups = set(config.transport.disabled_groups)
        # Per-group slice of the robot's initial qpos, used to fill state for
        # groups disabled in this deployment (e.g. an unused arm).
        self._group_initial_qpos: dict[str, np.ndarray] = {}
        offset = 0
        for group in robot.actuator_groups:
            self._group_initial_qpos[group.name] = np.asarray(
                robot.initial_qpos[offset : offset + group.dof], dtype=np.float32
            )
            offset += group.dof

        ctx = zmq_mod.Context.instance()
        self._sub = ctx.socket(zmq_mod.SUB)
        self._sub.connect(config.transport.sub_endpoint)
        self._sub.setsockopt(zmq_mod.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq_mod.RCVTIMEO, 0)  # non-blocking drain

    def _drain_latest(self) -> WireObservation | None:
        """Pop all queued SUB messages, keeping only the newest by timestamp."""
        with self._lock:
            newest = self._latest
            got_message = False
            while True:
                try:
                    payload = self._sub.recv(self._zmq.NOBLOCK)
                except self._zmq.Again:
                    break
                got_message = True
                obs = unpack_observation(payload)
                self._image_rate.mark_many(obs.images.keys(), obs.t)
                self._cache_images_locked(obs)
                if newest is None or obs.t >= newest.t:
                    newest = obs
            if got_message:
                self._freshness.mark()
            self._latest = newest
            return newest

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

    def clear_collection_backlog(self) -> float | None:
        """Drain the socket and drop queued collection frames captured pre-recording."""
        with self._lock:
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
                    payload_time = unpack_observation(last_payload).t
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

        images: dict[str, np.ndarray] = {}
        for camera in self._robot.observation_schema.cameras:
            if camera.observation_key in self._disabled_cameras:
                continue
            image = wire_obs.images.get(camera.observation_key)
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
        if self._preserve_collection_backlog:
            wire_obs = self._drain_collection_queue()
        else:
            wire_obs = self._drain_latest()
        if wire_obs is None:
            return None
        return self._wire_to_observation(wire_obs)

    def _drain_raw_collection(self) -> bytes | None:
        """Pop all queued SUB messages as raw bytes, return the oldest unconsumed.

        The main thread does zero msgpack work here: it only moves bytes off the
        socket into a backlog and hands back one payload. Decoding happens later
        through the snapshot's raw-batch closure on the ingest thread.
        """
        with self._lock:
            got_message = False
            for _ in range(COLLECTION_SOCKET_DRAIN_MAX):
                try:
                    payload = self._sub.recv(self._zmq.NOBLOCK)
                except self._zmq.Again:
                    break
                got_message = True
                self._raw_collection_queue.append(payload)
            if got_message:
                self._freshness.mark()
            if not self._raw_collection_queue:
                return None
            return self._raw_collection_queue.popleft()

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        """Capture one raw collection payload and expose it as timestamped streams.

        Returns None when the backlog is empty.
        """
        payload = self._drain_raw_collection()
        if payload is None:
            return None
        wire_obs = unpack_observation(payload)

        def decode_raw(wire_obs: WireObservation = wire_obs) -> CollectionRawBatch:
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

        return RawCollectionSnapshot(timestamp=wire_obs.t, decode_raw=decode_raw)

    def close(self) -> None:
        """Close this reader's SUB socket (idempotent)."""
        if self._closed:
            return
        self._closed = True
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
        self._collection_reader = _ObservationReader(config, robot, zmq)
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

    def get_latest_qpos(self) -> np.ndarray | None:
        """Latest joint state [qpos_dim] float32 from the dedicated qpos reader."""
        return self._qpos_reader.get_latest_qpos()

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

    def _send_collection_control(self, target: str) -> None:
        action = np.zeros(self._robot.total_action_dim, dtype=np.float32)
        for attempt in range(COLLECTION_CONTROL_REPEATS):
            self._pub.send(
                pack_action(WireAction(t=time.monotonic(), action=action, target=target))
            )
            if attempt + 1 < COLLECTION_CONTROL_REPEATS:
                time.sleep(COLLECTION_CONTROL_INTERVAL_S)

    def start_collection(self) -> None:
        """Tell the execution layer to start recording (sends repeated start signals)."""
        self._send_collection_control(COLLECTION_START_TARGET)

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
