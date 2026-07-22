"""Abstract transport bridge — defines the TransportBridge ABC for robot
communication (get_frame, publish_action, qpos feedback) with a built-in
sleep-based rate limiter for non-ROS environments.
"""

from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import importlib.util
import logging
import threading
from typing import Any, Protocol, cast

import numpy as np

from core.config import ConfigDict
from core.types import (
    CollectionRawBatch,
    CollectionRawImage,
    CollectionRawSample,
    Observation,
    RawCollectionSnapshot,
)
from robots.base import Robot
from transport.utils import ImageRateTracker, StreamFreshness, _SimpleRate

logger = logging.getLogger(__name__)

_COLLECTION_DEQUE_MAX = 2000


@dataclasses.dataclass(frozen=True)
class HilStatus:
    """Current hardware-in-the-loop takeover capability and state."""

    supported: bool
    active: bool = False
    error: str = ""


class ObservationSource(Protocol):
    """Read-only observation access used by auxiliary consumers (e.g. the console
    visualization). A transport may hand out a dedicated source so those consumers
    don't contend with the control loop for the same underlying socket.
    """

    def get_frame(self) -> Observation | None:
        """Return the latest observation (cameras + state), or None if none yet."""
        ...

    def get_latest_qpos(self) -> np.ndarray | None:
        """Return the most recent joint positions [qpos_dim] float32, or None."""
        ...

    def seconds_since_last_recv(self) -> float | None:
        """Return seconds since the last received message, or None if never received."""
        ...

    def image_min_hz(self) -> float | None:
        """Return the minimum image receive rate across cameras, or None if unknown."""
        ...


class TransportBridge(abc.ABC):
    """Abstract transport layer for robot communication.

    Defines the interface between EVA Client and the physical (or simulated) robot:
    get_frame() captures a synchronized observation (images + joint state),
    publish_action() sends a joint command, and get_latest_qpos() reads the most
    recent joint position without building a full observation.

    Concrete implementations: Ros1Transport (real robot via rospy), DatasetTransport
    (LeRobot v2 episode replay), and OfflineTransport (no-op stand-in when ROS is
    absent). The create_rate() helper returns a sleep-based rate limiter that
    Ros1Transport overrides with rospy.Rate for tighter timing.
    """

    @abc.abstractmethod
    def get_frame(self) -> Observation | None:
        """Capture one synchronized observation, or None if none is available.

        Returns:
            Observation bundling per-camera images (each [H, W, 3] uint8, keyed by
            observation_key) and the concatenated joint state vector [state_dim]
            float32, or None when the source has no current frame (link offline,
            episode exhausted, transport shut down).
        """
        ...

    @abc.abstractmethod
    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        """Send one joint-space action command to the robot.

        Args:
            action: [action_dim] float32 flat action vector (full robot DOF).
            target: destination — "real" for the physical robot, "sim" for a
                simulation visualizer when the backend has sim publishers.
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release sockets / subscribers / file handles and stop the transport."""
        ...

    def has_sim_publishers(self) -> bool:
        """True if the backend can also publish actions to a simulation target."""
        return False

    def is_offline(self) -> bool:
        """True if this transport has no live robot link (replay/synthetic/degraded).

        Drives whether the control loop skips robot-facing steps (reset trajectory,
        policy server contact). Live transports (zmq/ros connected) return False;
        dataset/debug/offline stand-ins return True.
        """
        return False

    def get_latest_qpos(self) -> np.ndarray | None:
        """Most recent joint position [qpos_dim] float32, or None if unavailable.

        A lightweight read of joint feedback that skips building a full observation
        (no image decode). Default returns None for sources with no joint feedback.
        """
        return None

    def seconds_since_last_recv(self) -> float | None:
        """Wall time since the last inbound message, or None if not tracked.

        Live transports (zmq/ros) override this to report link health; replay and
        synthetic sources leave it None since they have no notion of going offline.
        """
        return None

    def image_min_hz(self) -> float | None:
        """Minimum image receive rate across cameras, or None until enough samples."""
        return None

    def supports_collection(self) -> bool:
        """True if this transport can record data-collection frames."""
        return False

    def start_collection(self) -> None:
        """Signal the source to begin emitting collection frames (no-op default)."""
        return None

    def clear_collection_backlog(self) -> float | None:
        """Discard collection frames captured before the active recording episode.

        Returns:
            Source-clock timestamp at or before which future collection snapshots
            should be ignored, or None when the transport cannot provide one.
        """
        return None

    def get_collection_frame(self) -> Observation | None:
        """Next data-collection frame (synchronized sensors + teleop action), or None.

        Returns an Observation carrying the recorder's column set (images,
        state_qpos/state_eef, action_qpos/action_eef) for the active episode, or
        None when no new frame is ready. Default None for non-collecting transports.
        """
        return None

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        """O(1) capture of one collection frame's raw source refs + timestamp.

        The capture loop calls this every tick: it must only grab source
        references (undecoded bytes / pinned messages) and stamp the source
        capture time. The returned snapshot yields timestamped raw stream samples
        for episode-level fixed-clock alignment at save time. Returns None when
        no new source frame is available.
        """
        return None

    def collection_diagnostics(self) -> str:
        """Human-readable reason collection frames are not currently available."""
        return ""

    def stop_collection(self) -> None:
        """Signal the source to stop emitting collection frames (no-op default)."""
        return None

    def reset_hil_control(self) -> None:
        """Reset backend HIL relay state before a new intervention/collection."""
        return None

    def set_hil_control_mode(self, mode: str) -> None:
        """Set backend HIL relay mode when supported."""
        _ = mode
        return None

    def set_hil_relay_enabled(self, enabled: bool) -> None:
        """Enable or disable backend HIL input relay when supported."""
        _ = enabled
        return None

    def hil_status(self) -> HilStatus:
        """Return the backend's current HIL capability and activation state."""
        return HilStatus(supported=False, error="Transport does not support HIL")

    def start_hil_control(self, mode: str) -> HilStatus:
        """Request HIL takeover and return the confirmed backend state."""
        _ = mode
        return self.hil_status()

    def stop_hil_control(self) -> HilStatus:
        """Stop HIL takeover and return the confirmed backend state."""
        return self.hil_status()

    def get_hil_frame(self) -> Observation | None:
        """Return the next state/action frame produced during HIL takeover."""
        return None

    def record_collection_gripper_action(
        self, group_name: str, value: float, stamp: Any | None = None
    ) -> None:
        """Record a collection-only gripper action target when supported."""
        _ = group_name, value, stamp
        return None

    def create_observation_reader(self) -> ObservationSource:
        """Return an observation source for an auxiliary consumer.

        Default: the transport itself. Transports whose read path isn't safe to
        share across threads (e.g. ZMQ's single SUB socket) override this to hand
        back an independent reader so visualization never steals the control
        loop's frames.
        """
        return self

    def create_rate(self, hz: float) -> _SimpleRate:
        """Return a rate limiter with a .sleep() method."""
        return _SimpleRate(hz)

    def is_shutdown(self) -> bool:
        """Return True if the transport runtime is shutting down."""
        return False


@dataclasses.dataclass(frozen=True)
class GroupTopics:
    """ROS topic names for one actuator group (shared by the ros1/ros2 backends).

    Every group needs a ``state_topic`` (JointState subscription for current
    position) and a ``command_topic`` (JointState publication for actions). The
    remaining fields are backend-specific and stay None where unused:
    ``sim_command_topic`` (ros1 sim visualizer), ``eef_state_topic`` (PoseStamped
    EEF feedback, both backends), ``gripper_state_topic`` / ``gripper_command_topic``
    (ros2 split-gripper topics), and ``hil_input_topic`` (operator joint input
    relayed only while HIL takeover is active).
    """

    state_topic: str
    command_topic: str
    sim_command_topic: str | None = None
    eef_state_topic: str | None = None
    gripper_state_topic: str | None = None
    gripper_command_topic: str | None = None
    hil_input_topic: str | None = None


def resolve_topics(topics_cfg: Any) -> tuple[dict[str, str], dict[str, GroupTopics]]:
    """Resolve ``config.transport.topics`` into camera and per-group topic maps.

    Args:
        topics_cfg: dict-like with ``camera_topics`` (name -> topic) and
            ``group_topics`` (group name -> GroupTopics field dict).

    Returns:
        camera_topics: camera logical name -> image topic.
        group_topics: actuator group name -> GroupTopics.
    """
    if not topics_cfg:
        raise ValueError("transport.topics is required for the ros1/ros2 backends")
    camera_topics = dict(topics_cfg["camera_topics"])
    group_topics = {
        name: GroupTopics(**fields) for name, fields in topics_cfg["group_topics"].items()
    }
    return camera_topics, group_topics


# Always older than any online window, so the console reports the link as offline.
_OFFLINE_AGE_SECONDS = 1e9


class OfflineTransport(TransportBridge):
    """No-op transport used when the configured live backend is unavailable.

    Stands in when a live backend (ROS) isn't installed, so the web console boots
    on a dev box and the UI can be debugged against a config that targets a real
    robot. Produces no observations and silently drops published actions. Reports a
    permanently stale receipt age so the console shows the transport offline.
    """

    def __init__(self) -> None:
        self._shutdown = threading.Event()

    def get_frame(self) -> Observation | None:
        """Always None — this transport carries no observations."""
        return None

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        """Drop the action; there is no robot to receive it."""
        return None

    def is_offline(self) -> bool:
        """Always True — the offline transport has no live link."""
        return True

    def seconds_since_last_recv(self) -> float | None:
        """Always a huge stale age so the console reports the link offline."""
        return _OFFLINE_AGE_SECONDS

    def close(self) -> None:
        """Mark the transport shut down."""
        self._shutdown.set()

    def is_shutdown(self) -> bool:
        """Return True once close() has been called."""
        return self._shutdown.is_set()


# Core import each ROS backend needs; absent on a non-ROS dev box.
_ROS_BACKEND_MODULE = {"ros1": "rospy", "ros2": "rclpy"}


def build_transport(config: ConfigDict, robot: Robot) -> TransportBridge:
    """Build the configured transport, degrading to OfflineTransport when ROS is absent.

    Args:
        config: full client config; ``config.transport.type`` selects the backend.
        robot: the robot the transport communicates with.

    Returns:
        The constructed transport, or an OfflineTransport when a ROS backend is
        selected but its runtime is not installed (dev box without ROS).
    """
    from core.registry import TRANSPORT_REGISTRY

    backend_module = _ROS_BACKEND_MODULE.get(config.transport.type)
    if backend_module is not None and importlib.util.find_spec(backend_module) is None:
        # ROS isn't installed (dev box). Degrade before constructing the ROS transport
        # so the web console still boots for UI/debug, and config validation that the
        # ROS backend would do (e.g. topics) doesn't mask the real "no ROS" cause.
        logger.warning(
            "transport %s unavailable (no module named '%s'); starting offline transport",
            config.transport.type,
            backend_module,
        )
        return OfflineTransport()
    try:
        return TRANSPORT_REGISTRY.build(config.transport.type, config, robot)
    except ImportError as error:
        # A ROS submodule (e.g. cv_bridge) is missing even though the core module
        # imported; still degrade so the console boots on a partial install.
        logger.warning(
            "transport %s unavailable (%s); starting offline transport",
            config.transport.type,
            error,
        )
        return OfflineTransport()


class _RosTransportBase(TransportBridge):
    """Deque-sync data-collection mechanism shared by the ros1/ros2 transports.

    Subclasses construct the per-group deques and set the attributes this base
    reads (``_config``, ``_robot``, ``_bridge``, ``_freshness``, the collection
    deque dicts, ``_camera_deques``, ``_last_acquire_stamp``) and implement the
    four backend hooks: ``_stamp_to_sec``, ``_decode_image_msg``, ``_eef_from_msg``,
    and ``_collection_cfg``.
    """

    _config: ConfigDict
    _robot: Robot
    _bridge: Any
    _freshness: StreamFreshness
    _image_rate: ImageRateTracker
    _camera_deques: dict[str, collections.deque]
    _collection_camera_deques: dict[str, collections.deque]
    _collection_qpos_deques: dict[str, collections.deque]
    _collection_eef_deques: dict[str, collections.deque]
    _collection_action_qpos_deques: dict[str, collections.deque]
    _collection_action_eef_deques: dict[str, collections.deque]
    _last_acquire_stamp: float | None

    @abc.abstractmethod
    def _stamp_to_sec(self, msg: Any) -> float:
        """Header stamp of ``msg`` as seconds (backend-specific encoding)."""
        ...

    @abc.abstractmethod
    def _decode_image_msg(self, camera_name: str, msg: Any) -> np.ndarray | None:
        """Decode a pinned image message to a BGR uint8 array, or None on failure."""
        ...

    @abc.abstractmethod
    def _eef_from_msg(self, group: Any, msg: Any, frame_time: float) -> np.ndarray:
        """Build one group's EEF state vector from a pose msg + the group's gripper."""
        ...

    @property
    @abc.abstractmethod
    def _collection_cfg(self) -> Any:
        """The backend's ``config.collection.transport.ros{1,2}`` section."""
        ...

    def _deque_guard(self) -> Any:
        return getattr(self, "_deque_lock", contextlib.nullcontext())

    def _append_msg(self, deque: collections.deque, msg: Any) -> None:
        with self._deque_guard():
            if len(deque) >= _COLLECTION_DEQUE_MAX:
                deque.popleft()
            deque.append(msg)
        self._freshness.mark()

    def _append_camera_msg(
        self,
        camera_name: str,
        deque: collections.deque,
        msg: Any,
        timestamp: float | None = None,
    ) -> None:
        self._image_rate.mark(camera_name, timestamp)
        self._append_msg(deque, msg)

    def image_min_hz(self) -> float | None:
        """Minimum camera topic receive rate across subscribed cameras."""
        return self._image_rate.min_hz(self._camera_deques.keys())

    def seconds_since_last_recv(self) -> float | None:
        """Seconds since the last message landed on any subscription (link health)."""
        return self._freshness.seconds_since()

    def supports_collection(self) -> bool:
        """True when this config carries a collection schema (vs. a plain deploy/eval config)."""
        return bool(self._config.collection.schema.columns)

    def clear_collection_backlog(self) -> float | None:
        """Advance the collection cursor past the latest cached synchronized frame."""
        frame_time = self._collection_frame_time()
        if frame_time is not None:
            self._last_acquire_stamp = frame_time
        self._reset_collection_raw_cursors()
        return frame_time

    def _latest_before_or_at(self, deque: collections.deque, frame_time: float) -> Any | None:
        while len(deque) > 1 and self._stamp_to_sec(deque[1]) <= frame_time:
            deque.popleft()
        if not deque:
            return None
        if self._stamp_to_sec(deque[0]) > frame_time:
            return None
        return deque[0]

    def _collection_latest_msg(self, deque: collections.deque, frame_time: float) -> Any | None:
        msg = self._latest_before_or_at(deque, frame_time)
        if msg is None:
            return None
        max_skew = self._collection_cfg.max_frame_skew_sec
        if max_skew > 0.0 and frame_time - self._stamp_to_sec(msg) > max_skew:
            return None
        return msg

    def _collection_joint_vector(
        self,
        deques_by_group: dict[str, collections.deque],
        frame_time: float,
        value_attr: str = "position",
    ) -> np.ndarray | None:
        if not deques_by_group:
            return None
        parts: list[np.ndarray] = []
        for group in self._robot.actuator_groups:
            deque = deques_by_group.get(group.name)
            if deque is None:
                return None
            msg = self._collection_latest_msg(deque, frame_time)
            if msg is None:
                return None
            parts.append(np.asarray(getattr(msg, value_attr), dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def _collection_eef_vector(
        self,
        deques_by_group: dict[str, collections.deque],
        frame_time: float,
    ) -> np.ndarray | None:
        if not deques_by_group:
            return None
        parts: list[np.ndarray] = []
        for group in self._robot.actuator_groups:
            deque = deques_by_group.get(group.name)
            if deque is None:
                return None
            msg = self._collection_latest_msg(deque, frame_time)
            if msg is None:
                return None
            parts.append(self._eef_from_msg(group, msg, frame_time))
        return np.concatenate(parts, axis=0)

    def _collection_camera_specs(self) -> list[Any]:
        cameras = self._config.collection.schema.cameras
        return [
            camera
            for camera in self._robot.observation_schema.cameras
            if camera.observation_key in cameras or camera.name in cameras
        ]

    def _collection_capture_camera_deques(self) -> dict[str, collections.deque]:
        return getattr(self, "_collection_camera_deques", self._camera_deques)

    def _collection_images(self, frame_time: float) -> dict[str, np.ndarray] | None:
        images: dict[str, np.ndarray] = {}
        camera_deques = self._collection_capture_camera_deques()
        for camera in self._collection_camera_specs():
            deque = camera_deques.get(camera.name)
            if deque is None:
                return None
            msg = self._collection_latest_msg(deque, frame_time)
            if msg is None:
                return None
            image = self._decode_image_msg(camera.name, msg)
            if image is None:
                return None
            images[camera.observation_key] = image
        return images

    def _collection_frame_time(self) -> float | None:
        primary_camera = self._collection_cfg.primary_camera
        camera_deques = self._collection_capture_camera_deques()
        if primary_camera:
            for camera in self._robot.observation_schema.cameras:
                if primary_camera not in (camera.observation_key, camera.name):
                    continue
                deque = camera_deques.get(camera.name)
                if deque is None or len(deque) == 0:
                    return None
                return self._stamp_to_sec(deque[-1])
            return None

        required: list[collections.deque] = []
        for camera in self._collection_camera_specs():
            deque = camera_deques.get(camera.name)
            if deque is None:
                return None
            required.append(deque)

        columns = self._config.collection.schema.columns
        if "qpos" in columns:
            required.extend(self._collection_qpos_deques.values())
        if "eef" in columns:
            required.extend(self._collection_eef_deques.values())
        if "action_qpos" in columns:
            required.extend(self._collection_action_qpos_deques.values())
        if "action_eef" in columns:
            required.extend(self._collection_action_eef_deques.values())

        if not required or any(len(deque) == 0 for deque in required):
            return None
        return min(self._stamp_to_sec(deque[-1]) for deque in required)

    def _gather_collection_columns(
        self, frame_time: float
    ) -> tuple[np.ndarray | None, ...] | None:
        """Gather the configured collection column vectors at ``frame_time``.

        Returns:
            (state_qpos, state_eef, action_qpos, action_eef), each present only for
            its configured column (else None), or None if any required column is
            missing at ``frame_time``.
        """
        columns = self._config.collection.schema.columns
        state_qpos = state_eef = action_qpos = action_eef = None
        if "qpos" in columns:
            state_qpos = self._collection_joint_vector(self._collection_qpos_deques, frame_time)
            if state_qpos is None:
                return None
        if "eef" in columns:
            state_eef = self._collection_eef_vector(self._collection_eef_deques, frame_time)
            if state_eef is None:
                return None
        if "action_qpos" in columns:
            action_qpos = self._collection_joint_vector(
                self._collection_action_qpos_deques, frame_time
            )
            if action_qpos is None:
                return None
        if "action_eef" in columns:
            action_eef = self._collection_eef_vector(self._collection_action_eef_deques, frame_time)
            if action_eef is None:
                return None
        return state_qpos, state_eef, action_qpos, action_eef

    def get_collection_frame(self) -> Observation | None:
        """Build one synchronized data-collection frame from the collection deques.

        Aligns the configured columns (state_qpos/state_eef and their action_*
        variants) plus images to a common frame time and decodes the images inline.
        Returns None if collection is disabled or any required column/camera is
        missing at that frame time.

        Returns:
            Observation with images [H, W, 3] and the configured float32 vectors,
            or None.
        """
        if not self._config.collection.schema.columns:
            return None
        frame_time = self._collection_frame_time()
        if frame_time is None:
            return None
        images = self._collection_images(frame_time)
        if images is None:
            return None
        gathered = self._gather_collection_columns(frame_time)
        if gathered is None:
            return None
        state_qpos, state_eef, action_qpos, action_eef = gathered
        return Observation(
            images=images,
            state_qpos=cast(np.ndarray, state_qpos),
            state_eef=state_eef,
            action_qpos=action_qpos,
            action_eef=action_eef,
            timestamp=frame_time,
        )

    def _collection_raw_cursors(self) -> dict[str, float]:
        cursors = getattr(self, "_collection_raw_cursor_state", None)
        if cursors is None:
            cursors = {}
            self._collection_raw_cursor_state = cursors
        return cursors

    def _collection_raw_streams(self) -> list[tuple[str, str, str, Any, collections.deque]]:
        streams: list[tuple[str, str, str, Any, collections.deque]] = []
        camera_deques = self._collection_capture_camera_deques()
        for camera in self._collection_camera_specs():
            deque = camera_deques.get(camera.name)
            if deque is not None:
                streams.append(("image", camera.observation_key, camera.name, camera, deque))

        columns = self._config.collection.schema.columns
        if "qpos" in columns:
            for group in self._robot.actuator_groups:
                deque = self._collection_qpos_deques.get(group.name)
                if deque is not None:
                    streams.append(("vector", "state_qpos", group.name, group, deque))
        if "eef" in columns:
            for group in self._robot.actuator_groups:
                deque = self._collection_eef_deques.get(group.name)
                if deque is not None:
                    streams.append(("vector", "state_eef", group.name, group, deque))
        if "action_qpos" in columns:
            for group in self._robot.actuator_groups:
                deque = self._collection_action_qpos_deques.get(group.name)
                if deque is not None:
                    streams.append(("vector", "action_qpos", group.name, group, deque))
        if "action_eef" in columns:
            for group in self._robot.actuator_groups:
                deque = self._collection_action_eef_deques.get(group.name)
                if deque is not None:
                    streams.append(("vector", "action_eef", group.name, group, deque))
        return streams

    def _collection_raw_context_streams(self) -> list[tuple[str, collections.deque]]:
        return []

    def _reset_collection_raw_cursors(self) -> None:
        cursors = self._collection_raw_cursors()
        cursors.clear()
        for kind, field, source_name, _source, deque in self._collection_raw_streams():
            if len(deque) > 0:
                cursors[f"{kind}:{field}:{source_name}"] = self._stamp_to_sec(deque[-1])

    def _collection_raw_joint_value(
        self,
        field: str,
        group: Any,
        msg: Any,
        timestamp: float,
        pinned_streams: dict[str, list[Any]],
    ) -> np.ndarray:
        _ = field, group, timestamp, pinned_streams
        return np.asarray(msg.position, dtype=np.float32)

    def _collection_raw_eef_value(
        self,
        field: str,
        group: Any,
        msg: Any,
        timestamp: float,
        pinned_streams: dict[str, list[Any]],
    ) -> np.ndarray:
        _ = field, pinned_streams
        return self._eef_from_msg(group, msg, timestamp)

    def _collection_raw_batch_from_msgs(
        self,
        new_messages: dict[str, list[Any]],
        pinned_streams: dict[str, list[Any]],
    ) -> CollectionRawBatch:
        batch = CollectionRawBatch()
        for kind, field, source_name, source, _deque in self._collection_raw_streams():
            key = f"{kind}:{field}:{source_name}"
            messages = new_messages.get(key, [])
            if kind == "image":
                for msg in messages:
                    batch.images.setdefault(field, []).append(
                        CollectionRawSample(
                            timestamp=self._stamp_to_sec(msg),
                            value=self._deferred_collection_image(source.name, msg),
                        )
                    )
                continue

            sample_key = f"{field}:{source_name}"
            for msg in messages:
                timestamp = self._stamp_to_sec(msg)
                if field in {"state_qpos", "action_qpos"}:
                    value = self._collection_raw_joint_value(
                        field, source, msg, timestamp, pinned_streams
                    )
                else:
                    value = self._collection_raw_eef_value(
                        field, source, msg, timestamp, pinned_streams
                    )
                batch.vectors.setdefault(sample_key, []).append(
                    CollectionRawSample(timestamp=timestamp, value=value.astype(np.float32))
                )
        return batch

    def _deferred_collection_image(self, camera_name: str, msg: Any) -> CollectionRawImage:
        return CollectionRawImage(
            lambda camera_name=camera_name, raw_msg=msg: self._decode_image_msg(
                camera_name, raw_msg
            )
        )

    def _collection_fresh_messages(
        self,
        source_deque: collections.deque,
        last_stamp: float | None,
    ) -> tuple[list[Any], float | None, float | None]:
        if not source_deque:
            return [], None, None
        if last_stamp is None:
            messages = list(source_deque)
            first_stamp = self._stamp_to_sec(messages[0])
            latest_stamp = self._stamp_to_sec(messages[-1])
            return messages, first_stamp, latest_stamp

        fresh_reversed: list[Any] = []
        first_stamp = None
        latest_stamp = None
        for msg in reversed(source_deque):
            stamp = self._stamp_to_sec(msg)
            if stamp <= last_stamp:
                break
            first_stamp = stamp
            if latest_stamp is None:
                latest_stamp = stamp
            fresh_reversed.append(msg)
        if latest_stamp is None:
            return [], None, None
        fresh_reversed.reverse()
        return fresh_reversed, first_stamp, latest_stamp

    def _collection_context_messages(
        self,
        source_deque: collections.deque,
        start_time: float,
        end_time: float,
    ) -> list[Any]:
        selected_reversed: list[Any] = []
        latest_before_start = None
        for msg in reversed(source_deque):
            stamp = self._stamp_to_sec(msg)
            if stamp > end_time:
                continue
            if stamp < start_time:
                latest_before_start = msg
                break
            selected_reversed.append(msg)
        selected_reversed.reverse()
        if latest_before_start is None:
            return selected_reversed
        return [latest_before_start, *selected_reversed]

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        """O(1) capture of raw collection stream messages since the previous tick."""
        if not self._config.collection.schema.columns:
            return None
        streams = self._collection_raw_streams()
        if not streams:
            return None

        cursors = self._collection_raw_cursors()
        new_messages: dict[str, list[Any]] = {}
        pinned_streams: dict[str, list[Any]] = {}
        snapshot_timestamp: float | None = None
        first_fresh_timestamp: float | None = None
        with self._deque_guard():
            for kind, field, source_name, _source, deque in streams:
                key = f"{kind}:{field}:{source_name}"
                last_stamp = cursors.get(key)
                fresh, first_stamp, latest = self._collection_fresh_messages(
                    deque, last_stamp
                )
                if not fresh:
                    continue
                new_messages[key] = fresh
                assert first_stamp is not None and latest is not None
                cursors[key] = latest
                first_fresh_timestamp = (
                    first_stamp
                    if first_fresh_timestamp is None
                    else min(first_fresh_timestamp, first_stamp)
                )
                snapshot_timestamp = (
                    latest if snapshot_timestamp is None else max(snapshot_timestamp, latest)
                )

            if snapshot_timestamp is None:
                return None
            assert first_fresh_timestamp is not None

            for key, deque in self._collection_raw_context_streams():
                pinned_streams[key] = self._collection_context_messages(
                    deque,
                    first_fresh_timestamp,
                    snapshot_timestamp,
                )

        batch: CollectionRawBatch | None = None

        def decode_raw() -> CollectionRawBatch:
            nonlocal batch
            if batch is None:
                batch = self._collection_raw_batch_from_msgs(new_messages, pinned_streams)
                new_messages.clear()
                pinned_streams.clear()
            return batch

        return RawCollectionSnapshot(
            timestamp=snapshot_timestamp,
            decode_raw=decode_raw,
        )
