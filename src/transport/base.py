"""Abstract transport bridge — defines the TransportBridge ABC for robot
communication (get_frame, publish_action, qpos feedback) with a built-in
sleep-based rate limiter for non-ROS environments.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import importlib.util
import logging
import threading
from typing import Any, Protocol, cast

import numpy as np

from core.config import ConfigDict
from core.types import Observation, RawCollectionSnapshot
from robots.base import Robot
from transport.utils import StreamFreshness, _SimpleRate

logger = logging.getLogger(__name__)

_COLLECTION_DEQUE_MAX = 2000


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

    def supports_collection(self) -> bool:
        """True if this transport can record data-collection frames."""
        return False

    def get_intrinsics(
        self, camera_key: str
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
        """Return SDK-side intrinsics for ``camera_key`` if available.

        A live sensor transport (RealSense/ZED) can override this to report
        (K 3x3 float64, dist 1-D float64, (width, height)) straight from the
        camera SDK — the CALIBRATE workspace uses this as the option-A path.
        Default None means "no SDK intrinsics"; callers should fall back to
        solving from captured board images (option B).
        """
        del camera_key
        return None

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
        capture time, deferring all decode/copy/convert work to the returned
        snapshot's decode() closure run on the ingest thread. Returns None when
        no new source frame is available (the source timestamp has not strictly
        advanced), which also serves as dedup. Non-collection transports keep
        the default and stay on the synchronous get_collection_frame() path.
        """
        return None

    def stop_collection(self) -> None:
        """Signal the source to stop emitting collection frames (no-op default)."""
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
    (ros2 split-gripper topics).
    """

    state_topic: str
    command_topic: str
    sim_command_topic: str | None = None
    eef_state_topic: str | None = None
    gripper_state_topic: str | None = None
    gripper_command_topic: str | None = None


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
    _camera_deques: dict[str, collections.deque]
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

    def _append_msg(self, deque: collections.deque, msg: Any) -> None:
        if len(deque) >= _COLLECTION_DEQUE_MAX:
            deque.popleft()
        deque.append(msg)
        self._freshness.mark()

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

    def _collection_images(self, frame_time: float) -> dict[str, np.ndarray] | None:
        images: dict[str, np.ndarray] = {}
        for camera in self._collection_camera_specs():
            deque = self._camera_deques.get(camera.name)
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
        if primary_camera:
            for camera in self._robot.observation_schema.cameras:
                if primary_camera not in (camera.observation_key, camera.name):
                    continue
                deque = self._camera_deques.get(camera.name)
                if deque is None or len(deque) == 0:
                    return None
                return self._stamp_to_sec(deque[-1])
            return None

        required: list[collections.deque] = []
        for camera in self._collection_camera_specs():
            deque = self._camera_deques.get(camera.name)
            if deque is None:
                return None
            required.append(deque)

        columns = self._config.collection.schema.columns
        if "state_qpos" in columns:
            required.extend(self._collection_qpos_deques.values())
        if "state_eef" in columns:
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
        if "state_qpos" in columns:
            state_qpos = self._collection_joint_vector(self._collection_qpos_deques, frame_time)
            if state_qpos is None:
                return None
        if "state_eef" in columns:
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

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        """O(1) capture: pin image msgs + compute cheap value vectors; defer the
        expensive per-image decode to the snapshot's decode() closure.

        Dedup is the strict-monotonic frame_time gate: when the camera header
        stamp has not advanced, returns None instead of re-recording the msg the
        ingest thread already holds.
        """
        if not self._config.collection.schema.columns:
            return None
        frame_time = self._collection_frame_time()
        if frame_time is None:
            return None
        if self._last_acquire_stamp is not None and frame_time <= self._last_acquire_stamp:
            return None

        gathered = self._gather_collection_columns(frame_time)
        if gathered is None:
            return None
        state_qpos, state_eef, action_qpos, action_eef = gathered

        # Pin the still-encoded image msgs without decoding; missing any camera
        # invalidates the whole frame, matching get_collection_frame semantics.
        image_msgs: dict[str, tuple[str, Any]] = {}
        for camera in self._collection_camera_specs():
            deque = self._camera_deques.get(camera.name)
            if deque is None:
                return None
            msg = self._collection_latest_msg(deque, frame_time)
            if msg is None:
                return None
            image_msgs[camera.observation_key] = (camera.name, msg)

        self._last_acquire_stamp = frame_time

        def decode() -> Observation:
            """Decode the pinned image msgs and assemble the full Observation."""
            images: dict[str, np.ndarray] = {}
            for obs_key, (camera_name, msg) in image_msgs.items():
                image = self._decode_image_msg(camera_name, msg)
                if image is None:
                    raise ValueError(f"image decode failed for {camera_name}")
                images[obs_key] = np.ascontiguousarray(image).copy()
            return Observation(
                images=images,
                state_qpos=cast(np.ndarray, state_qpos),
                state_eef=state_eef,
                action_qpos=action_qpos,
                action_eef=action_eef,
                timestamp=frame_time,
            )

        return RawCollectionSnapshot(timestamp=frame_time, decode=decode)
