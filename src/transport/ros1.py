"""ROS1 transport implementation — subscribes to camera images, joint states,
and EEF poses via rospy, publishes joint commands to real and sim topics,
and provides time-synchronized observation frames.
"""

from __future__ import annotations

import collections
import importlib.util
import logging
import os
import threading
from typing import Any
from urllib import parse

import numpy as np

from core.config import ConfigDict
from core.registry import TRANSPORT_REGISTRY
from core.types import Observation
from core.utils.math import build_eef_pose_from_xyzw
from robots.base import Robot
from transport.base import (
    _RosTransportBase,
    resolve_topics,
)
from transport.utils import ImageRateTracker, StreamFreshness

logger = logging.getLogger(__name__)

_ROS_RUNTIME: _RosRuntime | None = None
_LOCAL_ROS_MASTER_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _stamp_to_sec(msg: Any) -> float:
    return float(msg.header.stamp.to_sec())


class _RosRuntime:
    """Singleton container for the initialized rospy runtime and message types.

    Lazily initialized via get_ros_runtime() on first use. Holds the rospy module,
    CvBridge instance, and all ROS message types needed by Ros1Transport, avoiding
    repeated imports and ensuring rospy.init_node() is called exactly once per process.
    """

    def __init__(
        self,
        rospy: Any,
        cv_bridge: Any,
        image_type: Any,
        joint_state_type: Any,
        pose_stamped_type: Any,
        node_name: str,
    ) -> None:
        self.rospy = rospy
        self.cv_bridge = cv_bridge
        self.image_type = image_type
        self.joint_state_type = joint_state_type
        self.pose_stamped_type = pose_stamped_type
        self.node_name = node_name


def _uses_local_ros_master() -> bool:
    master_uri = os.environ.get("ROS_MASTER_URI", "").strip()
    if master_uri == "":
        return False
    hostname = parse.urlparse(master_uri).hostname
    return hostname in _LOCAL_ROS_MASTER_HOSTS


def _apply_localhost_ros_ip_fallback() -> bool:
    if os.environ.get("ROS_IP") or os.environ.get("ROS_HOSTNAME"):
        return False
    if not _uses_local_ros_master():
        return False
    if importlib.util.find_spec("netifaces") is not None:
        return False
    os.environ["ROS_IP"] = "127.0.0.1"
    logger.info("ROS_IP is unset and netifaces is unavailable; using loopback ROS_IP=127.0.0.1")
    return True


def _patch_ros_logging_find_caller() -> bool:
    import rosgraph.roslogging

    current_find_caller = rosgraph.roslogging.RospyLogger.findCaller
    if getattr(current_find_caller, "_eva_safe_patch", False):
        return False

    def _safe_find_caller(self: Any, *args: Any, **kwargs: Any) -> tuple:
        file_name, lineno, func_name = logging.Logger.findCaller(self, *args, **kwargs)[:3]
        return file_name, lineno, func_name, None

    _safe_find_caller._eva_safe_patch = True  # type: ignore[attr-defined]
    rosgraph.roslogging.RospyLogger.findCaller = _safe_find_caller  # type: ignore[assignment]
    return True


def get_ros_runtime(node_name: str) -> _RosRuntime:
    """Return the process-wide rospy runtime, initializing it on first call.

    Lazily imports rospy and message types, applies the loopback ROS_IP fallback
    and a logging patch, and calls rospy.init_node() exactly once. Subsequent calls
    reuse the existing runtime (warning if a different node_name is requested).

    Args:
        node_name: ROS node name used the first time the runtime is created.

    Returns:
        The singleton _RosRuntime holding rospy, CvBridge, and message types.
    """
    global _ROS_RUNTIME
    if _ROS_RUNTIME is not None:
        if _ROS_RUNTIME.node_name != node_name:
            logger.warning(
                "ROS runtime already initialized as node '%s'; requested node '%s' will reuse it",
                _ROS_RUNTIME.node_name,
                node_name,
            )
        return _ROS_RUNTIME

    _apply_localhost_ros_ip_fallback()
    _patch_ros_logging_find_caller()
    import rospy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, JointState

    rospy.init_node(node_name, anonymous=True)
    _ROS_RUNTIME = _RosRuntime(
        rospy=rospy,
        cv_bridge=CvBridge(),
        image_type=Image,
        joint_state_type=JointState,
        pose_stamped_type=PoseStamped,
        node_name=node_name,
    )
    return _ROS_RUNTIME


def pose_and_gripper_to_eef(pose_msg: Any, gripper_value: float) -> np.ndarray:
    """Combine a PoseStamped and a gripper opening into an EEF state vector.

    Args:
        pose_msg: ROS PoseStamped with position (x, y, z) and quaternion (xyzw).
        gripper_value: scalar gripper opening appended as the last element.

    Returns:
        float32 EEF pose vector built from position, orientation, and gripper
        (layout determined by build_eef_pose_from_xyzw).
    """
    quat = pose_msg.pose.orientation
    return build_eef_pose_from_xyzw(
        x=float(pose_msg.pose.position.x),
        y=float(pose_msg.pose.position.y),
        z=float(pose_msg.pose.position.z),
        qx=float(quat.x),
        qy=float(quat.y),
        qz=float(quat.z),
        qw=float(quat.w),
        gripper=float(gripper_value),
    )


@TRANSPORT_REGISTRY.register("ros1")
class Ros1Transport(_RosTransportBase):
    """ROS1 transport bridge for real robot communication.

    Subscribes to camera image topics, joint state topics, and optional EEF pose
    topics via rospy. Publishes joint commands to real and sim topics.
    get_frame() returns time-synchronized observations by popping messages from
    per-topic deques aligned to the earliest common timestamp.

    Supports dual-arm robots with per-group topic mapping and EEF control mode
    (PoseStamped feedback).
    """

    def __init__(self, config: ConfigDict, robot: Robot) -> None:
        self._config = config
        self._robot = robot
        self._camera_topics, self._group_topics = resolve_topics(config.transport.topics)
        self._freshness = StreamFreshness()
        self._image_rate = ImageRateTracker()
        self._deque_lock = threading.RLock()

        runtime = get_ros_runtime(config.transport.node_name)
        self._rospy = runtime.rospy
        self._bridge = runtime.cv_bridge
        self._Image = runtime.image_type
        self._JointState = runtime.joint_state_type
        self._PoseStamped = runtime.pose_stamped_type

        # Dynamic deques for cameras and actuator groups
        self._camera_deques: dict[str, collections.deque] = {}
        self._group_state_deques: dict[str, collections.deque] = {}
        self._group_eef_deques: dict[str, collections.deque] = {}

        # Collection deques (populated only when collection is enabled)
        self._collection_qpos_deques: dict[str, collections.deque] = {}
        self._collection_eef_deques: dict[str, collections.deque] = {}
        self._collection_action_qpos_deques: dict[str, collections.deque] = {}
        self._collection_action_eef_deques: dict[str, collections.deque] = {}

        # Publishers
        self._group_cmd_publishers: dict[str, Any] = {}
        self._group_sim_cmd_publishers: dict[str, Any] = {}
        self._subscribers: list[Any] = []
        self._publishers: list[Any] = []

        # Joint names per group
        self._group_joint_names: dict[str, list[str]] = {}
        for group in self._robot.actuator_groups:
            self._group_joint_names[group.name] = list(group.joint_names)
        self._last_acquire_stamp: float | None = None

        self._init_ros()

    def create_rate(self, hz: float) -> Any:
        """Return a rospy.Rate for precise ROS-clock-driven loop timing."""
        return self._rospy.Rate(hz)

    def is_shutdown(self) -> bool:
        """Return True when rospy reports the node is shutting down."""
        return self._rospy.is_shutdown()

    def _register_subscriber(self, topic: str, msg_type: Any, callback: Any) -> Any:
        subscriber = self._rospy.Subscriber(
            topic, msg_type, callback, queue_size=1000, tcp_nodelay=True
        )
        self._subscribers.append(subscriber)
        return subscriber

    def _register_publisher(self, topic: str, msg_type: Any) -> Any:
        publisher = self._rospy.Publisher(topic, msg_type, queue_size=10)
        self._publishers.append(publisher)
        return publisher

    def _init_ros(self) -> None:
        schema = self._robot.observation_schema

        # Subscribe to cameras
        for camera in schema.cameras:
            topic = self._camera_topics.get(camera.name)
            if topic is None:
                continue
            deque: collections.deque = collections.deque()
            self._camera_deques[camera.name] = deque
            self._register_subscriber(
                topic,
                self._Image,
                lambda msg, c=camera.name, d=deque: self._append_camera_msg(c, d, msg),
            )

        # Subscribe to actuator group state topics
        for group in self._robot.actuator_groups:
            group_cfg = self._group_topics.get(group.name)
            if group_cfg is None:
                continue
            state_deque: collections.deque = collections.deque()
            self._group_state_deques[group.name] = state_deque
            self._register_subscriber(
                group_cfg.state_topic,
                self._JointState,
                lambda msg, d=state_deque: self._append_msg(d, msg),
            )

            # EEF state (for EEF control mode)
            if self._config.inference_cfg.obs_space.is_eef() and group_cfg.eef_state_topic:
                eef_deque: collections.deque = collections.deque()
                self._group_eef_deques[group.name] = eef_deque
                self._register_subscriber(
                    group_cfg.eef_state_topic,
                    self._PoseStamped,
                    lambda msg, d=eef_deque: self._append_msg(d, msg),
                )

            # Command publishers
            self._group_cmd_publishers[group.name] = self._register_publisher(
                group_cfg.command_topic,
                self._JointState,
            )
            if group_cfg.sim_command_topic:
                self._group_sim_cmd_publishers[group.name] = self._register_publisher(
                    group_cfg.sim_command_topic,
                    self._JointState,
                )

        if self._config.collection.schema.columns:
            self._init_collection_subscribers()

    def _init_collection_subscribers(self) -> None:
        ros1_cfg = self._config.collection.transport.ros1
        for group in self._robot.actuator_groups:
            topics = ros1_cfg.groups.get(group.name)
            if topics is None:
                continue
            if topics.qpos_topic:
                self._subscribe_into(
                    self._collection_qpos_deques, group.name, self._JointState, topics.qpos_topic
                )
            if topics.eef_topic:
                self._subscribe_into(
                    self._collection_eef_deques, group.name, self._PoseStamped, topics.eef_topic
                )
            if topics.action_qpos_topic:
                self._subscribe_into(
                    self._collection_action_qpos_deques,
                    group.name,
                    self._JointState,
                    topics.action_qpos_topic,
                )
            if topics.action_eef_topic:
                self._subscribe_into(
                    self._collection_action_eef_deques,
                    group.name,
                    self._PoseStamped,
                    topics.action_eef_topic,
                )

    def _subscribe_into(
        self,
        deques: dict[str, collections.deque],
        group_name: str,
        msg_type: Any,
        topic: str,
    ) -> None:
        """Create a deque for ``group_name`` and subscribe ``topic`` into it."""
        deque: collections.deque = collections.deque()
        deques[group_name] = deque
        self._register_subscriber(
            topic, msg_type, lambda msg, d=deque: self._append_msg(d, msg)
        )

    def close(self) -> None:
        """Unregister all subscribers and publishers and drop publisher handles."""
        for subscriber in self._subscribers:
            subscriber.unregister()
        for publisher in self._publishers:
            publisher.unregister()
        self._subscribers.clear()
        self._publishers.clear()
        self._group_cmd_publishers.clear()
        self._group_sim_cmd_publishers.clear()

    def get_frame(self) -> Observation | None:
        """Build a time-synchronized observation from the per-topic message deques.

        Aligns to the earliest of the deques' latest header stamps and pops each
        camera/state (and optional EEF) message at that frame time. Returns None if
        any required deque is empty or not yet caught up to the frame time.

        Returns:
            Observation with per-camera images (passthrough cv2 arrays [H, W, 3]) and
            the concatenated state vector [state_dim] float32 (EEF pose in EEF mode,
            else raw joint positions), or None.
        """
        required_deques: list[collections.deque] = []
        for deque in self._camera_deques.values():
            required_deques.append(deque)
        for deque in self._group_state_deques.values():
            required_deques.append(deque)
        if self._config.inference_cfg.obs_space.is_eef():
            for deque in self._group_eef_deques.values():
                required_deques.append(deque)
        if any(len(q) == 0 for q in required_deques):
            return None

        frame_time = min(q[-1].header.stamp.to_sec() for q in required_deques)
        if any(q[-1].header.stamp.to_sec() < frame_time for q in required_deques):
            return None

        # Pop synced images
        images: dict[str, np.ndarray] = {}
        schema = self._robot.observation_schema
        for camera in schema.cameras:
            deque = self._camera_deques[camera.name]
            msg = self._pop_synced_msg(deque, frame_time)
            images[camera.observation_key] = self._bridge.imgmsg_to_cv2(msg, "passthrough")

        # Build state vector
        state_parts: list[np.ndarray] = []
        for group_name in schema.state_composition:
            group_deque = self._group_state_deques[group_name]
            joint_msg = self._pop_synced_msg(group_deque, frame_time)
            if (
                self._config.inference_cfg.obs_space.is_eef()
                and group_name in self._group_eef_deques
            ):
                eef_deque = self._group_eef_deques[group_name]
                pose_msg = self._pop_synced_msg(eef_deque, frame_time)
                group = next(g for g in self._robot.actuator_groups if g.name == group_name)
                gripper_value = (
                    float(joint_msg.position[group.gripper_index])
                    if group.gripper_index is not None
                    else 0.0
                )
                state_parts.append(pose_and_gripper_to_eef(pose_msg, gripper_value))
            else:
                state_parts.append(np.asarray(joint_msg.position, dtype=np.float32))

        state = np.concatenate(state_parts, axis=0)

        return Observation(images=images, state_qpos=state, timestamp=frame_time)

    def _pop_synced_msg(self, deque: collections.deque, frame_time: float) -> Any:
        while deque[0].header.stamp.to_sec() < frame_time:
            deque.popleft()
        return deque.popleft()

    @property
    def _collection_cfg(self) -> Any:
        return self._config.collection.transport.ros1

    def _stamp_to_sec(self, msg: Any) -> float:
        return _stamp_to_sec(msg)

    def _decode_image_msg(self, camera_name: str, msg: Any) -> np.ndarray | None:
        return self._bridge.imgmsg_to_cv2(msg, "passthrough")

    def _eef_from_msg(self, group: Any, msg: Any, frame_time: float) -> np.ndarray:
        return pose_and_gripper_to_eef(msg, self._collection_gripper(group, frame_time))

    def _collection_gripper(self, group: Any, frame_time: float) -> float:
        if group.gripper_index is None:
            return 0.0
        deque = self._collection_qpos_deques.get(group.name)
        if deque is None:
            return 0.0
        msg = self._collection_latest_msg(deque, frame_time)
        if msg is None or len(msg.position) <= group.gripper_index:
            return 0.0
        return float(msg.position[group.gripper_index])

    def has_sim_publishers(self) -> bool:
        """True if any actuator group has a sim command publisher configured."""
        return len(self._group_sim_cmd_publishers) > 0

    def get_latest_qpos(self) -> np.ndarray | None:
        """Latest joint positions concatenated across groups [qpos_dim] float32.

        Reads the newest message in each group's state deque without popping or
        synchronizing. Returns None if any group has no state message yet.
        """
        parts: list[np.ndarray] = []
        for group in self._robot.actuator_groups:
            deque = self._group_state_deques.get(group.name)
            if deque is None or len(deque) == 0:
                return None
            parts.append(np.asarray(deque[-1].position, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        """Split the action by group and publish JointState commands to ROS topics.

        Args:
            action: [action_dim] float32 flat action; sliced per actuator group by DOF.
            target: "real" to use group command topics, "sim" to use sim command topics.
        """
        parts = self._robot.split_action(action)
        stamp = self._rospy.Time.now()

        for group, part in zip(self._robot.actuator_groups, parts, strict=True):
            if target == "sim":
                publisher = self._group_sim_cmd_publishers.get(group.name)
            else:
                publisher = self._group_cmd_publishers.get(group.name)
            if publisher is not None:
                publisher.publish(self._make_joint_state(group.name, part, stamp))

    def _make_joint_state(self, group_name: str, position: np.ndarray, stamp: Any) -> Any:
        message = self._JointState()
        message.header.stamp = stamp
        message.name = list(self._group_joint_names.get(group_name, []))
        message.position = [float(v) for v in position]
        return message
