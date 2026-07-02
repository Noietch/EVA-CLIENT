"""ROS2 transport implementation — subscribes to camera images and joint states
via rclpy, publishes joint commands, and provides time-synchronized observation
frames using message header stamps.

Mirrors the structure of ros1.py (per-robot topic defaults + YAML overrides,
per-deque timestamp alignment) but on rclpy. rclpy and message types are imported
lazily so this module never breaks package import on hosts without ROS2.
"""

from __future__ import annotations

import collections
import io
import logging
import threading
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from core.config import ConfigDict
from core.registry import TRANSPORT_REGISTRY
from core.types import Observation
from robots.base import Robot
from transport.base import (
    _RosTransportBase,
    resolve_topics,
)
from transport.utils import StreamFreshness

logger = logging.getLogger(__name__)

_ROS2_RUNTIME: _Ros2Runtime | None = None


class _Ros2Runtime:
    """Singleton container for the initialized rclpy runtime, node, message types,
    and a background spin thread. Lazily initialized on first use, ensuring
    rclpy.init() and node creation happen exactly once per process."""

    def __init__(
        self,
        rclpy: Any,
        node: Any,
        cv_bridge: Any,
        image_type: Any,
        compressed_image_type: Any,
        joint_state_type: Any,
        pose_stamped_type: Any,
        node_name: str,
    ) -> None:
        self.rclpy = rclpy
        self.node = node
        self.cv_bridge = cv_bridge
        self.image_type = image_type
        self.compressed_image_type = compressed_image_type
        self.joint_state_type = joint_state_type
        self.pose_stamped_type = pose_stamped_type
        self.node_name = node_name


def get_ros2_runtime(node_name: str) -> _Ros2Runtime:
    """Return the process-wide rclpy runtime, initializing it on first call.

    Lazily imports rclpy and message types, calls rclpy.init() if needed, creates
    the node, and starts a daemon spin thread driving a SingleThreadedExecutor.
    Subsequent calls reuse the existing runtime.

    Args:
        node_name: ROS2 node name used the first time the runtime is created.

    Returns:
        The singleton _Ros2Runtime holding rclpy, the node, CvBridge, and types.
    """
    global _ROS2_RUNTIME
    if _ROS2_RUNTIME is not None:
        return _ROS2_RUNTIME

    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image, JointState

    if not rclpy.ok():
        rclpy.init()
    node = Node(node_name)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    _ROS2_RUNTIME = _Ros2Runtime(
        rclpy=rclpy,
        node=node,
        cv_bridge=CvBridge(),
        image_type=Image,
        compressed_image_type=CompressedImage,
        joint_state_type=JointState,
        pose_stamped_type=PoseStamped,
        node_name=node_name,
    )
    return _ROS2_RUNTIME


def _make_live_qos(depth: int = 10) -> Any:
    """Build the best-effort, volatile QoS profile used for all live ROS2 entities."""
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _stamp_to_sec(msg: Any) -> float:
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@TRANSPORT_REGISTRY.register("ros2")
class Ros2Transport(_RosTransportBase):
    """ROS2 transport bridge for real robot communication via rclpy.

    Subscribes to camera image topics and joint state topics, publishes joint
    commands. get_frame() returns time-synchronized observations by popping
    messages from per-topic deques aligned to the earliest common header stamp.
    """

    def __init__(self, config: ConfigDict, robot: Robot) -> None:
        self._config = config
        self._robot = robot
        self._camera_topics, self._group_topics = resolve_topics(config.transport.topics)
        self._freshness = StreamFreshness()

        runtime = get_ros2_runtime(config.transport.node_name)
        self._runtime = runtime
        self._node = runtime.node
        self._bridge = runtime.cv_bridge
        self._Image = runtime.image_type
        self._CompressedImage = runtime.compressed_image_type
        self._JointState = runtime.joint_state_type
        self._PoseStamped = runtime.pose_stamped_type

        self._camera_deques: dict[str, collections.deque] = {}
        self._camera_is_compressed: dict[str, bool] = {}
        self._group_state_deques: dict[str, collections.deque] = {}
        self._group_eef_deques: dict[str, collections.deque] = {}
        self._group_gripper_deques: dict[str, collections.deque] = {}
        self._collection_qpos_deques: dict[str, collections.deque] = {}
        self._collection_eef_deques: dict[str, collections.deque] = {}
        self._collection_action_qpos_deques: dict[str, collections.deque] = {}
        self._collection_action_eef_deques: dict[str, collections.deque] = {}
        self._group_cmd_publishers: dict[str, Any] = {}
        self._group_gripper_publishers: dict[str, Any] = {}
        self._group_joint_names: dict[str, list[str]] = {
            g.name: list(g.joint_names) for g in self._robot.actuator_groups
        }
        self._last_acquire_stamp: float | None = None

        self._init_ros()

    def _init_ros(self) -> None:
        schema = self._robot.observation_schema
        live_qos = _make_live_qos()

        for camera in schema.cameras:
            topic = self._camera_topics.get(camera.name)
            if topic is None:
                continue
            deque: collections.deque = collections.deque()
            self._camera_deques[camera.name] = deque
            is_compressed = topic.endswith("/compressed")
            self._camera_is_compressed[camera.name] = is_compressed
            msg_type = self._CompressedImage if is_compressed else self._Image
            self._node.create_subscription(
                msg_type, topic, lambda msg, d=deque: self._append_msg(d, msg), live_qos
            )

        for group in self._robot.actuator_groups:
            group_cfg = self._group_topics.get(group.name)
            if group_cfg is None:
                continue
            state_deque: collections.deque = collections.deque()
            self._group_state_deques[group.name] = state_deque
            self._node.create_subscription(
                self._JointState,
                group_cfg.state_topic,
                lambda msg, d=state_deque: self._append_msg(d, msg),
                live_qos,
            )
            if group_cfg.eef_state_topic is not None:
                eef_deque: collections.deque = collections.deque()
                self._group_eef_deques[group.name] = eef_deque
                self._node.create_subscription(
                    self._PoseStamped,
                    group_cfg.eef_state_topic,
                    lambda msg, d=eef_deque: self._append_msg(d, msg),
                    live_qos,
                )
            if group_cfg.gripper_state_topic is not None:
                gripper_deque: collections.deque = collections.deque()
                self._group_gripper_deques[group.name] = gripper_deque
                self._node.create_subscription(
                    self._JointState,
                    group_cfg.gripper_state_topic,
                    lambda msg, d=gripper_deque: self._append_msg(d, msg),
                    live_qos,
                )
            self._group_cmd_publishers[group.name] = self._node.create_publisher(
                self._JointState, group_cfg.command_topic, live_qos
            )
            if group_cfg.gripper_command_topic is not None:
                self._group_gripper_publishers[group.name] = self._node.create_publisher(
                    self._JointState, group_cfg.gripper_command_topic, live_qos
                )

        if self._config.collection.schema.columns:
            ros2_cfg = self._config.collection.transport.ros2
            for group in self._robot.actuator_groups:
                topics = ros2_cfg.groups.get(group.name)
                if topics is None:
                    continue
                if topics.qpos_topic:
                    self._subscribe_into(
                        self._collection_qpos_deques,
                        group.name,
                        self._JointState,
                        topics.qpos_topic,
                        live_qos,
                    )
                if topics.eef_topic:
                    self._subscribe_into(
                        self._collection_eef_deques,
                        group.name,
                        self._PoseStamped,
                        topics.eef_topic,
                        live_qos,
                    )
                if topics.action_qpos_topic:
                    self._subscribe_into(
                        self._collection_action_qpos_deques,
                        group.name,
                        self._JointState,
                        topics.action_qpos_topic,
                        live_qos,
                    )
                if topics.action_eef_topic:
                    self._subscribe_into(
                        self._collection_action_eef_deques,
                        group.name,
                        self._PoseStamped,
                        topics.action_eef_topic,
                        live_qos,
                    )

    def _subscribe_into(
        self,
        deques: dict[str, collections.deque],
        group_name: str,
        msg_type: Any,
        topic: str,
        qos: Any,
    ) -> None:
        """Create a deque for ``group_name`` and subscribe ``topic`` into it."""
        deque: collections.deque = collections.deque()
        deques[group_name] = deque
        self._node.create_subscription(
            msg_type,
            topic,
            lambda msg, d=deque: self._append_msg(d, msg),
            qos,
        )

    def get_frame(self) -> Observation | None:
        """Build a time-synchronized observation from the per-topic message deques.

        Aligns to the earliest of the deques' latest header stamps, pops each camera
        and state (joint positions, or EEF poses in EEF mode) message at that frame
        time, and decodes images (raw or compressed). Returns None if any required
        deque is empty/not caught up or an image fails to decode.

        Returns:
            Observation with per-camera images [H, W, 3] (BGR uint8) and the
            concatenated state vector [state_dim] float32, or None.
        """
        required: list[collections.deque] = []
        required.extend(self._camera_deques.values())
        if self._config.inference_cfg.obs_space.is_eef():
            required.extend(self._group_eef_deques.values())
        else:
            required.extend(self._group_state_deques.values())
        required.extend(self._group_gripper_deques.values())
        if any(len(q) == 0 for q in required):
            return None

        frame_time = min(_stamp_to_sec(q[-1]) for q in required)
        if any(_stamp_to_sec(q[-1]) < frame_time for q in required):
            return None

        images: dict[str, np.ndarray] = {}
        schema = self._robot.observation_schema
        for camera in schema.cameras:
            deque = self._camera_deques[camera.name]
            msg = self._pop_synced_msg(deque, frame_time)
            image = self._decode_image_msg(camera.name, msg)
            if image is None:
                return None
            images[camera.observation_key] = image

        state_parts: list[np.ndarray] = []
        if self._config.inference_cfg.obs_space.is_eef():
            for group in self._robot.actuator_groups:
                eef_deque = self._group_eef_deques.get(group.name)
                if eef_deque is None:
                    return None
                eef_msg = self._pop_synced_msg(eef_deque, frame_time)
                state_parts.append(self._eef_msg_to_state(group.name, eef_msg, frame_time))
        else:
            for group_name in schema.state_composition:
                deque = self._group_state_deques[group_name]
                joint_msg = self._pop_synced_msg(deque, frame_time)
                group = next(g for g in self._robot.actuator_groups if g.name == group_name)
                state_parts.append(self._joint_msg_to_state(group, joint_msg, frame_time))
        state = np.concatenate(state_parts, axis=0)

        return Observation(images=images, state_qpos=state)

    @property
    def _collection_cfg(self) -> Any:
        return self._config.collection.transport.ros2

    def _stamp_to_sec(self, msg: Any) -> float:
        return _stamp_to_sec(msg)

    def _eef_from_msg(self, group: Any, msg: Any, frame_time: float) -> np.ndarray:
        return self._eef_msg_to_state(group.name, msg, frame_time)

    def _decode_image_msg(self, camera_name: str, msg: Any) -> np.ndarray | None:
        if not self._camera_is_compressed.get(camera_name, False):
            return self._bridge.imgmsg_to_cv2(msg, "passthrough")
        try:
            image = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        except UnidentifiedImageError:
            logger.debug("Failed to decode ROS2 compressed image", exc_info=True)
            return None
        return np.asarray(image)[:, :, ::-1].copy()

    def _pop_latest_before(self, deque: collections.deque, frame_time: float) -> Any | None:
        while len(deque) > 1 and _stamp_to_sec(deque[1]) <= frame_time:
            deque.popleft()
        if len(deque) == 0:
            return None
        return deque[0]

    def _gripper_value(self, group_name: str, frame_time: float) -> float:
        gripper_deque = self._group_gripper_deques.get(group_name)
        if gripper_deque is None:
            return 0.0
        msg = self._pop_latest_before(gripper_deque, frame_time)
        if msg is None or len(msg.position) == 0:
            return 0.0
        return float(msg.position[0])

    def _joint_msg_to_state(self, group: Any, msg: Any, frame_time: float) -> np.ndarray:
        position = np.asarray(msg.position, dtype=np.float32)
        if group.gripper_index is None or group.name not in self._group_gripper_deques:
            return position
        gripper = self._gripper_value(group.name, frame_time)
        return np.concatenate(
            [position[: group.gripper_index], np.asarray([gripper], dtype=np.float32)],
            axis=0,
        )

    def _eef_msg_to_state(self, group_name: str, msg: Any, frame_time: float) -> np.ndarray:
        pose = msg.pose
        eef_pose = np.asarray(
            [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.w),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                self._gripper_value(group_name, frame_time),
            ],
            dtype=np.float32,
        )
        return eef_pose

    def _pop_synced_msg(self, deque: collections.deque, frame_time: float) -> Any:
        while _stamp_to_sec(deque[0]) < frame_time:
            deque.popleft()
        return deque.popleft()

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        """Split the action by group and publish JointState commands via rclpy.

        Where a group has a separate gripper command topic, the gripper element is
        published on its own publisher and dropped from the arm command. Actions
        targeting "sim" are ignored (ROS2 has no sim publishers).

        Args:
            action: [action_dim] float32 flat action, sliced per group by DOF.
            target: "real" to publish; "sim" is a no-op.
        """
        if target == "sim":
            return
        parts = self._robot.split_action(action)
        stamp = self._node.get_clock().now().to_msg()
        for group, part in zip(self._robot.actuator_groups, parts, strict=True):
            publisher = self._group_cmd_publishers.get(group.name)
            if publisher is not None:
                joints = part
                if group.gripper_index is not None and group.name in self._group_gripper_publishers:
                    joints = part[: group.gripper_index]
                publisher.publish(self._make_joint_state(group.name, joints, stamp))
            gripper_publisher = self._group_gripper_publishers.get(group.name)
            if gripper_publisher is not None and group.gripper_index is not None:
                gripper_position = np.asarray([part[group.gripper_index]], dtype=np.float32)
                gripper_publisher.publish(
                    self._make_joint_state(group.name, gripper_position, stamp, gripper_only=True)
                )

    def _make_joint_state(
        self,
        group_name: str,
        position: np.ndarray,
        stamp: Any,
        gripper_only: bool = False,
    ) -> Any:
        message = self._JointState()
        message.header.stamp = stamp
        joint_names = list(self._group_joint_names.get(group_name, []))
        if gripper_only:
            joint_names = joint_names[-1:]
        elif len(position) < len(joint_names):
            joint_names = joint_names[: len(position)]
        message.name = joint_names
        message.position = [float(v) for v in position]
        return message

    def get_latest_qpos(self) -> np.ndarray | None:
        """Latest joint positions concatenated across groups [qpos_dim] float32.

        Reads the newest message in each group's state deque (merging the separate
        gripper feedback where present) without popping or synchronizing. Returns
        None if any group has no state message yet.
        """
        parts: list[np.ndarray] = []
        for group in self._robot.actuator_groups:
            deque = self._group_state_deques.get(group.name)
            if deque is None or len(deque) == 0:
                return None
            parts.append(self._joint_msg_to_state(group, deque[-1], _stamp_to_sec(deque[-1])))
        return np.concatenate(parts, axis=0)

    def close(self) -> None:
        """Destroy all command and gripper publishers and drop their handles."""
        for publisher in self._group_cmd_publishers.values():
            publisher.destroy()
        for publisher in self._group_gripper_publishers.values():
            publisher.destroy()
        self._group_cmd_publishers.clear()
        self._group_gripper_publishers.clear()
