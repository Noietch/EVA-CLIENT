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
from transport.utils import ImageRateTracker, StreamFreshness

logger = logging.getLogger(__name__)

_ROS2_RUNTIME: _Ros2Runtime | None = None
_HIL_CONTROL_MODES = frozenset({"absolute", "relative"})
_CAMERA_QOS_DEPTH = 120


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
        self._image_rate = ImageRateTracker()
        self._deque_lock = threading.RLock()

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
        self._collection_qpos_gripper_deques: dict[str, collections.deque] = {}
        self._collection_eef_deques: dict[str, collections.deque] = {}
        self._collection_action_qpos_deques: dict[str, collections.deque] = {}
        self._collection_action_qpos_gripper_deques: dict[str, collections.deque] = {}
        self._collection_action_eef_deques: dict[str, collections.deque] = {}
        self._group_cmd_publishers: dict[str, Any] = {}
        self._group_gripper_publishers: dict[str, Any] = {}
        self._hil_relay_enabled = False
        self._hil_control_mode = "absolute"
        self._hil_cat_anchors: dict[str, np.ndarray] = {}
        self._hil_robot_anchors: dict[str, np.ndarray] = {}
        self._last_group_joint_commands: dict[str, np.ndarray] = {}
        self._last_group_gripper_commands: dict[str, float] = {}
        self._group_joint_names: dict[str, list[str]] = {
            g.name: list(g.joint_names) for g in self._robot.actuator_groups
        }
        self._last_acquire_stamp: float | None = None

        self.set_hil_control_mode(self._configured_hil_control_mode())
        self._init_ros()

    def _configured_hil_control_mode(self) -> str:
        rollout = self._config.get("rollout") or {}
        intervention = rollout.get("intervention") or {}
        return str(intervention.get("control_mode", "absolute"))

    @property
    def hil_control_mode(self) -> str:
        return self._hil_control_mode

    def set_hil_control_mode(self, mode: str) -> None:
        if mode not in _HIL_CONTROL_MODES:
            allowed = ", ".join(sorted(_HIL_CONTROL_MODES))
            raise ValueError(f"Unsupported HIL control mode {mode!r}; expected: {allowed}")
        self._hil_control_mode = mode
        self.reset_hil_control()

    def reset_hil_control(self) -> None:
        self._hil_cat_anchors.clear()
        self._hil_robot_anchors.clear()

    def set_hil_relay_enabled(self, enabled: bool) -> None:
        if self._hil_relay_enabled == enabled:
            return
        self._hil_relay_enabled = enabled
        self.reset_hil_control()

    def _init_ros(self) -> None:
        schema = self._robot.observation_schema
        live_qos = _make_live_qos()
        camera_qos = _make_live_qos(depth=_CAMERA_QOS_DEPTH)
        hil_qos = _make_live_qos(depth=1)

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
                msg_type,
                topic,
                lambda msg, c=camera.name, d=deque: self._append_camera_msg(c, d, msg),
                camera_qos,
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
                if topics.get("qpos_topic"):
                    self._subscribe_into(
                        self._collection_qpos_deques,
                        group.name,
                        self._JointState,
                        topics.qpos_topic,
                        live_qos,
                    )
                if topics.get("qpos_gripper_topic"):
                    self._subscribe_into(
                        self._collection_qpos_gripper_deques,
                        group.name,
                        self._JointState,
                        topics.qpos_gripper_topic,
                        live_qos,
                    )
                if topics.get("eef_topic"):
                    self._subscribe_into(
                        self._collection_eef_deques,
                        group.name,
                        self._PoseStamped,
                        topics.eef_topic,
                        live_qos,
                    )
                if topics.get("action_qpos_topic"):
                    self._subscribe_into(
                        self._collection_action_qpos_deques,
                        group.name,
                        self._JointState,
                        topics.action_qpos_topic,
                        live_qos,
                    )
                action_qpos_gripper_source = topics.get("action_qpos_gripper_source", "topic")
                if action_qpos_gripper_source not in {"topic", "operator"}:
                    raise ValueError(
                        f"Unsupported action_qpos_gripper_source "
                        f"{action_qpos_gripper_source!r} for {group.name}"
                    )
                if action_qpos_gripper_source == "operator":
                    self._collection_action_qpos_gripper_deques[group.name] = collections.deque()
                elif topics.get("action_qpos_gripper_topic"):
                    self._subscribe_into(
                        self._collection_action_qpos_gripper_deques,
                        group.name,
                        self._JointState,
                        topics.action_qpos_gripper_topic,
                        live_qos,
                    )
                if topics.get("action_eef_topic"):
                    self._subscribe_into(
                        self._collection_action_eef_deques,
                        group.name,
                        self._PoseStamped,
                        topics.action_eef_topic,
                        live_qos,
                    )
                if topics.get("hil_qpos_topic"):
                    self._node.create_subscription(
                        self._JointState,
                        topics.hil_qpos_topic,
                        lambda msg, g=group: self._handle_hil_joint_msg(g, msg),
                        hil_qos,
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

    def start_collection(self) -> None:
        """Seed collection-only gripper action from the current target/feedback."""
        for group in self._robot.actuator_groups:
            if not self._uses_operator_action_gripper(group):
                continue
            value = self._last_group_gripper_commands.get(group.name)
            if value is None:
                value = self._latest_group_gripper_feedback(group)
            if value is not None:
                self.record_collection_gripper_action(group.name, value)

    def clear_collection_backlog(self) -> float | None:
        """Advance raw cursors after seeding operator-sourced gripper action."""
        self.start_collection()
        return super().clear_collection_backlog()

    def record_collection_gripper_action(
        self, group_name: str, value: float, stamp: Any | None = None
    ) -> None:
        """Record a collection-only gripper action target for operator-sourced configs."""
        group = self._group_by_name(group_name)
        if group is None or not self._uses_operator_action_gripper(group):
            return
        deque = self._collection_action_qpos_gripper_deques.get(group_name)
        if deque is None:
            return
        if stamp is None:
            stamp = self._node.get_clock().now().to_msg()
        position = np.asarray([value], dtype=np.float32)
        self._append_msg(
            deque,
            self._make_joint_state(group_name, position, stamp, gripper_only=True),
        )

    def _group_by_name(self, group_name: str) -> Any | None:
        for group in self._robot.actuator_groups:
            if group.name == group_name:
                return group
        return None

    def _uses_operator_action_gripper(self, group: Any) -> bool:
        topics = self._config.collection.transport.ros2.groups.get(group.name)
        return topics is not None and topics.get("action_qpos_gripper_source") == "operator"

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
            if msg is None:
                return None
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
                if eef_msg is None:
                    return None
                state_parts.append(self._eef_msg_to_state(group.name, eef_msg, frame_time))
        else:
            for group_name in schema.state_composition:
                deque = self._group_state_deques[group_name]
                joint_msg = self._pop_synced_msg(deque, frame_time)
                if joint_msg is None:
                    return None
                group = next(g for g in self._robot.actuator_groups if g.name == group_name)
                state_parts.append(self._joint_msg_to_state(group, joint_msg, frame_time))
        state = np.concatenate(state_parts, axis=0)

        return Observation(images=images, state_qpos=state, timestamp=frame_time)

    @property
    def _collection_cfg(self) -> Any:
        return self._config.collection.transport.ros2

    def get_camera_frame(self, key: str) -> np.ndarray | None:
        """Return the latest decoded image for one camera without consuming queues.

        Args:
            key: Camera observation key or ROS2 camera name.

        Returns:
            Image [H, W, 3] uint8, or None if no message is available.
        """
        for camera in self._robot.observation_schema.cameras:
            if key not in (camera.observation_key, camera.name):
                continue
            deque = self._camera_deques.get(camera.name)
            if deque is None:
                return None
            try:
                msg = deque[-1]
            except IndexError:
                return None
            return self._decode_image_msg(camera.name, msg)
        return None

    def get_camera_jpeg(self, key: str) -> bytes | None:
        """Return the latest compressed JPEG payload for one camera without consuming queues.

        Args:
            key: Camera observation key or ROS2 camera name.

        Returns:
            JPEG bytes, or None if the camera is unavailable or not compressed.
        """
        for camera in self._robot.observation_schema.cameras:
            if key not in (camera.observation_key, camera.name):
                continue
            if not self._camera_is_compressed.get(camera.name, False):
                return None
            deque = self._camera_deques.get(camera.name)
            if deque is None:
                return None
            try:
                msg = deque[-1]
            except IndexError:
                return None
            return bytes(msg.data)
        return None

    def _stamp_to_sec(self, msg: Any) -> float:
        return _stamp_to_sec(msg)

    def _eef_from_msg(self, group: Any, msg: Any, frame_time: float) -> np.ndarray:
        return self._eef_msg_to_state(group.name, msg, frame_time)

    def _collection_joint_vector(
        self,
        deques_by_group: dict[str, collections.deque],
        frame_time: float,
        value_attr: str = "position",
        gripper_deques_by_group: dict[str, collections.deque] | None = None,
        action_gripper: bool = False,
    ) -> np.ndarray | None:
        if gripper_deques_by_group is None:
            if deques_by_group is self._collection_qpos_deques:
                gripper_deques_by_group = self._collection_qpos_gripper_deques
            elif deques_by_group is self._collection_action_qpos_deques:
                gripper_deques_by_group = self._collection_action_qpos_gripper_deques
                action_gripper = True
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
            part = np.asarray(getattr(msg, value_attr), dtype=np.float32)
            if (
                group.gripper_index is not None
                and part.shape == (group.dof - 1,)
                and gripper_deques_by_group is not None
                and group.name in gripper_deques_by_group
            ):
                gripper = self._collection_gripper_value(
                    group,
                    gripper_deques_by_group[group.name],
                    frame_time,
                    action_gripper=action_gripper,
                )
                if gripper is None:
                    return None
                part = np.insert(part, group.gripper_index, gripper).astype(np.float32)
            parts.append(part)
        return np.concatenate(parts, axis=0)

    def _collection_raw_context_streams(self) -> list[tuple[str, collections.deque]]:
        streams: list[tuple[str, collections.deque]] = []
        for group_name, deque in self._collection_qpos_gripper_deques.items():
            streams.append((f"gripper:state_qpos:{group_name}", deque))
        for group_name, deque in self._collection_action_qpos_gripper_deques.items():
            streams.append((f"gripper:action_qpos:{group_name}", deque))
        for group_name, deque in self._group_gripper_deques.items():
            streams.append((f"gripper:state_eef:{group_name}", deque))
            streams.append((f"gripper:action_eef:{group_name}", deque))
        return streams

    def _collection_raw_joint_value(
        self,
        field: str,
        group: Any,
        msg: Any,
        timestamp: float,
        pinned_streams: dict[str, list[Any]],
    ) -> np.ndarray:
        part = np.asarray(msg.position, dtype=np.float32)
        if group.gripper_index is None or part.shape != (group.dof - 1,):
            return part
        gripper = self._raw_gripper_value(
            f"gripper:{field}:{group.name}",
            timestamp,
            pinned_streams,
            action_gripper=(field == "action_qpos"),
        )
        if gripper is None:
            return part
        return np.insert(part, group.gripper_index, gripper).astype(np.float32)

    def _raw_gripper_value(
        self,
        key: str,
        timestamp: float,
        pinned_streams: dict[str, list[Any]],
        *,
        action_gripper: bool,
    ) -> float | None:
        msg = self._latest_pinned_before_or_at(pinned_streams.get(key, []), timestamp)
        if msg is None:
            return None
        position = getattr(msg, "position", ())
        if len(position) == 0:
            return None
        value = float(position[0])
        group_name = key.rsplit(":", 1)[-1]
        topics = self._config.collection.transport.ros2.groups.get(group_name)
        binary = False
        if topics is not None:
            config_key = "action_qpos_gripper_binary" if action_gripper else "qpos_gripper_binary"
            binary = bool(topics.get(config_key, False))
        if not binary:
            return value
        return self._binary_gripper_value(value)

    def _latest_pinned_before_or_at(self, messages: list[Any], timestamp: float) -> Any | None:
        latest = None
        for msg in messages:
            if self._stamp_to_sec(msg) <= timestamp:
                latest = msg
            else:
                break
        return latest

    def collection_diagnostics(self) -> str:
        """Describe which configured collection streams are missing or stale."""
        if not self._config.collection.schema.columns:
            return "collection schema has no columns"
        frame_time = self._collection_frame_time()
        max_skew = float(self._collection_cfg.max_frame_skew_sec)
        problems: list[str] = []

        def check_stream(
            label: str,
            topic: str | None,
            deque: collections.deque | None,
            *,
            descriptor: str | None = None,
            latched: bool = False,
        ) -> None:
            if descriptor is None:
                if not topic:
                    problems.append(f"{label} topic not configured")
                    return
                descriptor = f"topic={topic}"
            if deque is None:
                problems.append(f"{label} {descriptor} not subscribed")
                return
            if len(deque) == 0:
                problems.append(f"{label} {descriptor} no messages")
                return
            if frame_time is None:
                return
            earliest = self._stamp_to_sec(deque[0])
            latest = self._stamp_to_sec(deque[-1])
            if earliest > frame_time:
                problems.append(
                    f"{label} {descriptor} starts after frame_time "
                    f"{frame_time:.3f}s (earliest {earliest:.3f}s)"
                )
            elif not latched and max_skew > 0.0 and latest < frame_time - max_skew:
                problems.append(
                    f"{label} {descriptor} stale at frame_time {frame_time:.3f}s "
                    f"(latest {latest:.3f}s, max_skew {max_skew:.3f}s)"
                )

        for camera in self._collection_camera_specs():
            check_stream(
                f"camera:{camera.observation_key}",
                self._camera_topics.get(camera.name),
                self._camera_deques.get(camera.name),
            )

        columns = self._config.collection.schema.columns
        for group in self._robot.actuator_groups:
            topics = self._config.collection.transport.ros2.groups.get(group.name)
            if topics is None:
                problems.append(f"group:{group.name} collection topics not configured")
                continue
            if "qpos" in columns:
                check_stream(
                    f"qpos:{group.name}",
                    topics.get("qpos_topic"),
                    self._collection_qpos_deques.get(group.name),
                )
                if topics.get("qpos_gripper_topic"):
                    check_stream(
                        f"qpos_gripper:{group.name}",
                        topics.get("qpos_gripper_topic"),
                        self._collection_qpos_gripper_deques.get(group.name),
                    )
            if "eef" in columns:
                check_stream(
                    f"eef:{group.name}",
                    topics.get("eef_topic"),
                    self._collection_eef_deques.get(group.name),
                )
            if "action_qpos" in columns:
                check_stream(
                    f"action_qpos:{group.name}",
                    topics.get("action_qpos_topic"),
                    self._collection_action_qpos_deques.get(group.name),
                )
                action_gripper_source = topics.get("action_qpos_gripper_source", "topic")
                if action_gripper_source == "operator":
                    check_stream(
                        f"action_qpos_gripper:{group.name}",
                        None,
                        self._collection_action_qpos_gripper_deques.get(group.name),
                        descriptor="source=operator",
                        latched=True,
                    )
                elif topics.get("action_qpos_gripper_topic"):
                    check_stream(
                        f"action_qpos_gripper:{group.name}",
                        topics.get("action_qpos_gripper_topic"),
                        self._collection_action_qpos_gripper_deques.get(group.name),
                    )
            if "action_eef" in columns:
                check_stream(
                    f"action_eef:{group.name}",
                    topics.get("action_eef_topic"),
                    self._collection_action_eef_deques.get(group.name),
                )

        if problems:
            return "missing collection streams: " + "; ".join(problems)
        if frame_time is None:
            return "collection streams have messages, but no frame_time is available"
        return (
            f"collection streams have messages, but no synchronized frame at "
            f"{frame_time:.3f}s (max_skew {max_skew:.3f}s)"
        )

    def _collection_gripper_value(
        self,
        group: Any,
        deque: collections.deque,
        frame_time: float,
        action_gripper: bool,
    ) -> float | None:
        if action_gripper and self._uses_operator_action_gripper(group):
            msg = self._latest_before_or_at(deque, frame_time)
        else:
            msg = self._collection_latest_msg(deque, frame_time)
        if msg is None:
            return None
        topics = self._config.collection.transport.ros2.groups.get(group.name)
        position = getattr(msg, "position", ())
        if len(position) == 0:
            return None
        value = float(position[0])
        binary = False
        if topics is not None:
            key = "action_qpos_gripper_binary" if action_gripper else "qpos_gripper_binary"
            binary = bool(topics.get(key, False))
        if not binary:
            return value
        return self._binary_gripper_value(value)

    def _binary_gripper_value(self, value: float) -> float:
        threshold = self._config.robot.gripper_threshold
        if threshold is None:
            threshold = (self._config.robot.gripper_open + self._config.robot.gripper_close) * 0.5
        if self._config.robot.gripper_open >= self._config.robot.gripper_close:
            return (
                self._config.robot.gripper_open
                if value >= threshold
                else self._config.robot.gripper_close
            )
        return (
            self._config.robot.gripper_open
            if value <= threshold
            else self._config.robot.gripper_close
        )

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

    def _latest_group_qpos(self, group: Any) -> np.ndarray | None:
        msg = self._latest_group_qpos_msg(group)
        if msg is None:
            return None
        return self._joint_msg_to_state(group, msg, _stamp_to_sec(msg))

    def _latest_group_qpos_msg(self, group: Any) -> Any | None:
        deque = self._group_state_deques.get(group.name)
        if deque is None or len(deque) == 0:
            return None
        return deque[-1]

    def _latest_group_gripper_feedback(self, group: Any) -> float | None:
        deque = self._group_gripper_deques.get(group.name)
        if deque is None or len(deque) == 0:
            return None
        position = getattr(deque[-1], "position", ())
        if len(position) == 0:
            return None
        return float(position[0])

    def _hil_robot_state_for_message(
        self,
        group: Any,
        cat_position: np.ndarray,
        robot_qpos: np.ndarray,
    ) -> np.ndarray:
        if robot_qpos.shape[0] == cat_position.shape[0]:
            return robot_qpos.astype(np.float32, copy=True)
        if (
            group.gripper_index is not None
            and robot_qpos.shape[0] == group.dof
            and cat_position.shape[0] == group.dof - 1
        ):
            return np.delete(robot_qpos, group.gripper_index).astype(np.float32)
        raise ValueError(
            f"HIL relay shape mismatch for {group.name}: "
            f"cat={cat_position.shape[0]} robot={robot_qpos.shape[0]}"
        )

    def _hil_anchor_state_for_message(
        self,
        group: Any,
        cat_position: np.ndarray,
        robot_qpos: np.ndarray,
    ) -> np.ndarray:
        last_command = self._last_group_joint_commands.get(group.name)
        if last_command is None:
            return self._hil_robot_state_for_message(group, cat_position, robot_qpos)
        command = np.asarray(last_command, dtype=np.float32)
        if command.shape[0] == cat_position.shape[0]:
            return command.copy()
        if group.gripper_index is None:
            return self._hil_robot_state_for_message(group, cat_position, robot_qpos)
        if command.shape[0] == group.dof and cat_position.shape[0] == group.dof - 1:
            return np.delete(command, group.gripper_index).astype(np.float32)
        if command.shape[0] == group.dof - 1 and cat_position.shape[0] == group.dof:
            anchor = self._hil_robot_state_for_message(group, cat_position, robot_qpos)
            anchor[: group.gripper_index] = command[: group.gripper_index]
            anchor[group.gripper_index + 1 :] = command[group.gripper_index :]
            return anchor.astype(np.float32)
        return self._hil_robot_state_for_message(group, cat_position, robot_qpos)

    def _resolve_hil_joint_command(
        self,
        group: Any,
        cat_position: np.ndarray,
        robot_qpos: np.ndarray,
    ) -> np.ndarray:
        cat = np.asarray(cat_position, dtype=np.float32)
        if self._hil_control_mode == "absolute":
            return cat.copy()
        robot = self._hil_anchor_state_for_message(group, cat, robot_qpos)
        if group.name not in self._hil_cat_anchors:
            self._hil_cat_anchors[group.name] = cat.copy()
            self._hil_robot_anchors[group.name] = robot.copy()
        command = self._hil_robot_anchors[group.name] + (cat - self._hil_cat_anchors[group.name])
        if group.gripper_index is not None and cat.shape[0] == group.dof:
            command[group.gripper_index] = cat[group.gripper_index]
        return command.astype(np.float32)

    def _handle_hil_joint_msg(self, group: Any, msg: Any) -> None:
        if not self._hil_relay_enabled:
            return
        robot_qpos = self._latest_group_qpos(group)
        if robot_qpos is None:
            logger.error("HIL relay cannot publish %s without joint feedback", group.name)
            return
        try:
            command = self._resolve_hil_joint_command(
                group,
                np.asarray(msg.position, dtype=np.float32),
                robot_qpos,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return
        publisher = self._group_cmd_publishers.get(group.name)
        if publisher is None:
            logger.error("HIL relay has no command publisher for %s", group.name)
            return
        stamp = self._node.get_clock().now().to_msg()
        publisher.publish(self._make_joint_state(group.name, command, stamp))
        self._last_group_joint_commands[group.name] = command.copy()

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

    def _pop_synced_msg(self, deque: collections.deque, frame_time: float) -> Any | None:
        while deque and _stamp_to_sec(deque[0]) < frame_time:
            deque.popleft()
        if not deque:
            return None
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
                self._last_group_joint_commands[group.name] = np.asarray(
                    joints,
                    dtype=np.float32,
                ).copy()
            gripper_publisher = self._group_gripper_publishers.get(group.name)
            if gripper_publisher is not None and group.gripper_index is not None:
                gripper_position = np.asarray([part[group.gripper_index]], dtype=np.float32)
                gripper_publisher.publish(
                    self._make_joint_state(group.name, gripper_position, stamp, gripper_only=True)
                )
                self._last_group_gripper_commands[group.name] = float(gripper_position[0])

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
