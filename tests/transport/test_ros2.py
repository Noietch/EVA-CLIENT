"""Tests for ROS2 transport helpers."""

from __future__ import annotations

import sys
import threading
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest

import robots  # noqa: F401  (registers robots)
import transport.ros2 as ros2
import transport.utils as transport_utils
from core.config import load_config
from core.registry import ROBOT_REGISTRY
from core.types import CollectionRawImage, Observation

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_R1LITE_COLLECTION = _CONFIGS_DIR / "02_collection" / "r1lite.py"


class _FakeQoSProfile:
    def __init__(self, depth, reliability, durability):
        self.depth = depth
        self.reliability = reliability
        self.durability = durability


class _FakeNode:
    def __init__(self):
        self.subscriptions = []
        self.publishers = []
        self.publishers_by_topic = {}
        self.clock_stamp = types.SimpleNamespace(sec=0, nanosec=0)

    def get_clock(self):
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(
                to_msg=lambda: types.SimpleNamespace(
                    sec=self.clock_stamp.sec,
                    nanosec=self.clock_stamp.nanosec,
                )
            )
        )

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))

    def create_publisher(self, msg_type, topic, qos):
        publisher = _FakePublisher()
        self.publishers.append((msg_type, topic, qos))
        self.publishers_by_topic[topic] = publisher
        return publisher


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)

    def destroy(self):
        pass


class _FakeJointState:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None)
        self.name = []
        self.position = []


def _stamped_msg(sec: int, position=None, nanosec: int = 0):
    stamp = types.SimpleNamespace(sec=sec, nanosec=nanosec)
    if position is None:
        position = [0.0]
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp), position=position)


def _compressed_msg(sec: int, data: bytes):
    msg = _stamped_msg(sec)
    msg.data = data
    return msg


def _subscription_callback(node: _FakeNode, topic: str):
    for _msg_type, sub_topic, callback, _qos in node.subscriptions:
        if sub_topic == topic:
            return callback
    raise AssertionError(f"missing subscription for {topic}")


def test_ros2_live_qos_matches_best_effort_publishers(monkeypatch):
    fake_qos = types.ModuleType("rclpy.qos")
    fake_qos.QoSProfile = _FakeQoSProfile  # type: ignore[reportAttributeAccessIssue]
    fake_qos.QoSReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="best_effort")  # type: ignore[reportAttributeAccessIssue]
    fake_qos.QoSDurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile")  # type: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "rclpy.qos", fake_qos)

    assert hasattr(ros2, "_make_live_qos")

    qos = ros2._make_live_qos()

    assert qos.reliability == "best_effort"
    assert qos.durability == "volatile"
    assert qos.depth == 10


def test_ros2_transport_uses_live_qos_for_entities(monkeypatch):
    node = _FakeNode()
    qos = object()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: qos)

    config = load_config(_CONFIGS_DIR / "01_deploy" / "r1lite" / "openpi_qpos.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)

    ros2.Ros2Transport(config, robot)

    assert node.subscriptions
    assert node.publishers
    assert {call[-1] for call in node.subscriptions} == {qos}
    assert {call[-1] for call in node.publishers} == {qos}


def test_ros2_camera_subscriptions_use_deeper_qos(monkeypatch):
    node = _FakeNode()
    qoses = {}
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)

    def make_qos(depth=10):
        qos = object()
        qoses[qos] = depth
        return qos

    monkeypatch.setattr(ros2, "_make_live_qos", make_qos)

    config = load_config(_CONFIGS_DIR / "01_deploy" / "r1lite" / "openpi_qpos.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    camera_topics = set(transport._camera_topics.values())

    assert camera_topics
    seen_camera_topics = set()
    for _msg_type, topic, _callback, qos in node.subscriptions:
        if topic in camera_topics:
            seen_camera_topics.add(topic)
            assert qoses[qos] == ros2._CAMERA_QOS_DEPTH
    assert seen_camera_topics == camera_topics


def test_ros2_camera_subscription_keeps_raw_capture_deque_independent(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    camera = robot.observation_schema.cameras[0]
    callback = _subscription_callback(node, transport._camera_topics[camera.name])

    callback(_compressed_msg(1, b"before collection"))

    assert len(transport._camera_deques[camera.name]) == 1
    assert len(transport._collection_camera_deques[camera.name]) == 0

    transport.start_collection()
    callback(_compressed_msg(2, b"during collection"))

    assert len(transport._camera_deques[camera.name]) == 2
    assert len(transport._collection_camera_deques[camera.name]) == 1
    transport._camera_deques[camera.name].popleft()
    assert len(transport._collection_camera_deques[camera.name]) == 1

    transport.stop_collection()

    assert len(transport._collection_camera_deques[camera.name]) == 0


def test_ros2_collection_raw_releases_consumed_camera_messages(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    camera = robot.observation_schema.cameras[0]
    callback = _subscription_callback(node, transport._camera_topics[camera.name])
    transport.start_collection()
    callback(_compressed_msg(1, b"one"))
    callback(_compressed_msg(2, b"two"))
    callback(_compressed_msg(3, b"three"))

    snapshot = transport.acquire_collection_raw()

    assert snapshot is not None
    assert len(transport._collection_camera_deques[camera.name]) == 1
    assert transport._collection_camera_deques[camera.name][0].data == b"three"


def test_ros2_hil_relay_switch_keeps_collection_camera_capture_active(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    camera = robot.observation_schema.cameras[0]
    callback = _subscription_callback(node, transport._camera_topics[camera.name])
    transport.start_collection()

    transport.set_hil_relay_enabled(True)
    callback(_compressed_msg(1, b"hil"))
    transport.set_hil_relay_enabled(False)
    callback(_compressed_msg(2, b"policy"))

    assert [msg.data for msg in transport._collection_camera_deques[camera.name]] == [
        b"hil",
        b"policy",
    ]


def test_ros2_camera_subscriptions_report_minimum_image_hz(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())
    now = [0.0]
    monkeypatch.setattr(transport_utils.time, "monotonic", lambda: now[0])

    config = load_config(_CONFIGS_DIR / "01_deploy" / "r1lite" / "openpi_qpos.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)

    topics = tuple(transport._camera_topics.values())
    for topic in topics:
        now[0] = 0.0
        _subscription_callback(node, topic)(_stamped_msg(1))
    for topic, t in zip(topics, (0.05, 0.10, 0.20), strict=True):
        now[0] = t
        _subscription_callback(node, topic)(_stamped_msg(2))

    assert transport.image_min_hz() == pytest.approx(5.0)


def test_ros2_hil_relay_subscriptions_use_depth_one_qos(monkeypatch):
    node = _FakeNode()
    qoses = {}
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)

    def make_qos(depth=10):
        qos = object()
        qoses[qos] = depth
        return qos

    monkeypatch.setattr(ros2, "_make_live_qos", make_qos)

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    ros2.Ros2Transport(config, robot)

    hil_qos_depths = {
        topic: qoses[qos]
        for _msg_type, topic, _callback, qos in node.subscriptions
        if topic.startswith("/eva/hil/input_joint_state_arm_")
    }
    assert hil_qos_depths == {
        "/eva/hil/input_joint_state_arm_left": 1,
        "/eva/hil/input_joint_state_arm_right": 1,
    }


def test_ros2_hil_relay_is_disabled_until_enabled(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [10, 20, 30, 40, 50, 60]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))

    published = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    assert published == []


def test_ros2_hil_relay_publishes_after_enable(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_control_mode("absolute")
    transport.set_hil_relay_enabled(True)
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [10, 20, 30, 40, 50, 60]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))

    published = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    np.testing.assert_allclose(published[0].position, [1, 2, 3, 4, 5, 6])


def test_ros2_hil_relative_relay_anchors_slave_to_first_cat_frame(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_relay_enabled(True)
    transport.set_hil_control_mode("relative")
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [10, 20, 30, 40, 50, 60]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))
    callback(_stamped_msg(3, [1.5, 1, 3, 4, 5, 8]))

    published = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    np.testing.assert_allclose(published[0].position, [10, 20, 30, 40, 50, 60])
    np.testing.assert_allclose(published[1].position, [10.5, 19, 30, 40, 50, 62])


def test_ros2_hil_relay_does_not_write_collection_action_qpos(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_relay_enabled(True)
    transport.set_hil_control_mode("relative")
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [10, 20, 30, 40, 50, 60]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    node.clock_stamp = types.SimpleNamespace(sec=20, nanosec=0)
    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))
    node.clock_stamp = types.SimpleNamespace(sec=30, nanosec=0)
    callback(_stamped_msg(3, [2, 4, 3, 4, 5, 8]))

    samples = list(transport._collection_action_qpos_deques["left_arm"])
    assert samples == []


def test_ros2_hil_relay_restamps_commands_with_local_clock(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_relay_enabled(True)
    transport.set_hil_control_mode("relative")
    transport._group_state_deques["left_arm"].append(
        _stamped_msg(100, [10, 20, 30, 40, 50, 60])
    )
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    node.clock_stamp = types.SimpleNamespace(sec=100, nanosec=123)
    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))
    node.clock_stamp = types.SimpleNamespace(sec=101, nanosec=456)
    callback(_stamped_msg(3, [2, 4, 3, 4, 5, 8]))

    published = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    samples = list(transport._collection_action_qpos_deques["left_arm"])
    assert [(m.header.stamp.sec, m.header.stamp.nanosec) for m in published] == [
        (100, 123),
        (101, 456),
    ]
    assert samples == []


def test_ros2_collection_action_qpos_subscribes_motion_target_when_hil_enabled(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    action_callback = _subscription_callback(
        node, "/motion_target/target_joint_state_arm_left"
    )

    action_callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))

    samples = list(transport._collection_action_qpos_deques["left_arm"])
    assert len(samples) == 1
    np.testing.assert_allclose(samples[0].position, [1, 2, 3, 4, 5, 6])


def test_ros2_subscribes_hil_topic_when_configured(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    ros2.Ros2Transport(config, robot)

    subscribed_topics = {topic for _msg_type, topic, _callback, _qos in node.subscriptions}
    assert "/eva/hil/input_joint_state_arm_left" in subscribed_topics
    assert "/eva/hil/input_joint_state_arm_right" in subscribed_topics
    assert "/motion_target/target_joint_state_arm_left" in subscribed_topics
    assert "/motion_target/target_joint_state_arm_right" in subscribed_topics


def test_ros2_hil_relative_relay_anchors_to_last_command_target(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_relay_enabled(True)
    transport.set_hil_control_mode("relative")
    transport.publish_action(
        np.array(
            [
                12,
                21,
                30,
                40,
                50,
                60,
                100,
                -10,
                -20,
                -30,
                -40,
                -50,
                -60,
                0,
            ],
            dtype=np.float32,
        )
    )
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [10, 20, 30, 40, 50, 60]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))

    published = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    np.testing.assert_allclose(published[-1].position, [12, 21, 30, 40, 50, 60])


def test_ros2_hil_relay_does_not_publish_gripper_target(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_relay_enabled(True)
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [10, 20, 30, 40, 50, 60]))
    transport._group_gripper_deques["left_arm"].append(_stamped_msg(1, [77]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")

    callback(_stamped_msg(2, [1, 2, 3, 4, 5, 6]))

    published = node.publishers_by_topic["/motion_target/target_position_gripper_left"].messages
    assert published == []


def test_ros2_hil_frame_uses_last_gripper_command_when_collection_frame_missing(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.get_collection_frame = lambda: None
    transport.get_frame = lambda: Observation(
        images={}, state_qpos=np.zeros(robot.total_action_dim, dtype=np.float32), timestamp=1.0
    )
    transport._last_group_joint_commands = {
        "left_arm": np.zeros(6, dtype=np.float32),
        "right_arm": np.zeros(6, dtype=np.float32),
    }
    transport._last_group_gripper_commands = {"left_arm": 0.0, "right_arm": 100.0}
    transport._latest_group_qpos = lambda group: np.full(group.dof, 100.0, dtype=np.float32)

    frame = transport.get_hil_frame()

    assert frame is not None
    assert frame.action_qpos is not None
    np.testing.assert_allclose(frame.action_qpos[[6, 13]], [0.0, 100.0])


def test_ros2_hil_reorders_named_full_group_and_publishes_split_gripper(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport.set_hil_control_mode("absolute")
    transport.set_hil_relay_enabled(True)
    transport._group_state_deques["left_arm"].append(_stamped_msg(1, [0, 0, 0, 0, 0, 0]))
    transport._group_gripper_deques["left_arm"].append(_stamped_msg(1, [100]))
    callback = _subscription_callback(node, "/eva/hil/input_joint_state_arm_left")
    message = _stamped_msg(2, [7, 6, 5, 4, 3, 2, 1])
    message.name = list(reversed(robot.actuator_groups[0].joint_names))

    callback(message)

    arm = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    gripper = node.publishers_by_topic["/motion_target/target_position_gripper_left"].messages
    np.testing.assert_allclose(arm[-1].position, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(gripper[-1].position, [7])


def test_ros2_hil_relative_relay_keeps_gripper_absolute(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    group = robot.actuator_groups[0]
    transport.set_hil_control_mode("relative")

    first = transport._resolve_hil_joint_command(
        group,
        np.array([1, 2, 3, 4, 5, 6, 10], dtype=np.float32),
        np.array([10, 20, 30, 40, 50, 60, 100], dtype=np.float32),
    )
    second = transport._resolve_hil_joint_command(
        group,
        np.array([2, 1, 3, 4, 7, 6, 0], dtype=np.float32),
        np.array([90, 90, 90, 90, 90, 90, 90], dtype=np.float32),
    )

    np.testing.assert_allclose(first, [10, 20, 30, 40, 50, 60, 10])
    np.testing.assert_allclose(second, [11, 19, 30, 40, 52, 60, 0])


def test_ros2_hil_mode_switch_resets_relative_anchor(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    group = robot.actuator_groups[0]
    transport.set_hil_control_mode("relative")
    transport._resolve_hil_joint_command(
        group,
        np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
    )

    transport.set_hil_control_mode("absolute")
    transport.set_hil_control_mode("relative")
    resolved = transport._resolve_hil_joint_command(
        group,
        np.array([9, 9, 9, 9, 9, 9], dtype=np.float32),
        np.array([100, 101, 102, 103, 104, 105], dtype=np.float32),
    )

    np.testing.assert_allclose(resolved, [100, 101, 102, 103, 104, 105])


def test_pop_synced_msg_returns_none_when_queue_drains_before_frame_time():
    messages = deque([_stamped_msg(1)])

    assert ros2.Ros2Transport._pop_synced_msg(None, messages, 2.0) is None
    assert not messages


def test_ros2_get_camera_frame_reads_latest_without_consuming(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(transport, "_decode_image_msg", lambda camera_name, msg: image)
    transport._camera_deques["head"].append(_stamped_msg(1))

    frame = transport.get_camera_frame("cam_high")

    np.testing.assert_array_equal(frame, image)
    assert len(transport._camera_deques["head"]) == 1


def test_ros2_get_camera_jpeg_reads_compressed_payload_without_consuming(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    payload = b"\xff\xd8raw-jpeg\xff\xd9"
    transport._camera_deques["head"].append(_compressed_msg(1, payload))

    jpeg = transport.get_camera_jpeg("cam_high")

    assert jpeg == payload
    assert len(transport._camera_deques["head"]) == 1


def test_ros2_collection_joint_vector_merges_configured_gripper_topics(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._collection_qpos_deques["left_arm"].append(_stamped_msg(1, [0, 1, 2, 3, 4, 5]))
    transport._collection_qpos_deques["right_arm"].append(
        _stamped_msg(1, [10, 11, 12, 13, 14, 15])
    )
    transport._collection_qpos_gripper_deques["left_arm"].append(_stamped_msg(1, [42]))
    transport._collection_qpos_gripper_deques["right_arm"].append(_stamped_msg(1, [43]))

    qpos = transport._collection_joint_vector(
        transport._collection_qpos_deques,
        1.0,
        gripper_deques_by_group=transport._collection_qpos_gripper_deques,
    )

    np.testing.assert_array_equal(
        qpos,
        np.array([0, 1, 2, 3, 4, 5, 42, 10, 11, 12, 13, 14, 15, 43], dtype=np.float32),
    )


def test_ros2_collection_action_gripper_snaps_to_robot_open_close(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._collection_action_qpos_deques["left_arm"].append(
        _stamped_msg(1, [0, 1, 2, 3, 4, 5])
    )
    transport._collection_action_qpos_deques["right_arm"].append(
        _stamped_msg(1, [10, 11, 12, 13, 14, 15])
    )
    transport._collection_action_qpos_gripper_deques["left_arm"].append(_stamped_msg(1, [1]))
    transport._collection_action_qpos_gripper_deques["right_arm"].append(_stamped_msg(1, [0]))

    action = transport._collection_joint_vector(
        transport._collection_action_qpos_deques,
        1.0,
        gripper_deques_by_group=transport._collection_action_qpos_gripper_deques,
        action_gripper=True,
    )

    np.testing.assert_array_equal(
        action,
        np.array([0, 1, 2, 3, 4, 5, 100, 10, 11, 12, 13, 14, 15, 0], dtype=np.float32),
    )


def test_ros2_collection_raw_batch_merges_gripper_topics(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    monkeypatch.setattr(
        transport,
        "_decode_image_msg",
        lambda _camera_name, _msg: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    for camera in transport._collection_camera_specs():
        transport._collection_camera_deques[camera.name].append(_compressed_msg(1, b"jpeg"))
    transport._collection_qpos_deques["left_arm"].append(_stamped_msg(1, [0, 1, 2, 3, 4, 5]))
    transport._collection_qpos_deques["right_arm"].append(
        _stamped_msg(1, [10, 11, 12, 13, 14, 15])
    )
    transport._collection_qpos_gripper_deques["left_arm"].append(_stamped_msg(1, [42]))
    transport._collection_qpos_gripper_deques["right_arm"].append(_stamped_msg(1, [43]))

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    batch = snapshot.decode_raw()

    np.testing.assert_array_equal(
        batch.vectors["state_qpos:left_arm"][0].value,
        np.array([0, 1, 2, 3, 4, 5, 42], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.vectors["state_qpos:right_arm"][0].value,
        np.array([10, 11, 12, 13, 14, 15, 43], dtype=np.float32),
    )


def test_ros2_collection_snapshot_keeps_compressed_bytes_not_ros_image_messages(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    camera = transport._collection_camera_specs()[0]
    message = _compressed_msg(1, b"jpeg payload")
    transport._collection_camera_deques[camera.name].append(message)

    snapshot = transport.acquire_collection_raw()

    assert snapshot is not None
    batch = snapshot.decode_raw()
    image = batch.images[camera.observation_key][0].value
    assert isinstance(image, CollectionRawImage)
    assert image.decoder.__closure__ is None
    assert image.decoder.__defaults__ == (b"jpeg payload",)


def test_ros2_live_gripper_consumer_holds_collection_deque_lock(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    messages = transport._group_gripper_deques["left_arm"]
    messages.extend([_stamped_msg(1), _stamped_msg(2), _stamped_msg(3)])
    started = threading.Event()
    finished = threading.Event()

    def consume_live_gripper() -> None:
        started.set()
        transport._pop_latest_before(messages, 3.0)
        finished.set()

    thread = threading.Thread(target=consume_live_gripper)
    with transport._deque_guard():
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)
    thread.join(timeout=1.0)

    assert finished.is_set()


def test_ros2_collection_operator_gripper_action_seed_is_latched(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._group_gripper_deques["left_arm"].append(_stamped_msg(1, [100]))
    transport._group_gripper_deques["right_arm"].append(_stamped_msg(1, [0]))
    transport.start_collection()
    transport._collection_action_qpos_deques["left_arm"].append(
        _stamped_msg(10, [0, 1, 2, 3, 4, 5])
    )
    transport._collection_action_qpos_deques["right_arm"].append(
        _stamped_msg(10, [10, 11, 12, 13, 14, 15])
    )

    action = transport._collection_joint_vector(transport._collection_action_qpos_deques, 10.0)

    np.testing.assert_array_equal(
        action,
        np.array([0, 1, 2, 3, 4, 5, 100, 10, 11, 12, 13, 14, 15, 0], dtype=np.float32),
    )


def test_ros2_clear_collection_backlog_seeds_operator_gripper_action(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._group_gripper_deques["left_arm"].append(_stamped_msg(1, [100]))
    transport._group_gripper_deques["right_arm"].append(_stamped_msg(1, [0]))

    transport.clear_collection_backlog()

    assert len(transport._collection_action_qpos_gripper_deques["left_arm"]) == 1
    assert len(transport._collection_action_qpos_gripper_deques["right_arm"]) == 1


def test_ros2_clear_collection_backlog_does_not_synthesize_action_qpos(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)

    transport.clear_collection_backlog()

    assert list(transport._collection_action_qpos_deques["left_arm"]) == []
    assert list(transport._collection_action_qpos_deques["right_arm"]) == []


def test_ros2_collection_raw_requires_motion_target_action_after_clear(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)

    transport.clear_collection_backlog()

    assert len(transport._collection_action_qpos_deques["left_arm"]) == 0
    assert len(transport._collection_action_qpos_deques["right_arm"]) == 0

    left = _stamped_msg(2, [0, 1, 2, 3, 4, 5])
    right = _stamped_msg(2, [10, 11, 12, 13, 14, 15])
    transport._collection_qpos_deques["left_arm"].append(left)
    transport._collection_qpos_deques["right_arm"].append(right)

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    batch = snapshot.decode_raw()

    assert "action_qpos:left_arm" not in batch.vectors
    assert "action_qpos:right_arm" not in batch.vectors


def test_ros2_collection_operator_gripper_action_records_new_latched_target(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._group_gripper_deques["left_arm"].append(_stamped_msg(1, [100]))
    transport._group_gripper_deques["right_arm"].append(_stamped_msg(1, [0]))
    transport.start_collection()
    transport.record_collection_gripper_action(
        "left_arm",
        0.0,
        stamp=_stamped_msg(5).header.stamp,
    )
    transport._collection_action_qpos_deques["left_arm"].append(
        _stamped_msg(10, [0, 1, 2, 3, 4, 5])
    )
    transport._collection_action_qpos_deques["right_arm"].append(
        _stamped_msg(10, [10, 11, 12, 13, 14, 15])
    )

    action = transport._collection_joint_vector(transport._collection_action_qpos_deques, 10.0)

    np.testing.assert_array_equal(
        action,
        np.array([0, 1, 2, 3, 4, 5, 0, 10, 11, 12, 13, 14, 15, 0], dtype=np.float32),
    )


def test_ros2_collection_raw_action_qpos_is_complete_from_seeded_gripper(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    monkeypatch.setattr(
        transport,
        "_decode_image_msg",
        lambda _camera_name, _msg: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    transport._group_gripper_deques["left_arm"].append(_stamped_msg(1, [100]))
    transport._group_gripper_deques["right_arm"].append(_stamped_msg(1, [0]))
    transport.clear_collection_backlog()
    for camera in transport._collection_camera_specs():
        transport._collection_camera_deques[camera.name].append(_compressed_msg(2, b"jpeg"))
    transport._collection_qpos_deques["left_arm"].append(_stamped_msg(2, [0, 1, 2, 3, 4, 5]))
    transport._collection_qpos_deques["right_arm"].append(
        _stamped_msg(2, [10, 11, 12, 13, 14, 15])
    )
    transport._collection_action_qpos_deques["left_arm"].append(
        _stamped_msg(2, [20, 21, 22, 23, 24, 25])
    )
    transport._collection_action_qpos_deques["right_arm"].append(
        _stamped_msg(2, [30, 31, 32, 33, 34, 35])
    )

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    batch = snapshot.decode_raw()

    np.testing.assert_array_equal(
        batch.vectors["action_qpos:left_arm"][0].value,
        np.array([20, 21, 22, 23, 24, 25, 100], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.vectors["action_qpos:right_arm"][0].value,
        np.array([30, 31, 32, 33, 34, 35, 0], dtype=np.float32),
    )


def test_ros2_collection_diagnostics_reports_missing_action_gripper(monkeypatch):
    node = _FakeNode()
    runtime = types.SimpleNamespace(
        node=node,
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    config = load_config(_R1LITE_COLLECTION)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ros2.Ros2Transport(config, robot)
    transport._collection_action_qpos_deques["left_arm"].append(
        _stamped_msg(1, [0, 1, 2, 3, 4, 5])
    )
    transport._collection_action_qpos_deques["right_arm"].append(
        _stamped_msg(1, [10, 11, 12, 13, 14, 15])
    )
    transport._collection_qpos_gripper_deques["left_arm"].append(_stamped_msg(1, [42]))
    transport._collection_action_qpos_gripper_deques["right_arm"].append(_stamped_msg(1, [0]))

    diagnostics = transport.collection_diagnostics()

    assert "action_qpos_gripper:left_arm" in diagnostics
    assert "source=operator" in diagnostics
    assert "no messages" in diagnostics
