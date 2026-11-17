"""Tests for ROS2 transport helpers."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import robots  # noqa: F401  (registers robots)
import transport.ros2 as ros2
from core.config import load_config
from core.registry import ROBOT_REGISTRY

pytestmark = pytest.mark.unit

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_R1LITE_COLLECTION = _CONFIGS_DIR / "02_collection" / "r1lite.py"


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
    callback = next(
        callback
        for _, topic, callback, _ in node.subscriptions
        if topic == "/eva/hil/input_joint_state_arm_left"
    )
    message = _stamped_msg(2, [7, 6, 5, 4, 3, 2, 1])
    message.name = list(reversed(robot.actuator_groups[0].joint_names))

    callback(message)

    arm = node.publishers_by_topic["/motion_target/target_joint_state_arm_left"].messages
    gripper = node.publishers_by_topic["/motion_target/target_position_gripper_left"].messages
    np.testing.assert_allclose(arm[-1].position, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(gripper[-1].position, [7])


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
