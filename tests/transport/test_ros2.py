"""Tests for ROS2 transport helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import robots  # noqa: F401  (registers robots)
import transport.ros2 as ros2
from core.config import load_config
from core.registry import ROBOT_REGISTRY

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class _FakeQoSProfile:
    def __init__(self, depth, reliability, durability):
        self.depth = depth
        self.reliability = reliability
        self.durability = durability


class _FakeNode:
    def __init__(self):
        self.subscriptions = []
        self.publishers = []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))

    def create_publisher(self, msg_type, topic, qos):
        self.publishers.append((msg_type, topic, qos))
        return _FakePublisher()


class _FakePublisher:
    def destroy(self):
        pass


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
