"""Tests for build_transport's no-ROS degradation."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import robots  # noqa: F401  (registers robots)
import transport.base as base
from core.config import load_config
from core.registry import ROBOT_REGISTRY
from transport.base import OfflineTransport, build_transport

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _build(config_relpath: str):
    config = load_config(_CONFIGS_DIR / config_relpath)
    robot = ROBOT_REGISTRY.build(config.robot.type)
    return build_transport(config, robot)


@pytest.mark.parametrize(
    "config_relpath",
    [
        # ros1 config leaves transport.topics empty; the ros1 backend validates
        # topics before importing rospy, so degradation must trigger before that.
        "01_deploy/dual_agilex_piper/openpi_qpos.py",
        # ros2 config with populated topics.
        "01_deploy/r1lite/openpi_qpos.py",
    ],
)
def test_ros_backend_degrades_to_offline_without_ros(monkeypatch, config_relpath):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: None)
    transport = _build(config_relpath)
    assert isinstance(transport, OfflineTransport)
    assert transport.is_offline()


def test_non_ros_backend_is_unaffected_by_ros_absence():
    # zmq has no ROS dependency, so it must build normally regardless of ROS.
    transport = _build("01_deploy/ur5e/openpi_qpos.py")
    assert not isinstance(transport, OfflineTransport)
    transport.close()


def test_ros_backend_builds_when_runtime_present(monkeypatch):
    # rclpy importable -> the pre-check passes and the real backend is constructed
    # (its runtime is stubbed so no actual ROS is needed).
    import transport.ros2 as ros2

    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: object())
    runtime = _fake_ros2_runtime()
    monkeypatch.setattr(ros2, "get_ros2_runtime", lambda node_name: runtime)
    monkeypatch.setattr(ros2, "_make_live_qos", lambda depth=10: object())

    transport = _build("01_deploy/r1lite/openpi_qpos.py")
    assert isinstance(transport, ros2.Ros2Transport)


def _fake_ros2_runtime():
    class _FakeNode:
        def create_subscription(self, *args, **kwargs):
            pass

        def create_publisher(self, *args, **kwargs):
            return types.SimpleNamespace(destroy=lambda: None)

    return types.SimpleNamespace(
        node=_FakeNode(),
        cv_bridge=object(),
        image_type=object,
        compressed_image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
