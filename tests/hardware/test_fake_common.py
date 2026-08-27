from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from core.config import load_config
from core.registry import ROBOT_REGISTRY
from examples.hardware.fake_common import FakeRobotNode, build_arg_parser
from examples.hardware.x5.fake_node import X5FakeRobotNode
from transport.zmq import ZmqTransport


def _node(robot_name: str = "arx_x5", *, dynamics_mode: str = "second-order") -> FakeRobotNode:
    suffix = uuid.uuid4().hex
    return FakeRobotNode(
        robot_name=robot_name,
        observation_endpoint=f"inproc://fake-observation-{suffix}",
        action_endpoint=f"inproc://fake-action-{suffix}",
        publish_rate_hz=60.0,
        image_height=12,
        image_width=16,
        dynamics_rate_hz=200.0,
        natural_frequency_hz=2.0,
        damping_ratio=1.0,
        dynamics_mode=dynamics_mode,
    )


def test_second_order_plant_moves_gradually_and_converges() -> None:
    node = _node()
    try:
        initial = node.qpos
        target = initial.copy()
        target[0] += 0.5

        assert node.set_target_qpos(target)
        np.testing.assert_allclose(node.qpos, initial)

        node.step_dynamics()
        assert initial[0] < node.qpos[0] < target[0]
        for _ in range(499):
            node.step_dynamics()

        np.testing.assert_allclose(node.qpos, target, atol=1e-4)
        np.testing.assert_allclose(node.qvel, 0.0, atol=1e-3)
    finally:
        node.close()


def test_every_qpos_uses_the_same_independent_dynamics() -> None:
    node = _node("agibot_g2")
    try:
        delta = np.linspace(-0.4, 0.4, node.qpos.size, dtype=np.float32)
        target = node.qpos + delta
        assert node.set_target_qpos(target)

        for _ in range(500):
            node.step_dynamics()

        np.testing.assert_allclose(node.qpos, target, atol=1e-4)
    finally:
        node.close()


def test_direct_mode_copies_action_to_state_and_clears_velocity() -> None:
    node = _node(dynamics_mode="direct")
    try:
        target = node.qpos + 0.5

        assert node.set_target_qpos(target)

        np.testing.assert_allclose(node.target_qpos, target)
        np.testing.assert_allclose(node.qpos, target)
        np.testing.assert_allclose(node.qvel, 0.0)
        node.step_dynamics()
        np.testing.assert_allclose(node.qpos, target)
    finally:
        node.close()


def test_dynamics_mode_cli_defaults_to_second_order_and_accepts_direct() -> None:
    parser = build_arg_parser("arx_x5")

    assert parser.parse_args([]).dynamics_mode == "second-order"
    assert parser.parse_args(["--dynamics-mode", "direct"]).dynamics_mode == "direct"


def test_reset_restores_position_velocity_and_target() -> None:
    node = _node()
    try:
        initial = node.qpos
        target = initial + 0.5
        node.set_target_qpos(target)
        for _ in range(20):
            node.step_dynamics()
        assert not np.allclose(node.qpos, initial)
        assert not np.allclose(node.qvel, 0.0)

        node.reset()

        np.testing.assert_allclose(node.qpos, initial)
        np.testing.assert_allclose(node.qvel, 0.0)
        np.testing.assert_allclose(node.target_qpos, initial)
    finally:
        node.close()


@pytest.mark.parametrize("invalid", [np.zeros(13), np.full(14, np.nan)])
def test_invalid_action_does_not_change_target(invalid: np.ndarray) -> None:
    node = _node()
    try:
        before = node.target_qpos
        assert not node.set_target_qpos(invalid)
        np.testing.assert_allclose(node.target_qpos, before)
    finally:
        node.close()


def test_camera_frames_are_distinct_and_periodic() -> None:
    node = _node()
    try:
        first = node._read_images(node._started_at)
        later = node._read_images(node._started_at + 1.0)

        assert set(first) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        assert all(image.shape == (12, 16, 3) for image in first.values())
        assert all(image.dtype == np.uint8 for image in first.values())
        assert not np.array_equal(first["cam_high"], first["cam_left_wrist"])
        assert not np.array_equal(first["cam_high"], later["cam_high"])
    finally:
        node.close()


@pytest.mark.parametrize(
    "robot_name",
    ["agibot_g2", "agilex_piper", "arx_r5", "arx_x5", "dual_franka", "ur5e"],
)
def test_fake_node_uses_registered_robot_dimensions(robot_name: str) -> None:
    node = _node(robot_name)
    try:
        robot = ROBOT_REGISTRY.build(robot_name)
        assert node.qpos.shape == (robot.total_action_dim,)
        assert set(node._split_by_group(node.qpos)) == {
            group.name for group in robot.actuator_groups
        }
        assert set(node._read_images(node._started_at)) == {
            camera.observation_key for camera in robot.observation_schema.cameras
        }
    finally:
        node.close()


def test_x5_zmq_action_drives_gradual_state_and_collection_does_not_control_plant() -> None:
    suffix = uuid.uuid4().hex
    observation_endpoint = f"inproc://x5-observation-{suffix}"
    action_endpoint = f"inproc://x5-action-{suffix}"
    node = FakeRobotNode(
        robot_name="arx_x5",
        observation_endpoint=observation_endpoint,
        action_endpoint=action_endpoint,
        publish_rate_hz=60.0,
        image_height=12,
        image_width=16,
        dynamics_rate_hz=200.0,
        natural_frequency_hz=2.0,
        damping_ratio=1.0,
    )
    thread = threading.Thread(target=node.serve_forever, daemon=True)
    thread.start()

    config = load_config(Path("configs/02_collection/arx_x5_vr.py"))
    config.transport.sub_endpoint = observation_endpoint
    config.transport.pub_endpoint = action_endpoint
    robot = ROBOT_REGISTRY.build("arx_x5")
    transport = ZmqTransport(config, robot)
    try:
        deadline = time.monotonic() + 2.0
        state = transport.get_latest_qpos()
        while state is None and time.monotonic() < deadline:
            time.sleep(0.02)
            state = transport.get_latest_qpos()
        assert state is not None

        target = np.asarray(robot.initial_qpos, dtype=np.float32).copy()
        target[0] += 0.4
        target[8] -= 0.3
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not np.allclose(node.target_qpos, target):
            transport.publish_action(target)
            time.sleep(0.02)
        np.testing.assert_allclose(node.target_qpos, target)

        intermediate = transport.get_latest_qpos()
        assert intermediate is not None
        assert not np.allclose(intermediate, target, atol=1e-3)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = transport.get_latest_qpos()
            if state is not None and np.allclose(state, target, atol=3e-3):
                break
            time.sleep(0.02)
        assert state is not None
        np.testing.assert_allclose(state, target, atol=3e-3)

        target_before_collection = node.target_qpos
        transport.start_collection()
        time.sleep(0.1)
        np.testing.assert_allclose(node.target_qpos, target_before_collection)
        assert not node.is_stopped
    finally:
        transport.close()
        node.stop()
        thread.join(timeout=2.0)
        node.close()


def test_x5_direct_mode_returns_action_as_observed_state() -> None:
    suffix = uuid.uuid4().hex
    observation_endpoint = f"inproc://x5-direct-observation-{suffix}"
    action_endpoint = f"inproc://x5-direct-action-{suffix}"
    node = FakeRobotNode(
        robot_name="arx_x5",
        observation_endpoint=observation_endpoint,
        action_endpoint=action_endpoint,
        publish_rate_hz=60.0,
        image_height=12,
        image_width=16,
        dynamics_mode="direct",
    )
    thread = threading.Thread(target=node.serve_forever, daemon=True)
    thread.start()

    config = load_config(Path("configs/02_collection/arx_x5_vr.py"))
    config.transport.sub_endpoint = observation_endpoint
    config.transport.pub_endpoint = action_endpoint
    robot = ROBOT_REGISTRY.build("arx_x5")
    transport = ZmqTransport(config, robot)
    try:
        target = np.asarray(robot.initial_qpos, dtype=np.float32).copy()
        target[0] += 0.4
        target[8] -= 0.3
        state = transport.get_latest_qpos()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            transport.publish_action(target)
            time.sleep(0.02)
            state = transport.get_latest_qpos()
            if state is not None and np.array_equal(state, target):
                break

        assert state is not None
        np.testing.assert_array_equal(node.qpos, target)
        np.testing.assert_array_equal(state, target)
    finally:
        transport.close()
        node.stop()
        thread.join(timeout=2.0)
        node.close()


def test_x5_vr_config_supports_fake_node_defaults() -> None:
    config = load_config(Path("configs/02_collection/arx_x5_vr.py"))

    assert config.robot.type == "arx_x5"
    assert config.transport.type == "zmq"
    assert config.transport.sub_endpoint == "tcp://127.0.0.1:5555"
    assert config.transport.pub_endpoint == "tcp://127.0.0.1:5556"
    assert config.transport.disabled_cameras == []
    assert config.collection.teleop.control_source == "client"
    assert config.collection.storage.log_dir == "work_dirs/collection/arx_x5_vr"


def test_x5_fake_node_inherits_the_common_plant() -> None:
    assert issubclass(X5FakeRobotNode, FakeRobotNode)
