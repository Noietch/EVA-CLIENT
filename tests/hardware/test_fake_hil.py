from __future__ import annotations

import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from core.config import load_config
from core.registry import ROBOT_REGISTRY
from examples.hardware.fake_common import FakeRobotNode
from transport.zmq import ZmqTransport


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.parametrize(
    "robot_name",
    ["r1_lite", "ur5e", "arx_r5", "agilex_piper"],
)
def test_generic_fake_hil_relative_takeover_for_every_robot(robot_name: str):
    suffix = uuid.uuid4().hex
    node = FakeRobotNode(
        robot_name=robot_name,
        observation_endpoint=f"inproc://fake-hil-observation-{suffix}",
        action_endpoint=f"inproc://fake-hil-action-{suffix}",
        publish_rate_hz=30.0,
        image_height=8,
        image_width=8,
    )
    try:
        before = node.hil_snapshot()
        group = before["groups"][0]
        group_name = group["name"]
        first_before = np.asarray(before["feedback"][group_name], dtype=np.float32)

        node.start_hil("relative")
        node.adjust_hil_joint(group_name, 0, 0.2)
        node._publish_observation()

        after = node.hil_snapshot()
        first_after = np.asarray(after["feedback"][group_name], dtype=np.float32)
        assert after["supported"] is True
        assert after["active"] is True
        assert first_after[0] == pytest.approx(first_before[0] + 0.2)

        node.stop_hil()
        assert node.hil_snapshot()["active"] is False
    finally:
        node.close()


@pytest.mark.parametrize("robot_name", ["dual_franka", "agibot_g2"])
def test_generic_fake_reports_hil_unsupported_without_leader_adapter(robot_name: str):
    suffix = uuid.uuid4().hex
    node = FakeRobotNode(
        robot_name=robot_name,
        observation_endpoint=f"inproc://fake-no-hil-observation-{suffix}",
        action_endpoint=f"inproc://fake-no-hil-action-{suffix}",
        publish_rate_hz=30.0,
        image_height=8,
        image_width=8,
    )
    try:
        node.start_hil("relative")
        snapshot = node.hil_snapshot()
        assert snapshot["supported"] is False
        assert snapshot["active"] is False
        assert snapshot["error"] == f"{robot_name} has no HIL leader adapter"
    finally:
        node.close()


@pytest.mark.parametrize(
    ("robot_name", "config_path"),
    [
        ("r1_lite", "configs/01_deploy/r1lite/openpi_qpos.py"),
        ("ur5e", "configs/01_deploy/ur5e/openpi_qpos.py"),
        ("arx_r5", "configs/01_deploy/arx_r5/openpi_qpos.py"),
        ("agilex_piper", "configs/01_deploy/dual_agilex_piper/openpi_qpos.py"),
    ],
)
def test_zmq_transport_confirms_fake_hil_and_receives_user_action(
    robot_name: str, config_path: str
):
    obs_port = _free_port()
    action_port = _free_port()
    obs_endpoint = f"tcp://127.0.0.1:{obs_port}"
    action_endpoint = f"tcp://127.0.0.1:{action_port}"
    node = FakeRobotNode(robot_name, obs_endpoint, action_endpoint, 60.0, 8, 8)
    thread = threading.Thread(target=node.serve_forever, daemon=True)
    thread.start()
    config = load_config(Path(config_path))
    config.transport.type = "zmq"
    config.transport.sub_endpoint = obs_endpoint
    config.transport.pub_endpoint = action_endpoint
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = ZmqTransport(config, robot)
    try:
        deadline = time.monotonic() + 2.0
        status = transport.hil_status()
        while not status.supported and time.monotonic() < deadline:
            time.sleep(0.02)
            status = transport.hil_status()
        assert status.supported is True

        started = transport.start_hil_control("relative")
        assert started.active is True
        group = robot.actuator_groups[0]
        node.adjust_hil_joint(group.name, 0, 0.2)

        frame = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            frame = transport.get_hil_frame()
            if frame is not None and frame.action_qpos is not None:
                break
            time.sleep(0.02)
        assert frame is not None
        assert frame.action_qpos is not None
        assert frame.action_qpos.shape == (robot.total_action_dim,)

        stopped = transport.stop_hil_control()
        assert stopped.active is False
    finally:
        transport.close()
        node.close()
        thread.join(timeout=2.0)
