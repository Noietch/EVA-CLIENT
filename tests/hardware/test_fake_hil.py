from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from core.config import load_config
from core.registry import ROBOT_REGISTRY
from examples.hardware.fake_common import FakeRobotNode
from transport.zmq import ZmqTransport

pytestmark = pytest.mark.integration


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.parametrize(
    ("robot_name", "config_path"),
    [
        ("r1_lite", "configs/01_deploy/r1lite/openpi_qpos.py"),
        ("ur5e", "configs/01_deploy/ur5e/openpi_qpos.py"),
        ("arx_r5", "configs/01_deploy/arx_r5/openpi_qpos.py"),
        ("agilex_piper", "configs/01_deploy/dual_agilex_piper/openpi_qpos.py"),
        ("dual_yam", "configs/01_deploy/dual_yam/openpi_qpos.py"),
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
