from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from examples.hardware.arx_x5.node import ArxX5ZmqNode
from transport.zmq import unpack_observation


def test_arx_x5_hardware_logging_keeps_native_output_visible() -> None:
    code = """
import os
from examples.hardware.arx_x5.node import configure_hardware_logging

stream = configure_hardware_logging("INFO")
os.write(1, b"native stdout hidden\\n")
os.write(2, b"native stderr visible\\n")
stream.close()
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "native stdout hidden" in result.stdout + result.stderr
    assert "native stderr visible" in result.stderr


def test_arx_x5_initializes_fk_before_native_hardware_workers(monkeypatch) -> None:
    events: list[str] = []

    class FakeFkSolver:
        def fk_chunk(self, qpos: np.ndarray) -> np.ndarray:
            events.append("fk-warmup")
            assert qpos.shape == (1, 14)
            return np.zeros((1, 16), dtype=np.float32)

    class FakeSocket:
        def bind(self, endpoint: str) -> None:
            return

        def setsockopt(self, option: int, value: object) -> None:
            return

    context = SimpleNamespace(socket=lambda socket_type: FakeSocket())
    fake_zmq = SimpleNamespace(
        Context=SimpleNamespace(instance=lambda: context),
        PUB=1,
        SUB=2,
        SUBSCRIBE=3,
        RCVTIMEO=4,
    )
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    build_fk_solver = ArxX5ZmqNode._build_fk_solver
    monkeypatch.setattr(
        ArxX5ZmqNode,
        "_build_fk_solver",
        staticmethod(lambda: events.append("fk-build") or build_fk_solver()),
    )
    monkeypatch.setattr(
        "core.registry.ROBOT_REGISTRY.build",
        lambda name: SimpleNamespace(build_kinematics=lambda **kwargs: FakeFkSolver()),
    )
    monkeypatch.setattr(
        "examples.hardware.arx_x5.node.ArxX5DualArm",
        lambda config: events.append("robot") or object(),
    )
    monkeypatch.setattr(
        ArxX5ZmqNode,
        "_build_camera_cache",
        lambda self, config: events.append("cameras") or object(),
    )

    config = SimpleNamespace(
        observation_endpoint="inproc://arx_x5-test-observation",
        action_endpoint="inproc://arx_x5-test-action",
        can_ports={},
        realsense_cameras=(),
        publish_rate_hz=100.0,
        status_log_interval_s=0.0,
        disabled_groups=(),
    )
    ArxX5ZmqNode(config)

    assert events == ["fk-build", "fk-warmup", "robot", "cameras"]


def test_arx_x5_fk_solver_can_be_seeded_from_both_live_arm_states(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeFkSolver:
        def fk_chunk(self, qpos: np.ndarray) -> np.ndarray:
            return np.zeros((qpos.shape[0], 16), dtype=np.float32)

    monkeypatch.setattr(
        "core.registry.ROBOT_REGISTRY.build",
        lambda name: SimpleNamespace(
            build_kinematics=lambda **kwargs: captured.update(kwargs) or FakeFkSolver()
        ),
    )
    left = np.asarray([0.1, 1.2, 0.3, 1.4, 0.5, 0.6, 1.0], dtype=np.float32)
    right = np.asarray([-0.1, 1.1, 0.4, 1.3, -0.5, -0.6, 0.0], dtype=np.float32)

    ArxX5ZmqNode._build_fk_solver([left, right])

    seeded_groups = captured["initial_qpos_groups"]
    assert isinstance(seeded_groups, list)
    np.testing.assert_array_equal(seeded_groups[0], left)
    np.testing.assert_array_equal(seeded_groups[1], right)


def test_arx_x5_observation_publishes_state_at_full_rate_and_each_camera_frame_once() -> None:
    node = object.__new__(ArxX5ZmqNode)
    left = np.arange(7, dtype=np.float32)
    right = np.arange(7, 14, dtype=np.float32)
    frame = np.full((8, 12, 3), 127, dtype=np.uint8)
    node._config = SimpleNamespace(passive_collection=False)
    node._collection_active = False
    node._collection_control_source = "transport"
    node._last_published_seqs = {}
    node._robot = SimpleNamespace(read_state=lambda: {"left_arm": left, "right_arm": right})
    node._cameras = SimpleNamespace(
        snapshot_versioned=lambda: ({"cam_high": 1}, {"cam_high": frame})
    )
    node._hil_active = False
    node._hil_error = ""
    node._published_observations = 0
    published: list[bytes] = []
    node._obs_pub = SimpleNamespace(send=published.append)

    node._publish_observation()
    node._publish_observation()

    observation = unpack_observation(published[0])
    next_observation = unpack_observation(published[1])
    np.testing.assert_array_equal(observation.state["left_arm"], left)
    np.testing.assert_array_equal(observation.state["right_arm"], right)
    np.testing.assert_array_equal(observation.images["cam_high"], frame)
    np.testing.assert_array_equal(next_observation.state["left_arm"], left)
    assert next_observation.images == {}
    assert node._published_observations == 2
