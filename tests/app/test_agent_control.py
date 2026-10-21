from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np

from core.app.agent_control import (
    AgentCommand,
    handle_agent_command,
    queue_agent_command,
    request_agent_stop,
    serialize_agent_status,
)
from core.app.state import RuntimeState, SessionState
from core.config import ConfigDict
from core.types import Observation
from robots.base import ActuatorGroup, ObservationSchema, Robot
from transport.base import TransportBridge


class FakeTransport(TransportBridge):
    def __init__(self, qpos: np.ndarray) -> None:
        self.qpos = qpos.copy()
        self.published: list[np.ndarray] = []

    def get_frame(self) -> Observation | None:
        return Observation(images={}, state_qpos=self.qpos.copy())

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        assert target == "real"
        self.qpos = np.asarray(action, dtype=np.float32).copy()
        self.published.append(self.qpos)

    def get_latest_qpos(self) -> np.ndarray | None:
        return self.qpos.copy()

    def close(self) -> None:
        return


class FakeSolver:
    def fk_chunk(self, qpos_chunk: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos_chunk, dtype=np.float32)
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        rows = []
        for row in qpos:
            rows.append(
                np.array(
                    [row[0], row[1], row[2], 1, 0, 0, 0, row[3], 0, 0, 0, 1, 0, 0, 0, row[7]],
                    dtype=np.float32,
                )
            )
        return np.asarray(rows)

    def solve_chunk(self, eef_chunk: np.ndarray, seed_qpos: np.ndarray | None = None) -> np.ndarray:
        assert seed_qpos is not None
        result = np.tile(np.asarray(seed_qpos, dtype=np.float32), (len(eef_chunk), 1))
        result[-1, :3] = eef_chunk[-1, :3]
        result[-1, 3] = eef_chunk[-1, 7]
        result[-1, 4:7] = 9.0
        return result


def _runtime() -> tuple[ConfigDict, RuntimeState, SessionState, FakeTransport]:
    groups = (
        ActuatorGroup("left_arm", 4, ("l0", "l1", "l2", "lg"), gripper_index=3),
        ActuatorGroup("right_arm", 4, ("r0", "r1", "r2", "rg"), gripper_index=3),
    )
    robot = Robot(
        "fake",
        groups,
        np.zeros(8, dtype=np.float32),
        ObservationSchema(cameras=(), state_composition=("left_arm", "right_arm")),
    )
    transport = FakeTransport(np.array([0.0, 0.0, 0.0, 0.2, 1.0, 1.0, 1.0, 0.8]))
    runtime = RuntimeState(robot=robot, transport=transport, ik_solver=FakeSolver())
    runtime.command_queue = queue.Queue()
    config = ConfigDict(
        {
            "robot": {
                "eef_reference_frame": "base",
                "gripper_open": 1.0,
                "gripper_close": 0.0,
            },
            "inference_cfg": {"publish_rate": 20, "manual_max_qpos_step": 0.1},
            "transport": {"type": "fake", "convert_bgr_to_rgb": True},
            "policy": {"type": "mock"},
        }
    )
    session = SessionState()
    runtime.console_ctx = SimpleNamespace(config=config, session=session, obs_reader=None)
    return config, runtime, session, transport


def test_move_eef_preserves_other_actuator_groups() -> None:
    config, runtime, session, transport = _runtime()
    reply = queue_agent_command(
        runtime,
        {
            "action": "move_eef",
            "arguments": {
                "group": "left_arm",
                "position": [0.3, 0.2, 0.1],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "frame": "base",
            },
        },
    )
    command = runtime.agent_command_queue.get_nowait()
    assert isinstance(command, AgentCommand)

    handle_agent_command(command, config, runtime, session)

    assert runtime.agent_operation is not None
    assert runtime.agent_operation["status"] == "succeeded"
    np.testing.assert_allclose(transport.published[-1][:3], [0.3, 0.2, 0.1])
    np.testing.assert_allclose(transport.published[-1][4:], [1.0, 1.0, 1.0, 0.8])
    assert reply["operation"]["operation_id"] == runtime.agent_operation["operation_id"]


def test_stop_cancels_queued_operation() -> None:
    config, runtime, session, transport = _runtime()
    queue_agent_command(
        runtime,
        {
            "action": "move_joints",
            "arguments": {"group": "left_arm", "positions": [0.1, 0.2, 0.3, 0.4]},
        },
    )
    command = runtime.agent_command_queue.get_nowait()

    reply = request_agent_stop(runtime)
    handle_agent_command(command, config, runtime, session)

    assert reply["operation"]["status"] == "cancel_requested"
    assert runtime.agent_operation is not None
    assert runtime.agent_operation["status"] == "cancelled"
    assert transport.published == []


def test_stop_without_direct_operation_does_not_leave_interrupt_pending() -> None:
    _, runtime, session, _ = _runtime()

    reply = request_agent_stop(runtime)

    assert reply["operation"] is None
    assert session.interrupt_requested is False
    assert runtime.command_queue.get_nowait() == "web:halt"


def test_status_keeps_joint_control_available_without_ik_capability() -> None:
    _, runtime, _, _ = _runtime()
    runtime.ik_solver = None

    status = serialize_agent_status(runtime)

    assert status["direct_control_available"] is True
    assert status["eef_control_available"] is False
    assert [group["name"] for group in status["actuator_groups"]] == [
        "left_arm",
        "right_arm",
    ]


def test_move_joints_rejects_non_finite_positions() -> None:
    config, runtime, session, transport = _runtime()
    queue_agent_command(
        runtime,
        {
            "action": "move_joints",
            "arguments": {"group": "left_arm", "positions": [0.1, float("nan"), 0.3, 0.4]},
        },
    )
    command = runtime.agent_command_queue.get_nowait()

    handle_agent_command(command, config, runtime, session)

    assert runtime.agent_operation is not None
    assert runtime.agent_operation["status"] == "failed"
    assert "finite" in runtime.agent_operation["error"]
    assert transport.published == []


def test_direct_motion_defers_to_armed_operator_teleop() -> None:
    config, runtime, session, transport = _runtime()
    runtime.collection_teleop_armed = True
    queue_agent_command(
        runtime,
        {
            "action": "move_joints",
            "arguments": {"group": "left_arm", "positions": [0.1, 0.2, 0.3, 0.4]},
        },
    )
    command = runtime.agent_command_queue.get_nowait()

    handle_agent_command(command, config, runtime, session)

    assert runtime.agent_operation is not None
    assert runtime.agent_operation["status"] == "failed"
    assert "teleoperation" in runtime.agent_operation["error"]
    assert transport.published == []
