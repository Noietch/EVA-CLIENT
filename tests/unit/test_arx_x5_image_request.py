"""An on-demand image request must not change the state the ARX X5 node reports."""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

HOME = np.array([0.0, 1.58, 0.49, 1.15, 0.0, 0.0, 1.0] * 2, dtype=np.float32)
OPEN_POS, CLOSE_POS = -3.4, 0.1


class _Again(Exception):
    pass


class _Arm:
    """Stand-in for the vendor SDK arm that records every gripper command."""

    gripper_error = 0
    gripper_commands: list[float] = []

    def __init__(self, _config: dict) -> None:
        self._joints = HOME[:6].tolist()

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def get_joint_currents(self) -> list[float]:
        return [0.0] * 6

    def set_joint_positions(self, joints: list[float], duration: float | None = None) -> bool:
        self._joints = list(joints)
        return True

    def set_gripper_pos(self, position: float, duration: float | None = None) -> bool:
        _Arm.gripper_commands.append(float(position))
        return True


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    sdk = types.ModuleType("examples.hardware.arx_x5.sdk")
    sdk.SingleArm = _Arm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "examples.hardware.arx_x5.sdk", sdk)
    monkeypatch.setattr(_Arm, "gripper_commands", [])
    from examples.hardware.arx_x5.robot import ArxX5DualArm

    robot = ArxX5DualArm(
        SimpleNamespace(
            can_ports={"left_arm": "can0", "right_arm": "can1"},
            arm_type=2,
            gripper_open_pos=OPEN_POS,
            gripper_close_pos=CLOSE_POS,
            initial_gripper_scalar=1.0,
            start_at_zero=False,
            disabled_groups=(),
            publish_rate_hz=100.0,
        )
    )
    pending: list[bytes] = []

    def recv(_flags: int) -> bytes:
        if not pending:
            raise _Again
        return pending.pop(0)

    return SimpleNamespace(
        pending=pending,
        _action_sub=SimpleNamespace(recv=recv),
        _zmq=SimpleNamespace(NOBLOCK=1, Again=_Again),
        _robot=robot,
        _collection_active=False,
        _config=SimpleNamespace(passive_collection=False),
        _received_actions=0,
    )


def _deliver(node: SimpleNamespace, action: np.ndarray, target: str) -> None:
    from examples.hardware.arx_x5.node import ArxX5ZmqNode
    from transport.zmq import WireAction, pack_action

    node.pending.append(pack_action(WireAction(t=time.monotonic(), action=action, target=target)))
    ArxX5ZmqNode._drain_actions(node)


def _reported_grippers(node: SimpleNamespace) -> list[float]:
    state = node._robot.read_state()
    return [float(state[group][6]) for group in ("left_arm", "right_arm")]


def test_image_request_keeps_the_reported_gripper(node: SimpleNamespace) -> None:
    from transport.zmq import IMAGE_REQUEST_TARGET

    _deliver(node, HOME, "real")
    commands_before = list(_Arm.gripper_commands)

    # What ZmqTransport sends before every on-demand policy frame.
    _deliver(node, np.zeros(14, dtype=np.float32), IMAGE_REQUEST_TARGET)

    assert _reported_grippers(node) == [1.0, 1.0]
    assert _Arm.gripper_commands == commands_before


def test_real_action_still_drives_the_gripper(node: SimpleNamespace) -> None:
    closed = HOME.copy()
    closed[[6, 13]] = 0.0

    _deliver(node, closed, "real")

    assert _reported_grippers(node) == [0.0, 0.0]
    # Each arm connects lazily (one open command), then receives its close command.
    assert [c for c in _Arm.gripper_commands if c == pytest.approx(CLOSE_POS)] == [CLOSE_POS] * 2
