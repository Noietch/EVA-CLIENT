"""Hardware nodes hand only "real" actions to the robot.

The client also publishes sim previews and, in on-demand image mode, a zero
placeholder before every policy frame on the same action channel. Applying either
would move the robot (UR5e servos to the zero pose) or overwrite the state the
node reports to the policy (the ARX grippers read 0).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import zmq

from transport.zmq import IMAGE_REQUEST_TARGET, WireAction, pack_action

pytestmark = pytest.mark.unit


def _queue(*actions: WireAction):
    pending = [pack_action(action) for action in actions]

    def recv(_flags: int) -> bytes:
        if not pending:
            raise zmq.Again
        return pending.pop(0)

    return SimpleNamespace(recv=recv)


def _arx_r5(sink, sub):
    from examples.hardware.arx_r5.node import ArxR5ZmqNode

    node = SimpleNamespace(
        _action_sub=sub,
        _zmq=zmq,
        _robot=SimpleNamespace(apply_action=lambda action: sink.append(action.action)),
        _collection_active=False,
        _config=SimpleNamespace(passive_collection=False),
        _received_actions=0,
    )
    return ArxR5ZmqNode._drain_actions, node


def _ur5e(sink, sub):
    from examples.hardware.ur5e.node import Ur5eZmqNode

    node = SimpleNamespace(_action_sub=sub, _zmq=zmq, _apply_action=sink.append)
    return Ur5eZmqNode._drain_actions, node


def _yam(sink, sub):
    from examples.hardware.yam.node import YamZmqNode

    node = SimpleNamespace(
        _action_sub=sub,
        _followers=SimpleNamespace(apply_action=lambda action: sink.append(action.action)),
        _direct_leader_control=False,
        _hil_active=False,
        _collection_active=False,
        _collection_control_source="transport",
        _received_actions=0,
    )
    return YamZmqNode._drain_actions, node


def _franka(sink, sub):
    from examples.hardware.franka.node import FrankaZmqNode

    node = SimpleNamespace(
        _action_sub=sub,
        _zmq=zmq,
        _robot=SimpleNamespace(apply_action=lambda action: sink.append(action.action)),
        _received_actions=0,
    )
    return FrankaZmqNode._drain_actions, node


@pytest.mark.parametrize(
    "build", [_arx_r5, _ur5e, _yam, _franka], ids=["arx_r5", "ur5e", "yam", "franka"]
)
def test_node_applies_only_real_actions(build) -> None:
    real = np.arange(1, 15, dtype=np.float32)
    sub = _queue(
        WireAction(t=time.monotonic(), action=np.full(14, 9.0, dtype=np.float32), target="sim"),
        WireAction(
            t=time.monotonic(), action=np.zeros(14, dtype=np.float32), target=IMAGE_REQUEST_TARGET
        ),
        WireAction(t=time.monotonic(), action=real, target="real"),
    )
    sink: list[np.ndarray] = []
    drain, node = build(sink, sub)

    drain(node)

    assert len(sink) == 1
    np.testing.assert_array_equal(sink[0], real)
