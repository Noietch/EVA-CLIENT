from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.hardware.arx.node import (
    COLLECTION_CONTROL_CLIENT,
    ArxZmqNode,
)
from transport.zmq import (
    COLLECTION_START_TARGET,
    WireAction,
    pack_action,
    unpack_observation,
)


class _TeleopSource:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.read_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def calibrate_delta(self, qpos: np.ndarray) -> None:
        self.calibrated_qpos = np.asarray(qpos).copy()

    def read_action_qpos(self, _qpos: np.ndarray) -> np.ndarray:
        self.read_calls += 1
        raise AssertionError("client-driven collection must not read Alicia-D")


def _node() -> ArxZmqNode:
    node = object.__new__(ArxZmqNode)
    node._config = SimpleNamespace(passive_collection=False)
    node._collection_active = False
    node._collection_control_source = "transport"
    node._last_published_seqs = {}
    node._teleop_source = _TeleopSource()
    node._hil_active = False
    node._hil_error = ""
    return node


def test_client_collection_mode_does_not_connect_alicia() -> None:
    node = _node()

    node.start_collection(COLLECTION_CONTROL_CLIENT)

    assert node._collection_active is True
    assert node._collection_control_source == COLLECTION_CONTROL_CLIENT
    assert node._teleop_source.connect_calls == 0


def test_transport_collection_mode_keeps_alicia_control() -> None:
    node = _node()
    qpos = np.arange(14, dtype=np.float32)
    node._robot = SimpleNamespace(
        read_state=lambda: {
            "left_arm": qpos[:7],
            "right_arm": qpos[7:],
        }
    )

    node.start_collection()

    assert node._collection_active is True
    assert node._collection_control_source == "transport"
    assert node._teleop_source.connect_calls == 1
    np.testing.assert_allclose(node._teleop_source.calibrated_qpos, qpos)


def test_client_collection_mode_accepts_external_real_actions() -> None:
    node = _node()
    real_action = WireAction(t=2.0, action=np.arange(14, dtype=np.float32), target="real")
    payloads = [
        pack_action(
            WireAction(
                t=1.0,
                action=np.zeros(14, dtype=np.float32),
                target=COLLECTION_START_TARGET,
                mode=COLLECTION_CONTROL_CLIENT,
            )
        ),
        pack_action(real_action),
    ]

    class _Again(Exception):
        pass

    class _ActionSub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.pop(0)

    class _Robot:
        def __init__(self) -> None:
            self.actions = []

        def apply_action(self, action: WireAction) -> None:
            self.actions.append(action)

    node._zmq = SimpleNamespace(NOBLOCK=object(), Again=_Again)
    node._action_sub = _ActionSub()
    node._robot = _Robot()
    node._received_actions = 0

    node._drain_actions()

    assert node._collection_control_source == COLLECTION_CONTROL_CLIENT
    assert node._teleop_source.connect_calls == 0
    assert node._received_actions == 1
    np.testing.assert_allclose(node._robot.actions[0].action, real_action.action)


def test_unknown_collection_control_mode_is_ignored() -> None:
    node = _node()
    payloads = [
        pack_action(
            WireAction(
                t=1.0,
                action=np.zeros(14, dtype=np.float32),
                target=COLLECTION_START_TARGET,
                mode="unknown",
            )
        )
    ]

    class _Again(Exception):
        pass

    class _ActionSub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.pop(0)

    node._zmq = SimpleNamespace(NOBLOCK=object(), Again=_Again)
    node._action_sub = _ActionSub()
    node._robot = SimpleNamespace(apply_action=lambda _action: None)
    node._received_actions = 0

    node._drain_actions()

    assert node._collection_active is False


def test_client_collection_observation_leaves_action_pairing_to_eva() -> None:
    node = _node()
    node._collection_active = True
    node._collection_control_source = COLLECTION_CONTROL_CLIENT
    node._robot = SimpleNamespace(
        read_state=lambda: {
            "left_arm": np.zeros(7, dtype=np.float32),
            "right_arm": np.zeros(7, dtype=np.float32),
        }
    )
    node._cameras = SimpleNamespace(snapshot_versioned=lambda: ({}, {}))
    node._fk = lambda _qpos: np.zeros(16, dtype=np.float32)
    published = []
    node._obs_pub = SimpleNamespace(send=published.append)
    node._published_observations = 0

    node._publish_observation()

    observation = unpack_observation(published[0])
    assert observation.eef is not None
    assert observation.action is None
    assert observation.action_eef is None
    assert node._teleop_source.read_calls == 0
