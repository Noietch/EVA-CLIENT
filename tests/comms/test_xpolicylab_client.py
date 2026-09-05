"""Focused tests for the standard XPolicyLab websocket client."""

from __future__ import annotations

import threading

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from websockets.sync.server import serve

import policy_client.xpolicylab as xpolicylab
from policy_client.base import PolicyRequestError
from policy_client.xpolicylab import XPolicyLabPolicyClient

pytestmark = pytest.mark.integration


def _get_dim_info(env_cfg_type: str) -> dict:
    assert env_cfg_type == "test_robot"
    return {"arm_dim": [2], "ee_dim": [1]}


def _pack_state(obs: dict, action_type: str, dim_info: dict, source_type: str) -> np.ndarray:
    assert action_type == "joint"
    assert dim_info == {"arm_dim": [2], "ee_dim": [1]}
    assert source_type == "obs"
    return np.concatenate([obs["state"]["joint_state"], obs["state"]["ee_joint_state"]])


def _unpack_state(state: object, action_type: str, dim_info: dict, source_type: str) -> dict:
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size != 3:
        raise ValueError(f"packed_state last dim mismatch: expected 3, got {values.size}")
    assert action_type == "joint"
    assert dim_info == {"arm_dim": [2], "ee_dim": [1]}
    assert source_type == "obs"
    return {"joint_state": values[:2], "ee_joint_state": values[2:]}


@pytest.fixture(autouse=True)
def robot_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xpolicylab,
        "_load_robot_codec",
        lambda root: (_get_dim_info, _pack_state, _unpack_state),
    )


class _Connection:
    def __init__(self, actions: object) -> None:
        self.actions = actions
        self.sent: list[dict] = []
        self.closed = False

    def send(self, payload: bytes) -> None:
        self.sent.append(XPolicyLabPolicyClient._decode_frame(payload))

    def recv(self, timeout: float | None = None) -> bytes:
        _ = timeout
        request = self.sent[-1]
        response_type = {
            "hello": "hello_ack",
            "infer": "infer_result",
            "reset": "reset_result",
        }[request["message_type"]]
        payload: dict = {"ok": True}
        if response_type == "hello_ack":
            payload["server"] = "xpolicylab_policy_server"
        elif response_type == "infer_result":
            payload.update(actions=self.actions, latency_ms=12.5)
        return msgpack.packb(
            {
                "message_type": response_type,
                "message_id": request["message_id"],
                "evaluation_id": request["evaluation_id"],
                "trial_id": request.get("trial_id"),
                "step": request["step"],
                "payload": payload,
            },
            default=msgpack_numpy.encode,
            use_bin_type=True,
        )

    def close(self) -> None:
        self.closed = True


def _observation() -> dict:
    return {
        "state": np.arange(3, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((8, 10, 3), dtype=np.uint8),
            "cam_left_wrist": np.ones((8, 10, 3), dtype=np.uint8),
        },
        "prompt": "stack the bowls",
    }


def test_client_round_trip_over_local_websocket():
    peer = _Connection(
        [
            {
                "joint_state": np.array([offset, offset + 1], dtype=np.float32),
                "ee_joint_state": np.array([offset + 2], dtype=np.float32),
            }
            for offset in (0, 10)
        ]
    )

    # Exercise the pinned websocket transport with real socket traffic
    def respond(connection):
        for payload in connection:
            peer.send(payload)
            connection.send(peer.recv())

    with serve(respond, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with XPolicyLabPolicyClient(
                "127.0.0.1",
                server.socket.getsockname()[1],
                env_cfg_type="test_robot",
                action_type="joint",
                retry_until_connected=False,
            ) as client:
                client.reset()
                actions = client.infer(_observation())["actions"]
                np.testing.assert_array_equal(actions, [[0, 1, 2], [10, 11, 12]])
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert [frame["message_type"] for frame in peer.sent] == ["hello", "reset", "infer"]


def test_client_rejects_nonfinite_actions():
    actions = np.zeros((2, 3), dtype=np.float32)
    actions[1, 2] = np.nan
    client = XPolicyLabPolicyClient(
        "127.0.0.1",
        19000,
        env_cfg_type="test_robot",
        action_type="joint",
        client=_Connection(actions),
    )

    with np.testing.assert_raises_regex(PolicyRequestError, "NaN or Inf"):
        client.infer(_observation())
