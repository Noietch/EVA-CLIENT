"""Focused tests for the standard XPolicyLab websocket client."""

from __future__ import annotations

import builtins
import sys
import threading

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from websockets.sync.server import serve

from policy_client.base import PolicyRequestError
from policy_client.xpolicylab import XPolicyLabPolicyClient

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def forbid_xpolicylab_imports(monkeypatch):
    original_import = builtins.__import__

    def import_without_xpolicylab(name, *args, **kwargs):
        if name == "XPolicyLab" or name.startswith("XPolicyLab."):
            raise AssertionError(f"client attempted to import external dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_xpolicylab)


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


def _observation(state_dim: int = 3) -> dict:
    return {
        "state": np.arange(state_dim, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((8, 10, 3), dtype=np.uint8),
            "cam_left_wrist": np.ones((8, 10, 3), dtype=np.uint8),
            "cam_right_wrist": np.full((8, 10, 3), 2, dtype=np.uint8),
        },
        "prompt": "stack the bowls",
    }


@pytest.mark.parametrize(
    ("action_type", "arm_dims", "gripper_dims", "fields"),
    [
        ("joint", [2], [1], {"joint_state": (0, 2), "ee_joint_state": (2, 3)}),
        (
            "joint",
            [6, 6],
            [1, 1],
            {
                "left_arm_joint_state": (0, 6),
                "left_ee_joint_state": (6, 7),
                "right_arm_joint_state": (7, 13),
                "right_ee_joint_state": (13, 14),
            },
        ),
        ("ee", [7], [1], {"ee_pose": (0, 7), "ee_joint_state": (7, 8)}),
        (
            "ee",
            [7, 7],
            [1, 1],
            {
                "left_ee_pose": (0, 7),
                "left_ee_joint_state": (7, 8),
                "right_ee_pose": (8, 15),
                "right_ee_joint_state": (15, 16),
            },
        ),
    ],
)
@pytest.mark.parametrize("action_format", ["steps", "chunk", "array"])
def test_client_round_trip_over_local_websocket(
    action_type, arm_dims, gripper_dims, fields, action_format
):
    action_dim = sum(arm_dims) + sum(gripper_dims)
    expected_actions = np.arange(2 * action_dim, dtype=np.float32).reshape(2, action_dim)
    if action_format == "steps":
        actions = [
            {key: step[start:end] for key, (start, end) in fields.items()}
            for step in expected_actions
        ]
    elif action_format == "chunk":
        actions = {key: expected_actions[:, start:end] for key, (start, end) in fields.items()}
    else:
        actions = expected_actions
    peer = _Connection(actions)
    observation = _observation(action_dim)
    original_path = sys.path.copy()

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
                action_type=action_type,
                arm_dims=arm_dims,
                gripper_dims=gripper_dims,
                retry_until_connected=False,
            ) as client:
                client.reset()
                actions = client.infer(observation)["actions"]
                np.testing.assert_array_equal(actions, expected_actions)
                assert actions.dtype == np.float32
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert sys.path == original_path
    assert [frame["message_type"] for frame in peer.sent] == ["hello", "reset", "infer"]
    vision = peer.sent[-1]["payload"]["observation"]["vision"]
    assert list(vision) == list(_observation()["images"])
    for name, image in _observation()["images"].items():
        np.testing.assert_array_equal(vision[name]["color"], image)
    state = peer.sent[-1]["payload"]["observation"]["state"]
    assert list(state) == list(fields)
    for key, (start, end) in fields.items():
        np.testing.assert_array_equal(state[key], observation["state"][start:end])


def test_client_rejects_nonfinite_actions():
    actions = np.zeros((2, 3), dtype=np.float32)
    actions[1, 2] = np.nan
    client = XPolicyLabPolicyClient(
        "127.0.0.1",
        19000,
        action_type="joint",
        arm_dims=[2],
        gripper_dims=[1],
        client=_Connection(actions),
    )

    with np.testing.assert_raises_regex(PolicyRequestError, "NaN or Inf"):
        client.infer(_observation())


def test_client_rejects_action_field_dimensions_before_executing():
    client = XPolicyLabPolicyClient(
        "127.0.0.1",
        19000,
        action_type="joint",
        arm_dims=[2],
        gripper_dims=[1],
        client=_Connection({"joint_state": [1], "ee_joint_state": [2, 3]}),
    )
    with pytest.raises(PolicyRequestError, match="joint_state.*last dimension 2"):
        client.infer(_observation())
