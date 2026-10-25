"""Exercise the Dream HTTP bridge through the existing OpenPI WebSocket client."""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from PIL import Image
from websockets.sync.server import serve

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client.base import PolicyBuildContext, PolicyRequestError
from policy_client.openpi import OpenPiPolicyClient

pytestmark = pytest.mark.integration

_VIDEO_KEYS = {
    "cam_high": "video.overhead",
    "cam_left_wrist": "video.wrist_l",
    "cam_right_wrist": "video.wrist_r",
}
_STATE_SPANS = {
    "state.left_arm": (0, 6),
    "state.left_gripper": (6, 7),
    "state.right_arm": (7, 13),
    "state.right_gripper": (13, 14),
}
_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
_ACTIONS = {
    "q_l_chunk": [[1, 2, 3, 4, 5, 6, 0.25], [7, 8, 9, 10, 11, 12, 0.5]],
    "q_r_chunk": [[21, 22, 23, 24, 25, 26, 0.75], [27, 28, 29, 30, 31, 32, 1.0]],
}


class _Gateway:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.actions: dict = dict(_ACTIONS)
        self.status = {"code": "DA-0000"}
        self.responses: list[dict] = []
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gateway.requests.append((self.path, body))
                assert self.headers["Authorization"] == "Bearer synthetic-test"
                response = {"status": {"code": "DA-0000"}}
                if self.path == "/v1/get_action":
                    response.update(
                        status=gateway.status,
                        session_id="test-session",
                        actions=gateway.actions,
                    )
                    if gateway.responses:
                        response.update(gateway.responses.pop(0))
                elif self.path != "/v1/end_session":
                    self.send_error(404)
                    return
                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _Bridge:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.server = serve(policy.handle_connection, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.socket.getsockname()[1]

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        self.policy.close()


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAM_LOG", "off")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "missing-credentials"))
    monkeypatch.setenv("DREAM_API_KEY", "synthetic-test")
    peer = _Gateway()
    url = f"http://127.0.0.1:{peer.server.server_port}"
    for name in ("DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL"):
        monkeypatch.setenv(name, url)
    try:
        pytest.importorskip("dream", reason="the example's optional Dream SDK is not installed")
        yield peer
    finally:
        peer.close()


@pytest.fixture
def robot_layout():
    return (6, 6), "gripper", _STATE_SPANS


@pytest.fixture
def bridge(gateway, robot_layout):
    from examples.policy.dreamlabs.server import DreamPolicy

    arm_dims, gripper_key, state_spans = robot_layout
    policy = DreamPolicy(
        {
            "checkpoint": "test/checkpoint",
            "embodiment": {
                "state": {
                    key.removeprefix("state."): {"shape": [end - start]}
                    for key, (start, end) in state_spans.items()
                },
                "video": {key.removeprefix("video."): {} for key in _VIDEO_KEYS.values()},
            },
            "obs_window": 3,
            # Retain each test tick without waiting for the default camera period.
            "obs_fps": 1_000_000,
            "timeout_ms": 1000,
        },
        arm_dims=arm_dims,
        gripper_key=gripper_key,
    )
    peer = _Bridge(policy)
    try:
        yield peer
    finally:
        peer.close()


@pytest.fixture
def client(bridge, robot_layout):
    config = ConfigDict(
        host="127.0.0.1",
        port=bridge.port,
    )
    client = POLICY_REGISTRY.build(
        "openpi", config, PolicyBuildContext(action_dim=sum(robot_layout[0]) + 2)
    )
    try:
        yield client
    finally:
        client._ws.close()


def _observation(colors=_COLORS, state_offset=0, state_dim=14):
    return {
        "images": {
            name: np.full((12, 16, 3), color, dtype=np.uint8)
            for name, color in zip(_VIDEO_KEYS, colors, strict=True)
        },
        "state": np.arange(state_dim, dtype=np.float32) + state_offset,
        "prompt": "stack the bowls",
    }


def test_openpi_client_round_trip_preserves_rgb_state_actions_and_session(gateway, client):
    assert isinstance(client, OpenPiPolicyClient)
    assert client.metadata == {
        "server_name": "dreamlabs",
        "action_mode": "qpos",
        "action_dim": 14,
        "chunk_size": 48,
        "supports_reset": True,
    }
    next_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    reset_colors = [(255, 255, 255), (128, 128, 128), (0, 0, 0)]
    expected = np.concatenate([_ACTIONS["q_l_chunk"], _ACTIONS["q_r_chunk"]], axis=1)
    actions = client.infer(_observation())["actions"]
    np.testing.assert_array_equal(actions, expected)
    assert actions.dtype == np.float32
    client.infer(_observation(next_colors, state_offset=20))
    client.reset()
    np.testing.assert_array_equal(
        client.infer(_observation(reset_colors, state_offset=40))["actions"], expected
    )

    assert [path for path, _ in gateway.requests] == [
        "/v1/get_action",
        "/v1/get_action",
        "/v1/end_session",
        "/v1/get_action",
    ]
    requests = [body for path, body in gateway.requests if path == "/v1/get_action"]
    assert [request["session_id"] for request in requests] == [None, "test-session", None]
    assert all(request["checkpoint"] == "test/checkpoint" for request in requests)
    for index, colors in enumerate([_COLORS, next_colors, reset_colors]):
        observation = requests[index]["observation"]
        assert observation["prompt"] == "stack the bowls"
        for key, (start, end) in _STATE_SPANS.items():
            assert observation[key] == list(range(index * 20 + start, index * 20 + end))
        for key, color in zip(_VIDEO_KEYS.values(), colors, strict=True):
            frames = observation[key]
            assert len(frames) == (2 if index == 1 else 1)
            with Image.open(io.BytesIO(base64.b64decode(frames[-1]))) as image:
                assert image.format == "JPEG"
                assert image.mode == "RGB"
                assert image.size == (320, 176)
                np.testing.assert_allclose(np.asarray(image)[88, 160], color, atol=3)
            if index == 1:
                assert frames[0] == requests[0]["observation"][key][0]


@pytest.mark.parametrize("code", ["DA-1000", "DA-5000"])
def test_sdk_failure_reaches_client_and_reset_allows_recovery(gateway, client, code):
    gateway.status = {"code": code, "reason": "synthetic_failure"}
    with pytest.raises(PolicyRequestError) as caught:
        client.infer(_observation())
    assert code in str(caught.value.__cause__)
    gateway.status = {"code": "DA-0000"}
    client.reset()
    assert client.infer(_observation())["actions"].shape == (2, 14)
    requests = [body for path, body in gateway.requests if path == "/v1/get_action"]
    assert [request["session_id"] for request in requests] == [None, None]


def test_expired_session_retries_current_observation_and_reset_releases_replacement(
    gateway, client
):
    client.infer(_observation())
    gateway.responses = [
        {
            "status": {"code": "DA-0001", "reason": "expired"},
            "session_id": "replacement-session",
            "actions": {},
        },
        {"session_id": "replacement-session"},
    ]
    current = _observation(state_offset=20)
    actions = client.infer(current)["actions"]
    expected = np.concatenate([_ACTIONS["q_l_chunk"], _ACTIONS["q_r_chunk"]], axis=1)
    np.testing.assert_array_equal(actions, expected)
    assert actions.dtype == np.float32
    assert [path for path, _ in gateway.requests] == ["/v1/get_action"] * 3
    requests = [body for _, body in gateway.requests]
    assert [request["session_id"] for request in requests] == [
        None, "test-session", "replacement-session"
    ]
    assert requests[1]["observation"] == requests[2]["observation"]
    for key, (start, end) in _STATE_SPANS.items():
        assert requests[2]["observation"][key] == current["state"][start:end].tolist()
    assert all(len(requests[2]["observation"][key]) == 2 for key in _VIDEO_KEYS.values())

    client.reset()
    assert gateway.requests[-1] == (
        "/v1/end_session",
        {"checkpoint": "test/checkpoint", "session_id": "replacement-session"},
    )
    client.infer(_observation(state_offset=40))
    restarted = gateway.requests[-1][1]
    assert restarted["session_id"] is None
    assert all(len(restarted["observation"][key]) == 1 for key in _VIDEO_KEYS.values())
    assert gateway.responses == []


def test_repeated_session_expiry_stops_after_one_retry(gateway, client):
    client.infer(_observation())
    gateway.responses = [
        {
            "status": {"code": "DA-0001", "reason": "expired"},
            "session_id": session_id,
            "actions": {},
        }
        for session_id in ("replacement-1", "replacement-2")
    ]
    # Any extra retry would receive the gateway's successful default and fail this assertion.
    with pytest.raises(PolicyRequestError) as caught:
        client.infer(_observation(state_offset=20))
    assert "replacement-2" in str(caught.value.__cause__)
    assert "expired" in str(caught.value.__cause__)
    assert [path for path, _ in gateway.requests] == ["/v1/get_action"] * 3
    requests = [body for _, body in gateway.requests]
    assert [request["session_id"] for request in requests] == [
        None, "test-session", "replacement-1"
    ]
    assert requests[1]["observation"] == requests[2]["observation"]
    assert gateway.responses == []


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ({"q_l_chunk": [[1, 2]], "q_r_chunk": [[3, 4, 5]]}, "shape [T, 7]"),
        ({"q_l_chunk": [[1, 2, 3, 4, 5, 6, 0.5]]}, "q_r_chunk"),
    ],
)
def test_bad_action_response_reaches_client_and_reset_allows_recovery(
    gateway, client, actions, message
):
    gateway.actions = actions
    with pytest.raises(PolicyRequestError) as caught:
        client.infer(_observation())
    assert message in str(caught.value.__cause__)
    gateway.actions = dict(_ACTIONS)
    client.reset()
    assert client.infer(_observation())["actions"].shape == (2, 14)


@pytest.mark.parametrize("invalid_field", ["state", "camera"])
def test_bad_observation_is_rejected_before_http_and_reset_allows_recovery(
    gateway, client, invalid_field
):
    observation = _observation()
    if invalid_field == "state":
        observation["state"] = observation["state"][:-1]
        message = "14 finite values"
    else:
        del observation["images"]["cam_high"]
        message = "cam_high"
    with pytest.raises(PolicyRequestError) as caught:
        client.infer(observation)
    assert message in str(caught.value.__cause__)
    assert gateway.requests == []
    client.reset()
    assert client.infer(_observation())["actions"].shape == (2, 14)


@pytest.mark.parametrize(
    "robot_layout",
    [
        ((6, 6), "gripper", _STATE_SPANS),
        (
            (7, 7),
            "grip",
            {
                "state.left_arm": (0, 7),
                "state.left_grip": (7, 8),
                "state.right_arm": (8, 15),
                "state.right_grip": (15, 16),
            },
        ),
        (
            (5, 5),
            "grip",
            {
                "state.left_arm": (0, 5),
                "state.left_grip": (5, 6),
                "state.right_arm": (6, 11),
                "state.right_grip": (11, 12),
            },
        ),
    ],
    ids=["yam-6dof", "7dof-grip", "so101-5dof-grip"],
)
@pytest.mark.parametrize("bad_side", ["q_l_chunk", "q_r_chunk"])
def test_robot_layout_preserves_joint_order_and_rejects_wrong_arm_width(
    gateway, client, robot_layout, bad_side
):
    arm_dims, _, state_spans = robot_layout
    state_dim = sum(arm_dims) + 2
    expected = np.arange(2 * state_dim, dtype=np.float32).reshape(2, state_dim)
    left_width = arm_dims[0] + 1
    gateway.actions = {
        "q_l_chunk": expected[:, :left_width].tolist(),
        "q_r_chunk": expected[:, left_width:].tolist(),
    }
    observation = _observation(state_dim=state_dim)
    actions = client.infer(observation)["actions"]
    np.testing.assert_array_equal(actions, expected)
    assert actions.dtype == np.float32
    assert client.metadata["action_dim"] == state_dim
    sent = gateway.requests[0][1]["observation"]
    assert {key for key in sent if key.startswith("state.")} == set(state_spans)
    for key, (start, end) in state_spans.items():
        assert sent[key] == list(range(start, end))

    gateway.actions[bad_side] = [row[:-1] for row in gateway.actions[bad_side]]
    with pytest.raises(PolicyRequestError) as caught:
        client.infer(observation)
    assert f"{bad_side} must have shape" in str(caught.value.__cause__)


def test_metadata_and_reset_need_no_credentials_or_model_lookup(gateway, monkeypatch):
    from examples.policy.dreamlabs.server import DreamPolicy

    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    # No embodiment: initializing this policy would require a platform model lookup.
    policy = DreamPolicy({"checkpoint": "test/checkpoint"})
    peer = _Bridge(policy)
    client = POLICY_REGISTRY.build(
        "openpi", ConfigDict(host="127.0.0.1", port=peer.port), PolicyBuildContext()
    )
    try:
        assert client.metadata == {
            "server_name": "dreamlabs",
            "action_mode": "qpos",
            "action_dim": 14,
            "chunk_size": 48,
            "supports_reset": True,
        }
        client.reset()
        client.reset()
    finally:
        client._ws.close()
        peer.close()
    assert gateway.requests == []
