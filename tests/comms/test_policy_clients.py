"""Tests for built-in mock policy clients (policy_client.base)."""

from __future__ import annotations

import numpy as np
import pytest
from openpi_client import msgpack_numpy

from policy_client.base import DatasetReplayPolicyClient, RandomPolicyClient
from policy_client.openpi import RtcOpenPiPolicyClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """RandomPolicyClient.infer sleeps to mimic latency; skip it in tests."""
    monkeypatch.setattr("policy_client.base.time.sleep", lambda *_a, **_k: None)


# --- RandomPolicyClient ---


def test_random_client_output_shape():
    client = RandomPolicyClient(action_dim=14, chunk_size=50)
    out = client.infer({})
    assert out["actions"].shape == (50, 14)
    assert out["actions"].dtype == np.float32


def test_random_client_base_trajectory_is_deterministic():
    """Same global step must yield the same base value across two fresh clients."""
    a = RandomPolicyClient(action_dim=4, chunk_size=10, step_advance=10)
    b = RandomPolicyClient(action_dim=4, chunk_size=10, step_advance=10)
    # Drop per-chunk polynomial perturbation; compare deterministic sinusoid base.
    # The base only depends on global step, so chunk 0 of both clients matches.
    out_a = a.infer({})["actions"]
    out_b = b.infer({})["actions"]
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


def test_random_client_metadata():
    client = RandomPolicyClient(action_dim=14, chunk_size=32)
    md = client.metadata
    assert md["action_dim"] == 14
    assert md["chunk_size"] == 32
    assert md["action_mode"] == "qpos"


def test_random_client_reset_restarts_chunk_index():
    client = RandomPolicyClient(action_dim=4, chunk_size=10, step_advance=10)
    first = client.infer({})["actions"]
    client.infer({})  # advance chunk_idx
    client.reset()
    after_reset = client.infer({})["actions"]
    np.testing.assert_allclose(first, after_reset, atol=1e-6)


# --- DatasetReplayPolicyClient ---


def test_replay_cursor_advances():
    actions = np.arange(30.0).reshape(15, 2).astype(np.float32)
    client = DatasetReplayPolicyClient(actions, chunk_size=5)
    first = client.infer({})["actions"]
    np.testing.assert_allclose(first, actions[:5], atol=1e-6)
    second = client.infer({})["actions"]
    np.testing.assert_allclose(second, actions[5:10], atol=1e-6)


def test_replay_short_tail_padded_with_last_pose():
    actions = np.arange(14.0).reshape(7, 2).astype(np.float32)
    client = DatasetReplayPolicyClient(actions, chunk_size=5)
    client.infer({})  # consume 0..5
    tail = client.infer({})["actions"]  # steps 5,6 + pad
    assert tail.shape == (5, 2)
    np.testing.assert_allclose(tail[0], actions[5], atol=1e-6)
    np.testing.assert_allclose(tail[1], actions[6], atol=1e-6)
    # remaining rows hold the last recorded pose
    np.testing.assert_allclose(tail[2], actions[-1], atol=1e-6)
    np.testing.assert_allclose(tail[4], actions[-1], atol=1e-6)


def test_replay_exhausted_holds_last_pose():
    actions = np.arange(6.0).reshape(3, 2).astype(np.float32)
    client = DatasetReplayPolicyClient(actions, chunk_size=3)
    client.infer({})  # consume all 3
    after = client.infer({})["actions"]
    # cursor clamped to last step -> all rows are the final pose
    np.testing.assert_allclose(after, np.tile(actions[-1], (3, 1)), atol=1e-6)


def test_replay_reset_restarts_cursor():
    actions = np.arange(20.0).reshape(10, 2).astype(np.float32)
    client = DatasetReplayPolicyClient(actions, chunk_size=4)
    first = client.infer({})["actions"]
    client.reset()
    np.testing.assert_allclose(client.infer({})["actions"], first, atol=1e-6)


def test_replay_metadata_reports_steps():
    actions = np.zeros((12, 7), dtype=np.float32)
    client = DatasetReplayPolicyClient(actions, chunk_size=5)
    md = client.metadata
    assert md["n_steps"] == 12
    assert md["action_dim"] == 7
    assert md["chunk_size"] == 5


# --- RtcOpenPiPolicyClient ---


class _RtcFakeConnection:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.sent: list[dict] = []

    def send(self, payload: bytes) -> None:
        self.sent.append(msgpack_numpy.unpackb(payload))

    def recv(self) -> bytes:
        return msgpack_numpy.packb(self._responses.pop(0))


def test_rtc_client_feeds_back_origin_actions_when_available():
    robot_actions = np.zeros((50, 14), dtype=np.float32)
    origin_actions = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    connection = _RtcFakeConnection(
        [
            {"actions": robot_actions, "origin_actions": origin_actions},
            {"actions": robot_actions + 1, "origin_actions": origin_actions + 1},
        ]
    )
    client = RtcOpenPiPolicyClient(
        "127.0.0.1",
        9000,
        latency_k=4,
        retry_until_connected=False,
        client=(connection, {"ready": True}),
    )

    client.infer({"state": np.zeros(14, dtype=np.float32)})
    client.infer({"state": np.ones(14, dtype=np.float32)})

    sent_prev_action = connection.sent[1]["prev_action"]
    expected = origin_actions.copy()
    expected[:46] = origin_actions[4:]
    expected[46:] = origin_actions[-1]
    assert sent_prev_action.shape == (50, 32)
    np.testing.assert_allclose(sent_prev_action, expected, atol=1e-6)
