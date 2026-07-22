from __future__ import annotations

import numpy as np
from openpi_client import msgpack_numpy

from core.config import ConfigDict
from critic_client.base import CriticBuildContext, MockCriticClient
from critic_client.websocket import WebSocketCriticClient


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes:
        return msgpack_numpy.packb({"value": 0.625})


def test_mock_critic_returns_one_scalar_value():
    critic = MockCriticClient.from_config(ConfigDict(backend_options={}), CriticBuildContext())

    value = critic.evaluate(
        {"state": np.array([0.1, 0.2], dtype=np.float32)},
        np.array([[0.3, 0.4]], dtype=np.float32),
    )

    assert isinstance(value, float)
    assert critic.metadata["server_name"] == "eva-mock-critic"


def test_websocket_critic_sends_observation_and_actions_and_requires_scalar_value():
    connection = _FakeConnection()
    critic = WebSocketCriticClient(
        "127.0.0.1",
        9100,
        client=(connection, {"server_name": "fake-critic"}),
    )
    observation = {"state": np.array([0.1, 0.2], dtype=np.float32)}
    actions = np.array([[0.3, 0.4]], dtype=np.float32)

    value = critic.evaluate(observation, actions)

    assert value == 0.625
    request = msgpack_numpy.unpackb(connection.sent[0])
    np.testing.assert_allclose(request["observation"]["state"], observation["state"])
    np.testing.assert_allclose(request["actions"], actions)
