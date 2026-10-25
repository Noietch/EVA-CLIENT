from __future__ import annotations

import pytest
import websockets.exceptions

from policy_client._ws import connect_ws
from policy_client.base import PolicyConnectionError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "error",
    [
        websockets.exceptions.SecurityError("line too long"),
        websockets.exceptions.InvalidHandshake("invalid protocol"),
    ],
)
def test_invalid_handshake_becomes_a_recoverable_policy_error(monkeypatch, error):
    def fail_connection(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("websockets.sync.client.connect", fail_connection)
    with pytest.raises(PolicyConnectionError, match="Invalid WebSocket handshake") as result:
        connect_ws("127.0.0.1", 5558, False, 1, "policy server", (OSError,))
    assert result.value.__cause__ is error
