from __future__ import annotations

from typing import cast

from openpi_client import msgpack_numpy

from policy_client import _ws


class _HandshakeConnection:
    def recv(self) -> bytes:
        return cast(bytes, msgpack_numpy.packb({"ready": True}))


def test_openpi_connect_uses_websockets_13_kwargs(monkeypatch):
    kwargs_seen = {}

    def fake_connect(uri, **kwargs):
        kwargs_seen.update(kwargs)
        return _HandshakeConnection()

    monkeypatch.setattr(_ws.websockets.sync.client, "connect", fake_connect)

    _, metadata = _ws.connect_ws(
        "127.0.0.1",
        10000,
        retry_until_connected=False,
        max_retries=0,
        server_label="policy server",
        retry_errors=(ConnectionRefusedError, OSError),
    )

    assert metadata == {"ready": True}
    assert "additional_headers" not in kwargs_seen
    assert kwargs_seen["compression"] is None
    assert kwargs_seen["max_size"] is None
    assert "ping_interval" not in kwargs_seen
    assert "ping_timeout" not in kwargs_seen


def test_starvla_connect_uses_websockets_13_kwargs(monkeypatch):
    kwargs_seen = {}

    def fake_connect(uri, **kwargs):
        kwargs_seen.update(kwargs)
        return _HandshakeConnection()

    monkeypatch.setattr(_ws.websockets.sync.client, "connect", fake_connect)

    _, metadata = _ws.connect_ws(
        "127.0.0.1",
        10001,
        retry_until_connected=False,
        max_retries=0,
        server_label="starVLA server",
        retry_errors=(ConnectionRefusedError, OSError),
    )

    assert metadata == {"ready": True}
    assert "additional_headers" not in kwargs_seen
    assert kwargs_seen["compression"] is None
    assert kwargs_seen["max_size"] is None
    assert "ping_interval" not in kwargs_seen
    assert "ping_timeout" not in kwargs_seen
