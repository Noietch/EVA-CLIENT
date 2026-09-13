"""High-rate SSE feedback stream used by the Collect VR control hints."""

from __future__ import annotations

import json
import socket

import pytest

from tests.integration.web._harness import WebHarness

pytestmark = pytest.mark.integration


def _read_sse_event(port: int, max_bytes: int = 4096) -> tuple[str, bytes]:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(5)
    sock.sendall(
        b"GET /api/teleop/feedback HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Accept: text/event-stream\r\n\r\n"
    )
    buf = b""
    try:
        while len(buf) < max_bytes and b"\n\ndata:" not in buf:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
        while len(buf) < max_bytes and buf.count(b"\n\n") < 2:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
    finally:
        sock.close()
    head, body = buf.split(b"\r\n\r\n", 1)
    return head.decode("latin-1"), body


def test_teleop_feedback_is_sse(console: WebHarness):
    head, body = _read_sse_event(console.port)
    assert "200" in head
    assert "text/event-stream" in head
    assert "X-Accel-Buffering: no" in head
    assert b"retry: 1000" in body
    data_line = next(line for line in body.splitlines() if line.startswith(b"data: "))
    payload = json.loads(data_line.removeprefix(b"data: "))
    assert set(payload) == {"connected", "pressed_controls", "hold_progress", "review_events"}
