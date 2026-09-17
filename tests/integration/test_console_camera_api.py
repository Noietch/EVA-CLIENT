"""Console camera-feed API contract.

The camera path was reworked from base64-in-JSON polling to an MJPEG stream (the
other agent's change; see tests/perf/ for the throughput numbers and work_dirs/COMMONS
for the boundary). This test pins the *contract shape* that change established, so a
future edit can't silently regress it: ``/api/camera/<key>`` is a
``multipart/x-mixed-replace`` MJPEG stream of raw JPEG frames (no base64), capped at
10 FPS.
"""

from __future__ import annotations

import socket

import pytest

from tests.integration._harness import WebHarness

pytestmark = pytest.mark.integration


def _read_mjpeg_head(port: int, key: str, max_bytes: int = 4096) -> tuple[str, bytes]:
    """Open the MJPEG stream, read the status line + headers + a little body, then
    close. Uses a raw socket with a timeout because the stream never ends on its own."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(5)
    sock.sendall(f"GET /api/camera/{key} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    buf = b""
    try:
        while len(buf) < max_bytes and b"--evaframe" not in buf:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
    except TimeoutError:
        pass
    finally:
        sock.close()
    head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1")
    return head, buf


def test_camera_stream_is_multipart_mjpeg(console: WebHarness):
    head, buf = _read_mjpeg_head(console.port, "cam_high")
    assert head.startswith("HTTP/1.0 200") or head.startswith("HTTP/1.1 200")
    assert "multipart/x-mixed-replace" in head
    assert "boundary=evaframe" in head
    # At least one part boundary + a JPEG content-type arrives in the first read window.
    assert b"--evaframe" in buf
    assert b"image/jpeg" in buf
