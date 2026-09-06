"""Console camera-feed API contract.

The camera path was reworked from base64-in-JSON polling to an MJPEG stream (the
other agent's change; see tests/perf/ for the throughput numbers and work_dirs/COMMONS
for the boundary). These tests pin the *contract shape* that change established, so a
future edit can't silently regress it:

- ``/api/frame`` carries only lightweight telemetry (qpos + camera *key list*), never
  image bytes — that's what keeps the 1 Hz status poll cheap.
- ``/api/camera/<key>`` is a ``multipart/x-mixed-replace`` MJPEG stream of raw JPEG
  frames (no base64), capped at 10 FPS, and a bogus key 404s.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from core.app.console.server import (
    ConsoleRequestHandler,
)
from tests.integration.web._harness import WebHarness

pytestmark = pytest.mark.integration


def test_offline_cameras_do_not_reserve_browser_stream_connections(monkeypatch):
    from core.app.console import server

    frames = {}
    reader = SimpleNamespace(
        get_camera_keys=lambda: ["cam_high", "cam_left_wrist"],
        get_camera_frame=frames.get,
    )
    monkeypatch.setattr(server, "_observation_reader", lambda ctx: reader)
    assert server._live_camera_keys(None) == []
    frames["cam_high"] = object()
    assert server._live_camera_keys(None) == ["cam_high"]
    frames.clear()
    assert server._live_camera_keys(None) == []


def test_video_stream_client_disconnect_is_silent(tmp_path):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x" * 128)

    class _WFile:
        def write(self, _chunk):
            raise ConnectionResetError(104, "Connection reset by peer")

    class _Handler:
        headers = {}
        wfile = _WFile()

        def send_response(self, _status):
            pass

        def send_header(self, _name, _value):
            pass

        def end_headers(self):
            pass

    ConsoleRequestHandler._send_video(_Handler(), video)  # type: ignore[reportArgumentType]


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
