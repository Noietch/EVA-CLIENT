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
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from typing import Any

import numpy as np
import pytest
from _harness import WebHarness, build_runtime, console_config

from core.app.console.server import (
    _CAMERA_STREAM_FPS,
    ConsoleContext,
    ConsoleRequestHandler,
    _camera_jpeg_payload,
    _serialize_config,
    _serialize_frame,
    _serialize_status,
)
from core.app.handlers.teleop import TeleopExecutionState
from core.config import ConfigDict
from teleop_client.base import TeleopStatus


def _serialize_console_payload(
    serializer: Callable[[ConsoleContext], dict[str, Any]],
    *,
    config: ConfigDict | None = None,
    obs_reader: Any = None,
    configure_runtime: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    config = config or console_config()
    runtime, session = build_runtime(config)
    if configure_runtime is not None:
        configure_runtime(runtime)
    reader = obs_reader if obs_reader is not None else runtime.transport.create_observation_reader()

    try:
        return serializer(
            ConsoleContext(
                config=config,
                runtime=runtime,
                session=session,
                obs_reader=reader,
            )
        )
    finally:
        runtime.transport.close()


def test_camera_stream_is_capped_at_ten_fps():
    assert _CAMERA_STREAM_FPS == 10.0


def test_frame_carries_telemetry_not_image_bytes(console: WebHarness):
    frame = console.get("/api/frame").json
    # qpos is the live joint vector (piper = 14 DoF).
    assert isinstance(frame["qpos"], list) and len(frame["qpos"]) == 14
    # cameras is a list of *keys*, not a {key: base64} blob — images stream separately.
    assert isinstance(frame["cameras"], list)
    assert frame["cameras"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for key in frame["cameras"]:
        assert isinstance(key, str)


def test_status_reports_image_min_hz_from_observation_reader():
    class _Reader:
        def get_latest_qpos(self):
            return np.zeros(14, dtype=np.float32)

        def seconds_since_last_recv(self):
            return 0.0

        def image_min_hz(self):
            return 14.25

    status = _serialize_console_payload(
        _serialize_status,
        obs_reader=_Reader(),  # type: ignore[reportArgumentType]
    )

    assert status["image_min_hz"] == 14.25


def test_status_consumes_observation_reader_before_checking_live_state():
    class _Reader:
        consumed = False

        def get_latest_qpos(self):
            self.consumed = True
            return np.zeros(14, dtype=np.float32)

        def seconds_since_last_recv(self):
            return 0.0 if self.consumed else None

        def image_min_hz(self):
            return 30.0 if self.consumed else None

    reader = _Reader()
    status = _serialize_console_payload(
        _serialize_status,
        obs_reader=reader,  # type: ignore[reportArgumentType]
    )

    assert reader.consumed
    assert status["transport_connected"] is True
    assert status["image_min_hz"] == 30.0


def test_status_exposes_collection_arm_lifecycle_without_transport_teleop_details():
    def configure_runtime(runtime: Any) -> None:
        runtime.collection_teleop_armed = True
        runtime.collection_teleop_active = True

    status = _serialize_console_payload(_serialize_status, configure_runtime=configure_runtime)

    assert status["collection_teleop_armed"] is True
    assert status["collection_teleop_active"] is True
    assert status["teleop"] is None
    assert status["rollout_intervention_source"] == "transport"
    assert "teleop_collection_metrics" not in status


def test_status_exposes_lightweight_client_input_source_health():
    class _TeleopClient:
        def status(self):
            return TeleopStatus(
                source_type="vr_webxr",
                connected=True,
                input_age_ms=12.5,
                engaged_groups=("left_arm",),
                held_groups=("right_arm",),
                authorized_groups=("left_arm", "right_arm"),
                pressed_controls=("right.primary",),
                hold_progress=(("right.primary", 0.65),),
                neutral=True,
            )

    def configure_runtime(runtime: Any) -> None:
        runtime.teleop_execution = TeleopExecutionState(
            control_source="client",
            client_type="vr_webxr",
            active=True,
        )
        runtime.teleop_client = _TeleopClient()

    status = _serialize_console_payload(_serialize_status, configure_runtime=configure_runtime)

    assert status["teleop"] == {
        "control_source": "client",
        "client_type": "vr_webxr",
        "active": True,
        "condition": "idle",
        "connected": True,
        "input_age_ms": 12.5,
        "engaged_groups": ["left_arm"],
        "held_groups": ["right_arm"],
        "authorized_groups": ["left_arm", "right_arm"],
        "pressed_controls": ["right.primary"],
        "hold_progress": {"right.primary": 0.65},
        "neutral": True,
        "motion_started": False,
        "published_actions": 0,
        "last_fault": "",
        "source_error": "",
    }


def test_status_exposes_rollout_intervention_source_for_vr_rl() -> None:
    config = console_config()
    config.collection.teleop = ConfigDict(
        control_source="client",
        client=ConfigDict(type="vr_webxr"),
    )
    config.rollout.intervention.source = "teleop_client"
    runtime, session = build_runtime(config)
    runtime.teleop_execution = TeleopExecutionState(
        control_source="client",
        client_type="vr_webxr",
        active=True,
    )
    runtime.teleop_client = type(
        "_TeleopClient",
        (),
        {
            "status": lambda self: TeleopStatus(
                source_type="vr_webxr",
                connected=True,
                neutral=True,
            )
        },
    )()
    try:
        status = _serialize_status(
            ConsoleContext(
                config=config,
                runtime=runtime,
                session=session,
                obs_reader=runtime.transport.create_observation_reader(),
            )
        )
    finally:
        runtime.transport.close()

    assert status["rollout_intervention_source"] == "teleop_client"


def test_config_exposes_client_teleop_identity_for_console_bootstrap():
    config = console_config()
    config.collection.teleop = ConfigDict(
        control_source="client",
        client=ConfigDict(
            type="vr_webxr",
            arms=ConfigDict(
                left_arm=ConfigDict(controller="left"),
                right_arm=ConfigDict(controller="right", label="Tool Arm"),
            ),
        ),
    )
    serialized = _serialize_console_payload(_serialize_config, config=config)

    assert serialized["collection"]["teleop"] == {
        "control_source": "client",
        "client_type": "vr_webxr",
    }
    assert serialized["collection"]["controls"]["mode"] == "vr"
    assert serialized["collection"]["controls"]["bindings"]["record_cancel"] == {
        "control": "right.primary",
        "key": "A",
        "gesture": "hold",
        "hold_ms": 1000,
    }
    assert serialized["collection"]["controls"]["bindings"]["left_arm_toggle"] == {
        "control": "left.grip",
        "key": "L GRIP",
        "gesture": "hold",
        "hold_ms": 1000,
    }
    assert serialized["collection"]["controls"]["bindings"]["right_arm_toggle"] == {
        "control": "right.grip",
        "key": "R GRIP",
        "gesture": "hold",
        "hold_ms": 1000,
    }
    assert serialized["collection"]["controls"]["groups"] == [
        {
            "id": "left_arm",
            "label": "LEFT ARM",
            "control": "left.grip",
            "binding": "left_arm_toggle",
        },
        {
            "id": "right_arm",
            "label": "TOOL ARM",
            "control": "right.grip",
            "binding": "right_arm_toggle",
        },
    ]


def test_config_exposes_keyboard_collection_controls_by_default():
    serialized = _serialize_console_payload(_serialize_config)

    controls = serialized["collection"]["controls"]
    assert controls["mode"] == "keyboard"
    assert controls["groups"] == []
    assert controls["bindings"]["motion"] == {
        "control": "KeyM",
        "key": "M",
        "gesture": "tap",
    }


def test_frame_uses_live_camera_keys_when_reader_reports_them():
    class _Reader:
        def get_latest_qpos(self):
            return np.zeros(14, dtype=np.float32)

        def get_camera_keys(self):
            return ["cam_high"]

    frame = _serialize_console_payload(
        _serialize_frame,
        obs_reader=_Reader(),  # type: ignore[reportArgumentType]
    )

    assert frame["cameras"] == ["cam_high"]


def test_frame_uses_replay_observation_only_on_replay_tab():
    config = console_config()
    runtime, session = build_runtime(config)
    live_qpos = np.zeros(14, dtype=np.float32)
    replay_qpos = np.ones(14, dtype=np.float32)

    class _Reader:
        def get_latest_qpos(self):
            return live_qpos

        def get_camera_keys(self):
            return ["cam_high"]

    class _ReplaySource:
        def get_scene_qpos(self):
            return replay_qpos

        def get_camera_keys(self):
            return ["replay_cam"]

    try:
        runtime.replay_source = _ReplaySource()  # type: ignore[reportAttributeAccessIssue]
        ctx = ConsoleContext(config=config, runtime=runtime, session=session, obs_reader=_Reader())  # type: ignore[reportArgumentType]

        ctx.active_tab = "replay"
        frame = _serialize_frame(ctx)
        np.testing.assert_allclose(frame["qpos"], replay_qpos)
        assert frame["cameras"] == ["replay_cam"]

        for tab in ("debug", "collect"):
            ctx.active_tab = tab
            frame = _serialize_frame(ctx)
            np.testing.assert_allclose(frame["qpos"], live_qpos)
            assert frame["cameras"] == ["cam_high"]
    finally:
        runtime.transport.close()


def test_unknown_camera_key_404s(console: WebHarness):
    assert console.get("/api/camera/no_such_cam").status == 404


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


def test_request_line_client_disconnect_is_silent(monkeypatch):
    def disconnect(_self):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", disconnect)

    ConsoleRequestHandler.handle(object.__new__(ConsoleRequestHandler))


def test_request_handler_does_not_hide_server_errors(monkeypatch):
    def fail(_self):
        raise RuntimeError("boom")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", fail)

    with pytest.raises(RuntimeError, match="boom"):
        ConsoleRequestHandler.handle(object.__new__(ConsoleRequestHandler))


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


def test_camera_jpeg_payload_prefers_reader_jpeg(monkeypatch):
    payload = b"\xff\xd8raw-jpeg\xff\xd9"

    class _Reader:
        def get_camera_jpeg(self, key: str):
            assert key == "cam_high"
            return payload

        def get_camera_frame(self, _key: str):
            raise AssertionError("raw JPEG path should not decode frames")

    def fail_encode(_image, _convert_bgr_to_rgb):
        raise AssertionError("raw JPEG path should not encode frames")

    monkeypatch.setattr("core.app.console.server._encode_jpeg", fail_encode)

    jpeg, sig = _camera_jpeg_payload(_Reader(), "cam_high", convert_bgr_to_rgb=True)

    assert jpeg == payload
    assert sig is not None


def test_camera_jpeg_payload_distinguishes_frames_with_equal_sampled_sum(monkeypatch):
    first = np.zeros((64, 64, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    first[0, 0, 0] = 10
    second[0, 0, 1] = 10
    assert first[::32, ::32].sum() == second[::32, ::32].sum()

    class _Reader:
        frame = first

        def get_camera_frame(self, _key: str):
            return self.frame

    monkeypatch.setattr(
        "core.app.console.server._encode_jpeg",
        lambda image, _convert: np.asarray(image).tobytes(),
    )
    reader = _Reader()
    _, first_sig = _camera_jpeg_payload(reader, "cam_right_wrist", False)
    reader.frame = second
    _, second_sig = _camera_jpeg_payload(reader, "cam_right_wrist", False)

    assert first_sig != second_sig
