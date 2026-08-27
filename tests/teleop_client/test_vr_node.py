from __future__ import annotations

import io
import logging
import threading

import numpy as np
import pytest

import examples.input_sources.vr_webxr.node as node_module
from examples.input_sources.vr_webxr.node import (
    FrameConsoleMonitor,
    GripToggleEncoder,
    OperatorEventMapper,
    WebXrNode,
    ZmqBridge,
    _PendingEvent,
    normalize_browser_frame,
    normalize_controller,
)


def test_frame_console_monitor_overwrites_one_line_and_tracks_receive_health() -> None:
    output = io.StringIO()
    monitor = FrameConsoleMonitor(interval_s=0.001, stream=output)
    monitor.set_rtt(0.0123)

    encoder = GripToggleEncoder()
    raw_first, face_first = normalize_browser_frame(_frame(10, timestamp=100.0), session_id="s")
    raw_second, face_second = normalize_browser_frame(_frame(12, timestamp=110.0), session_id="s")
    monitor.observe(encoder.encode(raw_first, face_first), arrival_ms=1000.0)
    monitor.observe(encoder.encode(raw_second, face_second), arrival_ms=1015.0)
    monitor.finish()

    rendered = output.getvalue()
    assert rendered.count("\r") == 2
    assert rendered.endswith("\n")
    assert "seq=12" in rendered
    assert "rtt=12.3ms" in rendered
    assert "relative_delay=5.0ms" in rendered
    assert "dropped=1" in rendered
    assert "L=(+0.10,+0.20,+0.30) t=0.00 off" in rendered


def test_frame_console_monitor_can_be_disabled() -> None:
    output = io.StringIO()
    monitor = FrameConsoleMonitor(interval_s=0.0, stream=output)

    monitor.observe(_frame(0), arrival_ms=1000.0)
    monitor.finish()

    assert output.getvalue() == ""


def test_webxr_page_attaches_visible_geometry_to_both_controller_grips() -> None:
    source = (node_module.STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "new THREE.BoxGeometry" in source
    assert "renderer.xr.getControllerGrip(index)" in source
    assert "addControllerGrip(0);" in source
    assert "addControllerGrip(1);" in source
    assert 'grip.addEventListener("connected"' in source
    assert 'grip.addEventListener("disconnected"' in source
    assert 'payload.type === "haptic"' in source
    assert "function pulseController(hand, intensity, durationMs)" in source
    assert "function updateHapticDiagnostics(reason)" in source
    assert "function drawHapticHud(text)" in source
    assert "new THREE.CanvasTexture(hapticHudCanvas)" in source
    assert "hapticActuators" in source
    assert 'id="haptic-status"' in (node_module.STATIC_ROOT / "index.html").read_text(
        encoding="utf-8"
    )


def _buttons(*pressed: int) -> list[dict[str, object]]:
    return [
        {
            "pressed": index in pressed,
            "touched": index in pressed,
            "value": 1.0 if index in pressed else 0.0,
        }
        for index in range(6)
    ]


def _controller(hand: str, *, pressed: tuple[int, ...] = ()) -> dict[str, object]:
    return {
        "valid": True,
        "position": [0.1, 0.2, 0.3],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "profiles": ["pico-4-ultra" if hand == "left" else "oculus-touch-v3"],
        "buttons": _buttons(*pressed),
        "axes": [],
    }


def _frame(seq: int, *, left=(), right=(), timestamp=100.0) -> dict[str, object]:
    return {
        "type": "frame",
        "version": 1,
        "seq": seq,
        "client_time_ms": timestamp,
        "reference_space": "local-floor",
        "controllers": {
            "left": _controller("left", pressed=left),
            "right": _controller("right", pressed=right),
        },
    }


def _event(event_id: int) -> dict[str, object]:
    return {
        "protocol": "eva.teleop.vr",
        "version": 1,
        "type": "event",
        "session_id": "session",
        "event_id": event_id,
        "client_time_ms": 100.0,
        "intent": "record_toggle",
    }


def test_node_normalizes_pico_and_quest_profiles() -> None:
    normalized, face = normalize_controller("right", _controller("right", pressed=(0, 1, 4)))

    assert normalized["trigger"] == 1.0
    assert normalized["profiles"] == ["oculus-touch-v3"]
    assert face == {"primary": True, "secondary": False, "grip": True}


def test_node_treats_digital_pico_trigger_press_as_full_trigger_value() -> None:
    controller = _controller("left")
    controller["buttons"][0] = {"pressed": True, "touched": True, "value": 0.0}

    normalized, _ = normalize_controller("left", controller)

    assert normalized["trigger"] == 1.0


def test_node_generates_short_toggle_and_long_cancel_events() -> None:
    mapper = OperatorEventMapper(long_press_ms=1000.0)
    mapper.reset()
    first, face = normalize_browser_frame(_frame(0), session_id="session")
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=100.0,
            controllers=first["controllers"],
            face_buttons=face,
        )
        == ()
    )

    pressed, face = normalize_browser_frame(
        _frame(1, right=(4,), timestamp=200.0), session_id="session"
    )
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=200.0,
            controllers=pressed["controllers"],
            face_buttons=face,
        )
        == ()
    )
    released, face = normalize_browser_frame(_frame(2, timestamp=300.0), session_id="session")
    events = mapper.update(
        session_id="session",
        client_time_ms=300.0,
        controllers=released["controllers"],
        face_buttons=face,
    )
    assert [(event["event_id"], event["intent"]) for event in events] == [(0, "record_toggle")]

    pressed, face = normalize_browser_frame(
        _frame(3, right=(4,), timestamp=400.0), session_id="session"
    )
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=400.0,
            controllers=pressed["controllers"],
            face_buttons=face,
        )
        == ()
    )
    held, face = normalize_browser_frame(
        _frame(4, right=(4,), timestamp=1500.0), session_id="session"
    )
    events = mapper.update(
        session_id="session",
        client_time_ms=1500.0,
        controllers=held["controllers"],
        face_buttons=face,
    )
    assert [(event["event_id"], event["intent"]) for event in events] == [(1, "record_cancel")]
    released, face = normalize_browser_frame(_frame(5, timestamp=1600.0), session_id="session")
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=1600.0,
            controllers=released["controllers"],
            face_buttons=face,
        )
        == ()
    )


def test_node_preserves_b_arm_toggle_event_flow() -> None:
    mapper = OperatorEventMapper()

    pressed, face = normalize_browser_frame(
        _frame(0, right=(5,), timestamp=100.0), session_id="session"
    )
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=100.0,
            controllers=pressed["controllers"],
            face_buttons=face,
        )
        == ()
    )

    released, face = normalize_browser_frame(_frame(1, timestamp=200.0), session_id="session")
    events = mapper.update(
        session_id="session",
        client_time_ms=200.0,
        controllers=released["controllers"],
        face_buttons=face,
    )
    assert [(event["event_id"], event["intent"]) for event in events] == [(0, "arm_toggle")]


def test_node_output_is_vendor_neutral_and_session_scoped() -> None:
    normalized, face = normalize_browser_frame(_frame(7), session_id="abc")
    encoded = GripToggleEncoder().encode(normalized, face)

    assert encoded["protocol"] == "eva.teleop.vr"
    assert encoded["version"] == 3
    assert encoded["session_id"] == "abc"
    assert encoded["seq"] == 7
    left = encoded["controllers"]["left"]
    assert set(left) == {
        "valid",
        "grip_engaged",
        "position",
        "orientation_xyzw",
        "trigger",
        "profiles",
    }
    assert "squeeze" not in left
    assert left["grip_engaged"] is False


def test_grip_long_press_toggles_each_hand_without_changing_absolute_pose() -> None:
    encoder = GripToggleEncoder(long_press_ms=1000.0)

    first, face = normalize_browser_frame(_frame(0), session_id="session")
    encoded = encoder.encode(first, face)
    assert encoded["controllers"]["left"]["grip_engaged"] is False

    pressed, face = normalize_browser_frame(
        _frame(1, left=(1,), right=(1,), timestamp=100.0), session_id="session"
    )
    encoded = encoder.encode(pressed, face)
    assert encoded["controllers"]["left"]["grip_engaged"] is False
    assert encoded["controllers"]["right"]["grip_engaged"] is False
    feedback = encoder.input_feedback(600.0)
    assert set(feedback["pressed"]) == {"left.grip", "right.grip"}
    assert feedback["hold_progress"] == {"left.grip": 0.5, "right.grip": 0.5}

    activated, face = normalize_browser_frame(
        _frame(2, left=(1,), right=(1,), timestamp=1100.0), session_id="session"
    )
    encoded = encoder.encode(activated, face)
    assert encoded["controllers"]["left"]["grip_engaged"] is True
    assert encoded["controllers"]["right"]["grip_engaged"] is True
    np.testing.assert_allclose(encoded["controllers"]["left"]["position"], [0.1, 0.2, 0.3])

    moved_raw = _frame(3, timestamp=1200.0)
    moved_raw["controllers"]["left"]["position"] = [0.3, 0.2, 0.3]
    moved, face = normalize_browser_frame(moved_raw, session_id="session")
    encoded = encoder.encode(moved, face)
    np.testing.assert_allclose(encoded["controllers"]["left"]["position"], [0.3, 0.2, 0.3])

    released, face = normalize_browser_frame(_frame(4, timestamp=2200.0), session_id="session")
    encoder.encode(released, face)
    pressed_again, face = normalize_browser_frame(
        _frame(5, left=(1,), timestamp=3200.0), session_id="session"
    )
    encoder.encode(pressed_again, face)
    deactivated, face = normalize_browser_frame(
        _frame(6, left=(1,), timestamp=4200.0), session_id="session"
    )
    encoded = encoder.encode(deactivated, face)
    assert encoded["controllers"]["left"]["grip_engaged"] is False

    released, face = normalize_browser_frame(_frame(7, timestamp=4300.0), session_id="session")
    encoder.encode(released, face)


def test_grip_toggle_reports_only_the_hand_that_changed() -> None:
    encoder = GripToggleEncoder(long_press_ms=1000.0)

    first, face = normalize_browser_frame(_frame(0), session_id="session")
    encoder.encode(first, face)
    assert encoder.consume_toggles() == ()

    held, face = normalize_browser_frame(
        _frame(1, left=(1,), timestamp=1000.0), session_id="session"
    )
    encoder.encode(held, face)
    assert encoder.consume_toggles() == ()

    toggled, face = normalize_browser_frame(
        _frame(2, left=(1,), timestamp=2000.0), session_id="session"
    )
    encoder.encode(toggled, face)
    assert encoder.consume_toggles() == ("left",)
    assert encoder.consume_toggles() == ()


def test_node_ignores_y_and_unmapped_buttons() -> None:
    mapper = OperatorEventMapper()
    for button in (2, 3):
        frame, face = normalize_browser_frame(
            _frame(0, right=(button,), timestamp=100.0), session_id="session"
        )
        assert (
            mapper.update(
                session_id="session",
                client_time_ms=100.0,
                controllers=frame["controllers"],
                face_buttons=face,
            )
            == ()
        )

    frame, face = normalize_browser_frame(
        _frame(1, left=(5,), timestamp=200.0), session_id="session"
    )
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=200.0,
            controllers=frame["controllers"],
            face_buttons=face,
        )
        == ()
    )


def test_node_generates_home_on_left_x_release() -> None:
    mapper = OperatorEventMapper()
    pressed, face = normalize_browser_frame(
        _frame(0, left=(4,), timestamp=100.0), session_id="session"
    )
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=100.0,
            controllers=pressed["controllers"],
            face_buttons=face,
        )
        == ()
    )
    released, face = normalize_browser_frame(_frame(1, timestamp=200.0), session_id="session")
    events = mapper.update(
        session_id="session",
        client_time_ms=200.0,
        controllers=released["controllers"],
        face_buttons=face,
    )
    assert [(event["event_id"], event["intent"]) for event in events] == [(0, "home")]


def test_grip_does_not_block_a_record_toggle() -> None:
    mapper = OperatorEventMapper()
    pressed, face = normalize_browser_frame(
        _frame(0, left=(1,), right=(1, 4), timestamp=100.0), session_id="session"
    )
    assert (
        mapper.update(
            session_id="session",
            client_time_ms=100.0,
            controllers=pressed["controllers"],
            face_buttons=face,
        )
        == ()
    )
    released, face = normalize_browser_frame(
        _frame(1, left=(1,), right=(1,), timestamp=200.0), session_id="session"
    )
    events = mapper.update(
        session_id="session",
        client_time_ms=200.0,
        controllers=released["controllers"],
        face_buttons=face,
    )
    assert [event["intent"] for event in events] == ["record_toggle"]


def test_zmq_bridge_reports_bind_failure() -> None:
    bridge = ZmqBridge("tcp://127.0.0.1:18779", "tcp://127.0.0.1:18779")

    with pytest.raises(RuntimeError, match="failed to start"):
        bridge.start()
    bridge.close()


def test_pending_event_retry_is_scoped_to_live_session_and_ttl() -> None:
    pending = _PendingEvent(
        payload={"session_id": "session-a", "event_id": 0},
        created_at=10.0,
    )

    assert pending.should_retry(
        browser_connected=True,
        session_id="session-a",
        now=10.5,
        retry_ttl_s=1.0,
    )
    assert not pending.should_retry(
        browser_connected=True,
        session_id="session-b",
        now=10.5,
        retry_ttl_s=1.0,
    )
    assert not pending.should_retry(
        browser_connected=False,
        session_id="session-a",
        now=10.5,
        retry_ttl_s=1.0,
    )
    assert not pending.should_retry(
        browser_connected=True,
        session_id="session-a",
        now=11.1,
        retry_ttl_s=1.0,
    )


def test_zmq_bridge_rejects_invalid_event_retry_ttl() -> None:
    with pytest.raises(ValueError, match="event_retry_ttl_s"):
        ZmqBridge(
            "tcp://127.0.0.1:18777",
            "tcp://127.0.0.1:18778",
            event_retry_ttl_s=0.0,
        )


def test_zmq_bridge_rejects_invalid_queue_capacities() -> None:
    with pytest.raises(ValueError, match="event_queue_maxsize"):
        ZmqBridge(
            "tcp://127.0.0.1:18777",
            "tcp://127.0.0.1:18778",
            event_queue_maxsize=0,
        )
    with pytest.raises(ValueError, match="acknowledgement_queue_maxsize"):
        ZmqBridge(
            "tcp://127.0.0.1:18777",
            "tcp://127.0.0.1:18778",
            acknowledgement_queue_maxsize=0,
        )


def test_zmq_bridge_bounded_event_queue_rejects_flood_and_ack_queue_keeps_latest() -> None:
    bridge = ZmqBridge(
        "tcp://127.0.0.1:18777",
        "tcp://127.0.0.1:18778",
        event_queue_maxsize=2,
        acknowledgement_queue_maxsize=2,
    )
    try:
        assert bridge.submit_event(_event(0)) is True
        assert bridge.submit_event(_event(1)) is True
        assert bridge.submit_event(_event(2)) is False
        assert bridge._new_events.qsize() == 2
        assert "event queue full" in bridge._source_error

        for event_id in range(5):
            bridge._store_acknowledgement({"event_id": event_id})
        assert bridge.acknowledgements.qsize() == 2
        assert [bridge.acknowledgements.get_nowait()["event_id"] for _ in range(2)] == [3, 4]
    finally:
        bridge.close()


def test_zmq_bridge_close_timeout_retains_worker_and_rejects_reentry(monkeypatch) -> None:
    bridge = ZmqBridge("tcp://127.0.0.1:18777", "tcp://127.0.0.1:18778")
    release = threading.Event()
    old_worker = threading.Thread(target=release.wait, daemon=True)
    old_worker.start()
    bridge._thread = old_worker
    bridge._stop.set()
    monkeypatch.setattr(node_module, "_BRIDGE_STOP_TIMEOUT_S", 0.01)

    with pytest.raises(RuntimeError, match="timed out stopping"):
        bridge.close()
    assert bridge._thread is old_worker
    assert bridge._stop.is_set()
    with pytest.raises(RuntimeError, match="previous worker is stopping"):
        bridge.start()

    release.set()
    old_worker.join(timeout=1.0)

    def fake_run() -> None:
        bridge._ready.set()
        bridge._stop.wait()

    monkeypatch.setattr(bridge, "_run", fake_run)
    bridge.set_browser(True, "old-session", "old error")
    bridge.submit_frame({"seq": 1})
    bridge.submit_event(_event(1))
    bridge._store_acknowledgement({"event_id": 1})
    bridge.start()

    assert bridge._thread is not None and bridge._thread.is_alive()
    assert bridge._frames.empty()
    assert bridge._new_events.empty()
    assert bridge.acknowledgements.empty()
    assert bridge._browser_connected is False
    assert bridge._session_id == ""
    assert bridge._source_error == ""
    bridge.close()


def test_remote_webxr_without_tls_emits_secure_context_warning(caplog) -> None:
    bridge = ZmqBridge("tcp://127.0.0.1:18777", "tcp://127.0.0.1:18778")
    node = WebXrNode(
        host="0.0.0.0",
        port=8443,
        token="test",
        public_host="vr.example.com",
        tls_cert="",
        tls_key="",
        bridge=bridge,
    )
    try:
        with caplog.at_level(logging.WARNING, logger="eva.vr_webxr_node"):
            node._warn_if_remote_without_tls()
    finally:
        bridge.close()

    assert "HTTPS/WSS" in caplog.text
