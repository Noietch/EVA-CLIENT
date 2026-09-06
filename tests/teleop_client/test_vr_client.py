from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest

import teleop_client.vr.client as vr_client_module
from examples.input_sources.vr_webxr.node import ZmqBridge
from teleop_client.base import (
    CanonicalEefCommand,
    TeleopClientError,
    TeleopContext,
    TeleopResultKind,
)
from teleop_client.vr.client import (
    VrTeleopClient,
)

pytestmark = pytest.mark.integration


def _frame(seq: int = 0, *, session: str = "s", squeeze: float = 1.0) -> dict:
    return {
        "protocol": "eva.teleop.vr",
        "version": 3,
        "type": "frame",
        "session_id": session,
        "seq": seq,
        "client_time_ms": 10.0,
        "reference_space": "local-floor",
        "controllers": {
            "left": {
                "valid": True,
                "grip_engaged": True,
                "position": [seq * 0.1, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "trigger": 0.0,
                "profiles": ["pico-4-ultra"],
            },
            "right": {"valid": False, "grip_engaged": False, "trigger": 0.0, "profiles": []},
        },
    }


def _event(event_id: int, *, session: str = "s", intent: str = "record_toggle") -> dict:
    return {
        "protocol": "eva.teleop.vr",
        "version": 3,
        "type": "event",
        "session_id": session,
        "event_id": event_id,
        "client_time_ms": 10.0,
        "intent": intent,
    }


def _client(
    *,
    workspace: dict[str, list[float]] | None = None,
) -> VrTeleopClient:
    arm: dict[str, object] = {
        "group_name": "left_arm",
        "controller": "left",
        "base_from_xr_rotation": np.eye(3),
        "position_scale": 0.5,
        "gripper": {
            "mode": "binary",
            "threshold": 0.5,
            "open_value": 1.0,
            "close_value": 0.0,
        },
    }
    if workspace is not None:
        arm["workspace"] = workspace
    return VrTeleopClient(
        endpoint="tcp://127.0.0.1:18765",
        ack_endpoint="tcp://127.0.0.1:18766",
        input_timeout_s=0.25,
        arms=[arm],
    )


def _context(now: float | None = None) -> TeleopContext:
    return TeleopContext(
        measured_qpos=np.zeros(7, dtype=np.float32),
        measured_eef=np.asarray([0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0]),
        now=time.monotonic() if now is None else now,
    )


def _tcp_endpoint() -> str:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{listener.getsockname()[1]}"


def test_client_result_token_rejects_reset_disconnect_and_source_timeout(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(vr_client_module.time, "monotonic", lambda: clock[0])
    client = _client()
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    result = client.poll(_context(now=10.0))
    assert client.validate_result(result)

    clock[0] = 10.3
    assert not client.validate_result(result)

    clock[0] = 10.0
    client.reset()
    assert not client.validate_result(result)

    client._ingest(json.dumps(_frame(1, squeeze=0.0)).encode())
    fresh = client.poll(_context(now=10.0))
    client._ingest(
        {
            "protocol": "eva.teleop.vr",
            "version": 3,
            "type": "heartbeat",
            "session_id": "s",
            "browser_connected": False,
            "source_error": "headset disconnected",
        }
    )
    assert not client.validate_result(fresh)


def test_client_connection_requires_fresh_vr_frames(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(vr_client_module.time, "monotonic", lambda: clock[0])
    client = _client()

    client._ingest(json.dumps(_frame()).encode())
    assert client.status().connected

    clock[0] = 10.3
    assert not client.status().connected


def test_client_uses_configured_robot_pose_as_vr_home(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(vr_client_module.time, "monotonic", lambda: clock[0])
    client = _client()
    configured_home = np.asarray(
        [0.8, -0.2, 0.6, 0.9238795, 0.0, 0.3826834, 0.0, 1.0],
        dtype=np.float32,
    )
    client.set_home_eef({"left_arm": configured_home})

    active_frame = _frame(squeeze=0.0)
    active_frame["controllers"]["left"]["position"] = [0.0, 0.0, 0.0]
    client._ingest(json.dumps(active_frame).encode())
    result = client.poll(_context(now=10.0))

    assert result.command is not None
    np.testing.assert_allclose(result.command.value[:3], configured_home[:3], atol=1e-6)
    np.testing.assert_allclose(
        result.command.value[3:7],
        configured_home[3:7] / np.linalg.norm(configured_home[3:7]),
        atol=1e-6,
    )

    client.reset()
    client._ingest(json.dumps(_frame(1, squeeze=0.0)).encode())
    reset_result = client.poll(_context(now=10.0))
    assert reset_result.command is not None
    np.testing.assert_allclose(reset_result.command.value[:3], configured_home[:3], atol=1e-6)


def test_client_event_flood_rejects_and_fails_closed() -> None:
    client = _client()
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    for event_id in range(64):
        client._ingest(json.dumps(_event(event_id)).encode())

    assert len(client._events) == 64
    client._ingest(json.dumps(_event(64)).encode())

    assert client._stop.is_set()
    assert client.drain_events() == ()
    ack = client._ack_queue.get_nowait()
    assert ack["accepted"] is False
    assert "event buffer full" in str(ack["message"])
    assert "event buffer full" in client.status().source_error


def test_client_shutdown_is_bounded_without_ack_receiver(monkeypatch) -> None:
    endpoint = _tcp_endpoint()
    ack_endpoint = _tcp_endpoint()
    while ack_endpoint == endpoint:
        ack_endpoint = _tcp_endpoint()
    client = VrTeleopClient(
        endpoint=endpoint,
        ack_endpoint=ack_endpoint,
        input_timeout_s=0.25,
        arms=[
            {
                "group_name": "left_arm",
                "controller": "left",
                "base_from_xr_rotation": np.eye(3),
                "gripper": {"mode": "binary", "threshold": 0.5},
            }
        ],
    )
    monkeypatch.setattr(vr_client_module, "_STOP_TIMEOUT_S", 1.0)
    client.start()
    running_thread = client._thread
    client.start()
    assert client._thread is running_thread
    with client._lock:
        for event_id in range(200):
            client._enqueue_ack_locked({"event_id": event_id})
    time.sleep(0.1)

    started = time.monotonic()
    client.close()

    assert time.monotonic() - started < 1.0
    assert client._thread is None
    assert client.status().connected is False
    assert client._ack_queue.empty()
    client.close()


def test_external_zmq_bridge_reaches_vr_client() -> None:
    endpoint = _tcp_endpoint()
    ack_endpoint = _tcp_endpoint()
    while ack_endpoint == endpoint:
        ack_endpoint = _tcp_endpoint()
    bridge = ZmqBridge(endpoint, ack_endpoint)
    client = VrTeleopClient(
        endpoint=endpoint,
        ack_endpoint=ack_endpoint,
        input_timeout_s=0.5,
        arms=[
            {
                "group_name": "left_arm",
                "controller": "left",
                "base_from_xr_rotation": np.eye(3),
                "position_scale": 0.5,
                "gripper": {"mode": "binary", "threshold": 0.5},
            }
        ],
    )
    bridge.start()
    client.start()
    try:
        bridge.set_browser(True, "wire-session")
        neutral = _frame(session="wire-session", squeeze=0.0)
        neutral["controllers"]["left"]["grip_engaged"] = False
        neutral["controllers"]["left"]["trigger"] = 0.0
        neutral["input_feedback"] = {"pressed": [], "hold_progress": {}}
        bridge.submit_frame(neutral)
        deadline = time.monotonic() + 3.0
        while not client.status().connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.status().connected

        bridge.submit_frame(neutral)
        result = None
        while time.monotonic() < deadline:
            try:
                result = client.poll(_context())
            except TeleopClientError:
                time.sleep(0.01)
                continue
            break
        assert result is not None and result.kind is TeleopResultKind.IDLE

        bridge.submit_frame(_frame(1, session="wire-session", squeeze=0.0))
        command = None
        while command is None and time.monotonic() < deadline:
            command = client.poll(_context()).command
            if command is None:
                time.sleep(0.01)
        assert isinstance(command, CanonicalEefCommand)

        bridge.submit_frame(_frame(2, session="wire-session", squeeze=1.0))
        command = None
        while command is None and time.monotonic() < deadline:
            command = client.poll(_context()).command
            if command is None:
                time.sleep(0.01)
        assert isinstance(command, CanonicalEefCommand)

        bridge.submit_event(_event(99, session="stale-session"))
        time.sleep(0.3)
        assert client.drain_events() == ()

        bridge.submit_event(_event(0, session="wire-session"))
        deadline = time.monotonic() + 3.0
        events = ()
        while not events and time.monotonic() < deadline:
            events = client.drain_events()
            if not events:
                time.sleep(0.01)
        assert len(events) == 1
        client.acknowledge_event(events[0], accepted=True, message="started")
        ack = bridge.acknowledgements.get(timeout=3.0)
        assert ack["session_id"] == "wire-session"
        assert ack["event_id"] == 0
        assert ack["accepted"] is True
    finally:
        client.close()
        bridge.close()
