from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import numpy as np
import pytest

import teleop_client.vr.client as vr_client_module
from examples.input_sources.vr_webxr.node import ZmqBridge
from teleop_client import build_client, validate_client_config
from teleop_client.base import (
    CanonicalEefCommand,
    TeleopClientError,
    TeleopContext,
    TeleopResult,
    TeleopResultKind,
)
from teleop_client.vr.client import (
    VrProtocolError,
    VrTeleopClient,
    parse_operator_event,
    parse_vr_pose_frame,
)


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


def _vr_client_config() -> dict[str, Any]:
    return {
        "type": "vr_webxr",
        "endpoint": "tcp://127.0.0.1:18765",
        "ack_endpoint": "tcp://127.0.0.1:18766",
        "input_timeout_s": 0.25,
        "heartbeat_timeout_s": 2.0,
        "base_from_xr_rotation": np.eye(3),
        "position_scale": 0.5,
        "gripper": {
            "mode": "binary",
            "threshold": 0.5,
            "open_value": 1.0,
            "close_value": 0.0,
        },
        "arms": {"left_arm": {"controller": "left"}},
    }


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


def test_parsers_validate_version_session_and_intent() -> None:
    raw_frame = _frame()
    raw_frame["input_feedback"] = {
        "pressed": ["right.primary", "left.grip"],
        "hold_progress": {"right.primary": 0.65, "left.grip": 0.25},
    }
    frame = parse_vr_pose_frame(raw_frame, received_at=1.0)
    event = parse_operator_event(_event(2, intent="record_toggle"), received_at=2.0)
    arm_event = parse_operator_event(_event(4, intent="arm_toggle"), received_at=2.0)
    home_event = parse_operator_event(_event(5, intent="home"), received_at=2.0)
    intervention_event = parse_operator_event(
        _event(6, intent="intervention_toggle"), received_at=2.0
    )

    assert frame.session_id == "s"
    assert frame.controllers["left"].profiles == ("pico-4-ultra",)
    assert frame.pressed_controls == ("right.primary", "left.grip")
    assert dict(frame.hold_progress) == {"right.primary": 0.65, "left.grip": 0.25}
    assert event.event_id == 2
    assert event.intent == "record_toggle"
    assert arm_event.intent == "arm_toggle"
    assert home_event.intent == "home"
    assert intervention_event.intent == "intervention_toggle"
    with pytest.raises(VrProtocolError, match="unsupported operator intent"):
        parse_operator_event(_event(3, intent="scene_reset"), received_at=2.0)
    invalid = _frame()
    invalid["version"] = 9
    with pytest.raises(VrProtocolError, match="version"):
        parse_vr_pose_frame(invalid)


def test_teleop_result_requires_an_explicit_valid_outcome() -> None:
    command = CanonicalEefCommand(np.zeros(8, dtype=np.float32))

    assert TeleopResult.idle().kind is TeleopResultKind.IDLE
    assert TeleopResult.rejected("outside workspace").kind is TeleopResultKind.REJECTED
    assert TeleopResult.from_command(command).kind is TeleopResultKind.COMMAND
    with pytest.raises(ValueError, match="exactly one command"):
        TeleopResult(TeleopResultKind.COMMAND)
    with pytest.raises(ValueError, match="requires a message"):
        TeleopResult(TeleopResultKind.REJECTED)


def test_client_produces_canonical_eef_without_robot_dependencies() -> None:
    client = _client()
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    first = client.poll(_context())
    client._ingest(json.dumps(_frame(1)).encode())
    moved = client.poll(_context())

    assert isinstance(first.command, CanonicalEefCommand)
    np.testing.assert_allclose(first.command.value, _context().measured_eef)
    assert isinstance(moved.command, CanonicalEefCommand)
    np.testing.assert_allclose(moved.command.value[:3], [0.45, 0.1, 0.3], atol=1e-6)


def test_client_result_token_survives_fresh_frame_in_same_health_session() -> None:
    client = _client()
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    result = client.poll(_context())

    assert result.source_token is not None
    assert client.validate_result(result)
    client._ingest(json.dumps(_frame(1, squeeze=0.0)).encode())
    assert client.validate_result(result)

    client._ingest(json.dumps(_frame(0, session="new", squeeze=0.0)).encode())
    assert not client.validate_result(result)


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


def test_generic_factory_applies_validated_per_arm_override() -> None:
    config = _vr_client_config()
    config["arms"]["left_arm"]["position_scale"] = 0.25
    client = build_client(config, arm_group_names=("left_arm",))
    assert isinstance(client, VrTeleopClient)

    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    assert client.poll(_context()).command is not None
    client._ingest(json.dumps(_frame(1)).encode())
    moved = client.poll(_context())

    assert isinstance(moved.command, CanonicalEefCommand)
    np.testing.assert_allclose(moved.command.value[:3], [0.425, 0.1, 0.3], atol=1e-6)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("position_scale", -1.0, "position_scale"),
        ("eef_filter_alpha", 0.0, "eef_filter_alpha"),
        ("base_from_xr_rotation", np.zeros((3, 3)), "base_from_xr_rotation"),
        ("workspace", {"min": [0, 0, 0], "max": [0, 1, 1]}, "workspace"),
        ("gripper", {"threshold": 1.5}, "gripper.threshold"),
    ],
)
def test_vr_validator_rejects_invalid_per_arm_override(field, value, match) -> None:
    config = _vr_client_config()
    config["arms"]["left_arm"][field] = value

    with pytest.raises(ValueError, match=match):
        validate_client_config(config, arm_group_names=("left_arm",))


def test_generic_validator_rejects_unknown_client_type() -> None:
    with pytest.raises(ValueError, match="unsupported teleop client type"):
        validate_client_config({"type": "space_mouse"}, arm_group_names=())


def test_generic_validator_requires_client_object() -> None:
    with pytest.raises(ValueError, match="client must be an object"):
        validate_client_config([], arm_group_names=())


def test_vr_validator_requires_configured_arms_to_match_robot() -> None:
    with pytest.raises(ValueError, match="expected arm groups"):
        validate_client_config(
            _vr_client_config(),
            arm_group_names=("left_arm", "right_arm"),
        )


def test_client_rejects_stale_input_and_reanchors_after_reset() -> None:
    client = _client()
    now = time.monotonic()
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    assert client.poll(_context(now)).command is not None
    client._ingest(json.dumps(_frame(1, squeeze=1.0)).encode())
    assert client.poll(_context(now)).command is not None

    client.reset(require_neutral=True)
    client._ingest(json.dumps(_frame(2, squeeze=1.0)).encode())
    reanchored = client.poll(_context(now))
    assert isinstance(reanchored.command, CanonicalEefCommand)
    np.testing.assert_allclose(reanchored.command.value, _context().measured_eef)
    with pytest.raises(TeleopClientError, match="stale"):
        client.poll(_context(now=now + 1.0))


def test_workspace_rejection_drops_only_the_invalid_frame() -> None:
    client = _client(
        workspace={
            "min": [0.0, 0.0, 0.0],
            "max": [0.46, 1.0, 1.0],
        }
    )
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    assert client.poll(_context()).kind is TeleopResultKind.COMMAND
    client._ingest(json.dumps(_frame(1, squeeze=1.0)).encode())
    assert client.poll(_context()).kind is TeleopResultKind.COMMAND

    client._ingest(json.dumps(_frame(2, squeeze=1.0)).encode())
    rejected = client.poll(_context())
    assert rejected.kind is TeleopResultKind.REJECTED
    assert "outside workspace" in rejected.message
    assert client.status().engaged_groups == ()
    assert client.status().held_groups == ("left_arm",)

    recovered_frame = _frame(3, squeeze=1.0)
    recovered_frame["controllers"]["left"]["position"] = [0.0, 0.0, 0.0]
    client._ingest(json.dumps(recovered_frame).encode())
    recovered = client.poll(_context())
    assert recovered.kind is TeleopResultKind.COMMAND
    assert client.status().source_error == ""
    assert client.status().engaged_groups == ("left_arm",)


def test_workspace_rejection_drops_the_dual_arm_tick_without_resetting_anchors() -> None:
    client = VrTeleopClient(
        endpoint="tcp://127.0.0.1:18765",
        ack_endpoint="tcp://127.0.0.1:18766",
        input_timeout_s=0.25,
        arms=[
            {
                "group_name": "left_arm",
                "controller": "left",
                "base_from_xr_rotation": np.eye(3),
                "gripper": {"mode": "binary"},
            },
            {
                "group_name": "right_arm",
                "controller": "right",
                "base_from_xr_rotation": np.eye(3),
                "position_scale": 0.5,
                "workspace": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [0.46, 1.0, 1.0],
                },
                "gripper": {"mode": "binary"},
            },
        ],
    )
    measured_arm = np.asarray([0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0])
    context = TeleopContext(
        measured_qpos=np.zeros(14, dtype=np.float32),
        measured_eef=np.tile(measured_arm, 2),
        now=time.monotonic(),
    )

    def dual_frame(seq: int, *, squeeze: float, right_x: float) -> dict:
        payload = _frame(seq, squeeze=squeeze)
        payload["controllers"]["right"] = {
            "valid": True,
            "grip_engaged": True,
            "position": [right_x, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "profiles": ["pico-4-ultra"],
        }
        return payload

    client._ingest(json.dumps(dual_frame(0, squeeze=0.0, right_x=0.1)).encode())
    assert client.poll(context).kind is TeleopResultKind.COMMAND
    client._ingest(json.dumps(dual_frame(1, squeeze=1.0, right_x=0.2)).encode())
    assert client.poll(context).kind is TeleopResultKind.COMMAND
    client._ingest(json.dumps(dual_frame(2, squeeze=1.0, right_x=0.4)).encode())

    assert client.poll(context).kind is TeleopResultKind.REJECTED
    assert client.status().engaged_groups == ()
    assert client.status().held_groups == ("left_arm", "right_arm")


def test_client_deduplicates_events_by_session_and_acknowledges() -> None:
    client = _client()
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    for payload in (
        _event(1, intent="record_toggle"),
        _event(1, intent="record_toggle"),
        _event(0, intent="record_toggle"),
        _event(2, intent="record_cancel"),
    ):
        client._ingest(json.dumps(payload).encode())
    events = client.drain_events()

    assert [(event.event_id, event.intent) for event in events] == [
        (1, "record_toggle"),
        (2, "record_cancel"),
    ]
    client.acknowledge_event(events[0], accepted=True, message="started")
    ack = client._ack_queue.get_nowait()
    assert ack["session_id"] == "s"
    assert ack["event_id"] == 1
    assert ack["accepted"] is True


def test_client_rejects_event_from_non_current_session() -> None:
    client = _client()
    client._ingest(json.dumps(_frame(session="current", squeeze=0.0)).encode())
    client._ingest(json.dumps(_event(0, session="stale")).encode())

    assert client.drain_events() == ()
    ack = client._ack_queue.get_nowait()
    assert ack["session_id"] == "stale"
    assert ack["event_id"] == 0
    assert ack["accepted"] is False
    assert "active input session" in str(ack["message"])


def test_session_restart_accepts_sequence_zero_without_reanchoring() -> None:
    client = _client()
    client._ingest(json.dumps(_frame(4, session="old", squeeze=0.0)).encode())
    assert client.poll(_context()).command is not None
    client._ingest(json.dumps(_frame(5, session="old")).encode())
    assert client.poll(_context()).command is not None
    client._ingest(json.dumps(_event(0, session="old")).encode())
    client._ingest(json.dumps(_frame(0, session="new", squeeze=0.0)).encode())

    assert client.drain_events() == ()
    result = client.poll(_context())
    assert isinstance(result.command, CanonicalEefCommand)
    np.testing.assert_allclose(result.command.value[:3], _context().measured_eef[:3])


def test_client_start_timeout_stops_worker(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(vr_client_module, "_START_TIMEOUT_S", 0.01)
    monkeypatch.setattr(vr_client_module, "_STOP_TIMEOUT_S", 0.2)
    monkeypatch.setattr(client, "_receive_loop", lambda: client._stop.wait())

    with pytest.raises(RuntimeError, match="timed out starting"):
        client.start()

    assert client._thread is None


def test_client_start_propagates_worker_setup_error(monkeypatch) -> None:
    client = _client()

    def fail_during_setup() -> None:
        with client._lock:
            client._startup_error = ValueError("socket setup failed")
        client._ready.set()

    monkeypatch.setattr(client, "_receive_loop", fail_during_setup)

    with pytest.raises(RuntimeError, match="socket setup failed"):
        client.start()

    assert client._thread is None


def test_client_close_retains_live_worker_for_retry(monkeypatch) -> None:
    client = _client()
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    client._thread = thread
    monkeypatch.setattr(vr_client_module, "_STOP_TIMEOUT_S", 0.01)

    with pytest.raises(RuntimeError, match="timed out stopping"):
        client.close()

    assert client._thread is thread
    release.set()
    thread.join(timeout=1.0)
    client.close()
    assert client._thread is None


def test_client_rejects_reentry_while_close_timeout_worker_is_alive(monkeypatch) -> None:
    client = _client()
    release = threading.Event()
    old_thread = threading.Thread(target=release.wait, daemon=True)
    old_thread.start()
    client._thread = old_thread
    client._worker_ever_started = True
    client._stop.set()

    with pytest.raises(RuntimeError, match="previous worker is stopping"):
        client.start()

    release.set()
    old_thread.join(timeout=1.0)

    def fake_receive_loop() -> None:
        client._ready.set()
        client._stop.wait()

    monkeypatch.setattr(client, "_receive_loop", fake_receive_loop)
    client._ingest(json.dumps(_frame(session="old")).encode())
    client._ingest(json.dumps(_event(0, session="old")).encode())
    client._ack_queue.put_nowait({"event_id": 77})
    client.start()

    assert client._thread is not None and client._thread.is_alive()
    assert client._frame is None
    assert client._session_id == ""
    assert client._last_seq == -1
    assert client.drain_events() == ()
    assert client._event_cursor == {}
    assert client._ack_queue.empty()
    assert client.status().engaged_groups == ()
    client.close()


def test_client_poll_rejects_old_frame_after_worker_dies() -> None:
    client = _client()
    client._worker_ever_started = True
    dead_thread = threading.Thread(target=lambda: None, daemon=True)
    dead_thread.start()
    dead_thread.join(timeout=1.0)
    client._thread = dead_thread
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())

    with pytest.raises(TeleopClientError, match="worker is not running"):
        client.poll(_context())


def test_client_worker_failure_clears_motion_protocol_state() -> None:
    client = _client()
    client._worker_ever_started = True
    client._ingest(json.dumps(_frame(squeeze=0.0)).encode())
    client._ingest(json.dumps(_event(0)).encode())
    with client._lock:
        client._engaged_groups = ("left_arm",)

    def fail_worker() -> None:
        with client._lock:
            client._fail_closed_locked("VR worker exploded")

    worker = threading.Thread(target=fail_worker, daemon=True)
    client._thread = worker
    worker.start()
    worker.join(timeout=1.0)

    with pytest.raises(TeleopClientError, match="worker is not running"):
        client.poll(_context())
    assert client._frame is None
    assert client._session_id == ""
    assert client._last_seq == -1
    assert client.drain_events() == ()
    assert client.status().engaged_groups == ()
    assert "worker exploded" in client.status().source_error


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


def test_client_ack_flood_remains_bounded_and_keeps_latest() -> None:
    client = _client()
    with client._lock:
        for event_id in range(256):
            client._enqueue_ack_locked({"event_id": event_id})

    assert client._ack_queue.qsize() == 64
    retained = [client._ack_queue.get_nowait()["event_id"] for _ in range(64)]
    assert retained == list(range(192, 256))
    assert "acknowledgement queue full" in client.status().source_error


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
        deadline = time.monotonic() + 3.0
        while not client.status().connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.status().connected

        bridge.submit_frame(_frame(session="wire-session", squeeze=0.0))
        command = None
        while time.monotonic() < deadline:
            try:
                command = client.poll(_context()).command
            except TeleopClientError:
                time.sleep(0.01)
                continue
            break
        assert isinstance(command, CanonicalEefCommand)

        bridge.submit_frame(_frame(1, session="wire-session", squeeze=1.0))
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
