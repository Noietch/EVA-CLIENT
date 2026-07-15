from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from _harness import WebHarness, build_runtime, free_port

from core.app import run as app
from core.app.console.server import ConsoleContext, ConsoleRequestHandler, ThreadingHTTPServer
from core.app.handlers.recording import record_rollout_intervention_step
from core.app.state import SessionMode, SessionStatus
from core.config import ConfigDict, load_config
from examples.hardware.fake_common import FakeRobotNode, make_ui_handler


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _post_json(port: int, path: str, body: dict) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "POST",
        path,
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    response.read()
    connection.close()
    assert response.status == 200


@pytest.mark.parametrize(
    ("robot_name", "config_path"),
    [
        ("r1_lite", "configs/01_deploy/r1lite/openpi_qpos.py"),
        ("ur5e", "configs/01_deploy/ur5e/openpi_qpos.py"),
        ("arx_r5", "configs/01_deploy/arx_r5/openpi_qpos.py"),
        ("agilex_piper", "configs/01_deploy/dual_agilex_piper/openpi_qpos.py"),
    ],
)
def test_console_operator_hil_resume_and_abandon_for_every_fake_robot(
    monkeypatch: pytest.MonkeyPatch,
    robot_name: str,
    config_path: str,
):
    obs_endpoint = f"tcp://127.0.0.1:{_free_port()}"
    action_endpoint = f"tcp://127.0.0.1:{_free_port()}"
    fake = FakeRobotNode(robot_name, obs_endpoint, action_endpoint, 60.0, 8, 8)
    fake_thread = threading.Thread(target=fake.serve_forever, daemon=True)
    fake_thread.start()

    config = load_config(Path(config_path))
    config.transport.type = "zmq"
    config.policy.type = "mock"
    config.policy.backend_options = ConfigDict(chunk_size=8)
    config.inference_strategies = ConfigDict(
        {"sync": {"type": "BaseInferStrategy", "args": {"execute_horizon": 5}}}
    )
    config.transport.sub_endpoint = obs_endpoint
    config.transport.pub_endpoint = action_endpoint
    config.inference_cfg.publish_rate = 1000
    runtime, session = build_runtime(config)
    runtime.hil_control_mode = "relative"
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    session.is_setup_done = True
    session.selected_task = "fake HIL test"
    monkeypatch.setattr(app, "run_warmup_and_start", lambda *_args, **_kwargs: True)

    deadline = time.monotonic() + 2.0
    while runtime.transport.get_latest_qpos() is None and time.monotonic() < deadline:
        time.sleep(0.02)

    port = free_port()
    context = ConsoleContext(
        config=config,
        runtime=runtime,
        session=session,
        obs_reader=runtime.transport.create_observation_reader(),
        scene=None,
    )
    ConsoleRequestHandler.ctx = context
    server = ThreadingHTTPServer(("127.0.0.1", port), ConsoleRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    harness = WebHarness(config, runtime, session, port)
    fake_ui_port = _free_port()
    fake_ui = ThreadingHTTPServer(
        ("127.0.0.1", fake_ui_port),
        make_ui_handler(fake, f"http://127.0.0.1:{port}"),
    )
    fake_ui_thread = threading.Thread(target=fake_ui.serve_forever, daemon=True)
    fake_ui_thread.start()
    try:
        harness.do("/api/rollout_intervention_enabled", {"enabled": True})
        _post_json(fake_ui_port, "/api/operator", {"button": "x"})
        harness.pump()
        assert runtime.rollout_intervention_active is True
        assert harness.status()["rollout_intervention_active"] is True

        first_group = runtime.robot.actuator_groups[0]
        _post_json(
            fake_ui_port,
            "/api/joint_delta",
            {"group": first_group.name, "index": 0, "delta": 0.2},
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if record_rollout_intervention_step(runtime, session):
                break
            time.sleep(0.02)
        segment = runtime.rollout_intervention_active_segment
        assert segment is not None
        assert len(segment.frames) >= 1
        assert segment.frames[-1].action_qpos is not None

        _post_json(fake_ui_port, "/api/operator", {"button": "y"})
        harness.pump()
        assert runtime.rollout_intervention_active is False
        assert session.status is SessionStatus.RUNNING
        assert len(runtime.rollout_intervention_segments) == 1

        _post_json(fake_ui_port, "/api/operator", {"button": "x"})
        harness.pump()
        assert runtime.rollout_intervention_active is True
        pre_abandon = runtime.rollout_intervention_pre_qpos.copy()
        _post_json(
            fake_ui_port,
            "/api/joint_delta",
            {"group": first_group.name, "index": 0, "delta": -0.15},
        )
        time.sleep(0.05)
        _post_json(fake_ui_port, "/api/operator", {"button": "cancel"})
        harness.pump()
        assert runtime.rollout_intervention_active is False
        assert runtime.rollout_intervention_active_segment is None

        deadline = time.monotonic() + 2.0
        latest = runtime.transport.get_latest_qpos()
        while time.monotonic() < deadline and not np.allclose(latest, pre_abandon, atol=1e-3):
            time.sleep(0.02)
            latest = runtime.transport.get_latest_qpos()
        np.testing.assert_allclose(latest, pre_abandon, atol=1e-3)
    finally:
        fake_ui.shutdown()
        fake_ui.server_close()
        server.shutdown()
        server.server_close()
        runtime.transport.close()
        fake.close()
        fake_thread.join(timeout=2.0)
