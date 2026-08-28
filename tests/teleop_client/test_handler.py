from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.app import run as app_run
from core.app.handlers import control, recording, teleop
from core.app.handlers import io as handler_io
from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from teleop_client.base import (
    CanonicalEefCommand,
    QposCommand,
    TeleopClientError,
    TeleopInputToken,
    TeleopResult,
    TeleopStatus,
)


class _Transport:
    def __init__(self) -> None:
        self.qpos = np.zeros(4, dtype=np.float32)
        self.published: list[tuple[str, np.ndarray]] = []
        self.started = 0
        self.stopped = 0
        self.hil_relay_enabled: list[bool] = []
        self.fail_reset_hil = False
        self.fail_relay = False
        self.fail_start = False
        self.fail_stop = False

    def get_latest_qpos(self):
        return self.qpos.copy()

    def publish_action(self, action, target="real"):
        self.published.append((target, np.asarray(action).copy()))

    def supports_collection(self):
        return True

    def reset_hil_control(self):
        if self.fail_reset_hil:
            raise RuntimeError("reset HIL failed")

    def set_hil_relay_enabled(self, enabled):
        self.hil_relay_enabled.append(bool(enabled))
        if self.fail_relay:
            raise RuntimeError("relay off failed")

    def start_collection(self):
        self.started += 1
        if self.fail_start:
            raise RuntimeError("start collection failed")

    def stop_collection(self):
        self.stopped += 1
        if self.fail_stop:
            raise RuntimeError("stop collection failed")


class _Client:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.resets: list[bool] = []
        self.starts = 0
        self.closes = 0
        self.connected = True
        self.neutral = True
        self.source_error = ""
        self.fail_start = False
        self.fail_reset = False
        self.fail_close = False
        self.worker_running = False

    def start(self):
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("client start failed")
        self.worker_running = True

    def close(self):
        self.closes += 1
        self.worker_running = False
        if self.fail_close:
            raise RuntimeError("client close failed")

    def poll(self, _context):
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def reset(self, *, require_neutral=False):
        self.resets.append(require_neutral)
        if self.fail_reset:
            raise RuntimeError("client reset failed")

    def status(self):
        return TeleopStatus(
            source_type="test",
            connected=self.connected,
            neutral=self.neutral,
            source_error=self.source_error,
        )


class _GenerationClient(_Client):
    def __init__(self, result) -> None:
        super().__init__([result])
        self.result_is_current = True

    def validate_result(self, _result) -> bool:
        return self.result_is_current


def _robot() -> Robot:
    return Robot(
        "test",
        (
            ActuatorGroup("left_arm", 2, ("l0", "lg"), gripper_index=1),
            ActuatorGroup("right_arm", 2, ("r0", "rg"), gripper_index=1),
        ),
        np.zeros(4, dtype=np.float32),
        ObservationSchema((CameraSpec("front", "cam"),), ("left_arm", "right_arm")),
    )


def _config() -> ConfigDict:
    return ConfigDict(
        collection=ConfigDict(
            teleop=ConfigDict(
                control_source="client",
                client=ConfigDict(type="test", gripper=ConfigDict(open_value=1.0, close_value=0.0)),
                safety=ConfigDict(
                    max_qpos_step=0.1,
                    max_position_error_m=0.05,
                    max_orientation_error_rad=0.2,
                ),
            )
        )
    )


def _runtime(client) -> RuntimeState:
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    runtime.teleop_client = client
    runtime.teleop_execution = teleop.TeleopExecutionState(
        control_source="client", client_type="test", active=True, max_qpos_step=0.1
    )
    runtime.collection_teleop_armed = True
    runtime.collection_teleop_active = True
    return runtime


def _session() -> SessionState:
    return SessionState(mode=SessionMode.COLLECT)


def test_handler_dispatches_qpos_and_limits_joint_step(monkeypatch) -> None:
    client = _Client([TeleopResult.from_command(QposCommand(np.asarray([0.3, 0.5, -0.2, -1.0])))])
    runtime = _runtime(client)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    assert teleop.step_teleop(_config(), runtime, _session(), now=1.0)
    target, action = runtime.transport.published[0]
    assert target == "real"
    np.testing.assert_allclose(action, [0.1, 0.5, -0.1, 0.0])


def test_rollout_step_dispatches_client_qpos_without_collection_gate(monkeypatch) -> None:
    client = _Client([TeleopResult.from_command(QposCommand(np.asarray([0.2, 0.5, 0.2, 0.0])))])
    runtime = _runtime(client)
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    runtime.rollout_intervention_active = True
    session = SessionState(mode=SessionMode.REAL)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    assert teleop.step_rollout_teleop(_config(), runtime, session, now=1.0)
    assert runtime.transport.published[0][0] == "real"
    np.testing.assert_allclose(runtime.transport.published[0][1], [0.1, 0.5, 0.1, 0.0])


def test_activate_rollout_teleop_requires_connected_neutral_client() -> None:
    client = _Client([])
    runtime = _runtime(client)
    runtime.teleop_execution.active = False
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    session = SessionState(mode=SessionMode.REAL)

    assert teleop.activate_rollout_teleop(_config(), runtime, session) is True

    assert client.starts == 1
    assert client.resets == [True]
    assert runtime.teleop_execution.active is True


def test_activate_rollout_teleop_rejects_non_neutral_client() -> None:
    client = _Client([])
    client.neutral = False
    runtime = _runtime(client)
    runtime.teleop_execution.active = False
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    session = SessionState(mode=SessionMode.REAL)

    assert teleop.activate_rollout_teleop(_config(), runtime, session) is False

    assert client.starts == 1
    assert client.resets == []
    assert runtime.teleop_execution.active is False
    assert "neutral before activation" in session.last_error


def test_deactivate_rollout_teleop_leaves_state_active_when_neutral_reset_fails() -> None:
    client = _Client([])
    client.fail_reset = True
    runtime = _runtime(client)
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    runtime.rollout_intervention_active = True

    with pytest.raises(RuntimeError, match="client reset failed"):
        teleop.deactivate_rollout_teleop(runtime)

    assert runtime.teleop_execution.active is True
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.IDLE
    assert runtime.teleop_execution.last_fault == ""


def test_prewarm_teleop_ik_runs_once_without_publishing() -> None:
    class _Solver:
        def __init__(self) -> None:
            self.fk_calls: list[np.ndarray] = []
            self.solve_calls: list[tuple[np.ndarray, np.ndarray]] = []

        def fk_chunk(self, qpos_chunk):
            self.fk_calls.append(np.asarray(qpos_chunk).copy())
            return np.zeros((1, 16), dtype=np.float32)

        def solve_chunk(self, eef_chunk, *, seed_qpos):
            self.solve_calls.append((np.asarray(eef_chunk).copy(), np.asarray(seed_qpos).copy()))
            return np.asarray(seed_qpos, dtype=np.float32).reshape(1, -1)

    runtime = _runtime(_Client([]))
    solver = _Solver()
    runtime.ik_solver = solver
    runtime.transport.get_latest_qpos = lambda: (_ for _ in ()).throw(
        AssertionError("prewarm must not read live qpos")
    )

    assert teleop.prewarm_teleop_ik(_config(), runtime)
    assert teleop.prewarm_teleop_ik(_config(), runtime)

    assert len(solver.fk_calls) == 2
    assert len(solver.solve_calls) == 1
    np.testing.assert_allclose(solver.solve_calls[0][1], runtime.robot.initial_qpos)
    assert runtime.transport.published == []
    assert runtime.teleop_execution.ik_prewarmed is True


def test_ik_solver_uses_latest_transport_state_as_seed() -> None:
    runtime = _runtime(_Client([]))
    live_qpos = np.asarray([0.2, 0.8, -0.3, 0.4], dtype=np.float32)
    runtime.transport.qpos = live_qpos.copy()
    captured: dict[str, object] = {}
    runtime.robot.build_kinematics = lambda **kwargs: captured.update(kwargs) or object()
    config = SimpleNamespace(
        inference_cfg=SimpleNamespace(publish_rate=30),
        robot=SimpleNamespace(eef_reference_frame="world"),
    )

    handler_io.ensure_ik_solver(config, runtime)

    seeded_groups = captured["initial_qpos_groups"]
    assert isinstance(seeded_groups, list)
    np.testing.assert_array_equal(seeded_groups[0], live_qpos[:2])
    np.testing.assert_array_equal(seeded_groups[1], live_qpos[2:])


def test_tab_switch_to_collect_prewarms_teleop_ik(monkeypatch) -> None:
    runtime = _runtime(_Client([]))
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    runtime.teleop_execution.active = False
    calls: list[str] = []
    monkeypatch.setattr(
        app_run,
        "prewarm_teleop_ik",
        lambda *_args: calls.append("prewarm") or True,
    )

    app_run.handle_command("web:tab_switch:collect", _config(), runtime, _session())

    assert calls == ["prewarm"]
    assert runtime.collection_teleop_armed is False


def test_client_teleop_ignores_legacy_sim_output_target(monkeypatch) -> None:
    client = _Client([TeleopResult.from_command(QposCommand(np.zeros(4)))])
    runtime = _runtime(client)
    config = _config()
    config.collection.teleop.output_target = "sim"
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    assert teleop.step_teleop(config, runtime, _session(), now=1.0)
    assert runtime.transport.published[0][0] == "real"


def test_handler_rejects_result_when_worker_generation_changes_before_publish(monkeypatch) -> None:
    target = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 2,
        dtype=np.float32,
    )
    client = _GenerationClient(
        TeleopResult.from_command(
            CanonicalEefCommand(target, (True, True)),
            source_token=TeleopInputToken(1),
        )
    )
    runtime = _runtime(client)
    solving = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: target.copy())

    def solve(*_args, **_kwargs):
        solving.set()
        assert release.wait(timeout=1.0)
        return np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(teleop, "solve_canonical_eef", solve)
    errors: list[Exception] = []

    def run_step() -> None:
        try:
            teleop.step_teleop(_config(), runtime, _session(), now=1.0)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run_step)
    worker.start()
    assert solving.wait(timeout=1.0)
    client.result_is_current = False
    release.set()
    worker.join(timeout=1.0)

    assert errors == []
    assert "changed before publish" in runtime.teleop_execution.last_fault
    assert runtime.transport.published == []


def test_handler_skips_collection_publish_when_collection_deactivates_mid_tick(monkeypatch) -> None:
    target = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 2,
        dtype=np.float32,
    )
    client = _GenerationClient(
        TeleopResult.from_command(
            CanonicalEefCommand(target, (True, True)),
            source_token=TeleopInputToken(2),
        )
    )
    runtime = _runtime(client)
    solving = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: target.copy())

    def solve(*_args, **_kwargs):
        solving.set()
        assert release.wait(timeout=1.0)
        return np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(teleop, "solve_canonical_eef", solve)

    def run_step() -> None:
        teleop.step_teleop(_config(), runtime, _session(), now=1.0)

    worker = threading.Thread(target=run_step)
    worker.start()
    assert solving.wait(timeout=1.0)
    runtime.collection_teleop_active = False
    release.set()
    worker.join(timeout=1.0)

    assert runtime.transport.published == []


def test_handler_solves_canonical_eef_and_checks_residual(monkeypatch) -> None:
    target = np.asarray(
        [0.4, 0.1, 0.3, 1, 0, 0, 0, 1, 0.5, -0.1, 0.2, 1, 0, 0, 0, 0],
        dtype=np.float32,
    )
    client = _Client([TeleopResult.from_command(CanonicalEefCommand(target, (True, True)))])
    runtime = _runtime(client)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: target.copy())
    monkeypatch.setattr(teleop, "solve_canonical_eef", lambda *_args, **_kwargs: np.zeros(4))

    assert teleop.step_teleop(_config(), runtime, _session(), now=1.0)
    assert len(runtime.transport.published) == 1


@pytest.mark.parametrize(
    ("active_arms", "expected"),
    [
        ((True, False), [0.05, 0.8, -0.03, 0.2]),
        ((False, True), [0.02, 0.8, 0.05, 0.2]),
        ((True, True), [0.05, 0.8, 0.05, 0.2]),
        ((False, False), [0.02, 0.8, -0.03, 0.2]),
    ],
)
def test_handler_freezes_inactive_arm_joints_but_allows_gripper(
    monkeypatch,
    active_arms: tuple[bool, ...],
    expected: list[float],
) -> None:
    target = np.asarray(
        [0.4, 0.1, 0.3, 1, 0, 0, 0, 0.8, 0.5, -0.1, 0.2, 1, 0, 0, 0, 0.2],
        dtype=np.float32,
    )
    client = _Client([TeleopResult.from_command(CanonicalEefCommand(target, active_arms))])
    runtime = _runtime(client)
    runtime.transport.qpos = np.asarray([0.01, 0.3, -0.04, 0.7], dtype=np.float32)
    runtime.teleop_execution.last_safe_qpos = np.asarray([0.02, 0.4, -0.03, 0.6], dtype=np.float32)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: target.copy())
    monkeypatch.setattr(
        teleop,
        "solve_canonical_eef",
        lambda *_args, **_kwargs: np.asarray([0.05, 0.8, 0.05, 0.2], dtype=np.float32),
    )

    assert teleop.step_teleop(_config(), runtime, _session(), now=1.0)
    np.testing.assert_allclose(runtime.transport.published[0][1], expected)


def test_handler_drops_disconnected_and_stale_ticks_without_stopping(monkeypatch) -> None:
    client = _Client(
        [
            TeleopClientError("not connected"),
            TeleopResult.from_command(QposCommand(np.zeros(4))),
            TeleopClientError("stale"),
        ]
    )
    runtime = _runtime(client)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    assert not teleop.step_teleop(_config(), runtime, _session(), now=1.0)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.IDLE
    assert runtime.teleop_execution.last_fault == "not connected"
    assert teleop.step_teleop(_config(), runtime, _session(), now=1.1)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.COMMANDING
    assert not teleop.step_teleop(_config(), runtime, _session(), now=1.2)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.IDLE
    assert runtime.teleop_execution.last_fault == "stale"
    assert runtime.collection_teleop_active


def test_handler_retries_after_rejected_tick_without_neutral_gate(monkeypatch) -> None:
    client = _Client(
        [
            TeleopResult.rejected("target outside workspace"),
            TeleopResult.idle(),
            TeleopResult.idle(),
        ]
    )
    client.neutral = False
    runtime = _runtime(client)
    session = _session()
    session.status = teleop.SessionStatus.RUNNING
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    assert not teleop.step_teleop(_config(), runtime, session, now=1.0)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.REJECTED
    assert runtime.teleop_execution.last_fault == "target outside workspace"
    assert runtime.collection_teleop_active
    assert session.status is teleop.SessionStatus.RUNNING
    assert runtime.transport.published == []

    assert not teleop.step_teleop(_config(), runtime, session, now=1.1)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.IDLE
    assert runtime.teleop_execution.last_fault == ""


def test_publish_failure_drops_tick_and_keeps_teleop_active(monkeypatch) -> None:
    client = _Client([TeleopResult.from_command(QposCommand(np.zeros(4)))])
    runtime = _runtime(client)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    def fail_publish(*_args, **_kwargs):
        raise RuntimeError("transport unavailable")

    monkeypatch.setattr(runtime.transport, "publish_action", fail_publish)

    assert not teleop.step_teleop(_config(), runtime, _session(), now=1.0)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.REJECTED
    assert "transport unavailable" in runtime.teleop_execution.last_fault
    assert runtime.collection_teleop_active


def test_ik_rejection_and_later_stale_tick_do_not_reset_anchor_or_stop(monkeypatch) -> None:
    target = np.asarray(
        [0.4, 0.1, 0.3, 1, 0, 0, 0, 1, 0.5, -0.1, 0.2, 1, 0, 0, 0, 0],
        dtype=np.float32,
    )
    tracked = target.copy()
    tracked[0] += 0.1
    client = _Client(
        [
            TeleopResult.from_command(CanonicalEefCommand(target, (True, False))),
            TeleopClientError("stale after rejection"),
        ]
    )
    runtime = _runtime(client)
    runtime.teleop_execution.motion_started = True
    forward_values = iter((target, tracked, target))
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: next(forward_values))
    monkeypatch.setattr(teleop, "solve_canonical_eef", lambda *_args, **_kwargs: np.zeros(4))

    assert not teleop.step_teleop(_config(), runtime, _session(), now=1.0)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.REJECTED
    assert runtime.teleop_execution.motion_started
    assert "IK position error" in runtime.teleop_execution.last_fault
    assert client.resets == []

    assert not teleop.step_teleop(_config(), runtime, _session(), now=1.1)
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.IDLE
    assert runtime.collection_teleop_active


def test_transport_lifecycle_does_not_require_client() -> None:
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    runtime.collection_teleop_armed = True
    session = _session()
    config = ConfigDict(collection=ConfigDict(teleop=ConfigDict(control_source="transport")))

    assert teleop.activate_teleop(config, runtime, session)
    teleop.deactivate_teleop(config, runtime, session)
    teleop.deactivate_teleop(config, runtime, session)

    assert runtime.transport.started == 1
    assert runtime.transport.stopped == 1


def test_select_mode_stops_teleop_but_keeps_input_worker(monkeypatch) -> None:
    client = _Client([])
    client.worker_running = True
    runtime = _runtime(client)
    session = _session()
    runtime.teleop_execution.condition = teleop.TeleopExecutionCondition.REJECTED
    runtime.teleop_execution.last_fault = "stale input"
    runner = object()
    runtime.collection_capture_runner = runner
    stopped_capture = []

    def stop_capture(current_runtime) -> None:
        stopped_capture.append(current_runtime.collection_capture_runner)
        current_runtime.collection_capture_runner = None

    monkeypatch.setattr(teleop, "stop_collection_capture", stop_capture)

    control.select_mode(SessionMode.SIM, ConfigDict(), session, runtime)

    assert stopped_capture == [runner]
    assert client.resets == [False]
    assert client.closes == 0
    assert client.worker_running is True
    assert runtime.teleop_execution.active is False
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.IDLE
    assert runtime.teleop_execution.last_fault == ""
    assert runtime.collection_teleop_active is False
    assert runtime.transport.stopped == 1


@pytest.mark.parametrize(
    ("entrypoint", "stop_module"),
    [
        ("select_mode", control),
        ("tab_switch", app_run),
    ],
)
@pytest.mark.parametrize("message", ["capture stop failed", "episode finalize failed"])
def test_collection_exit_deactivates_when_collection_stop_raises(
    monkeypatch,
    entrypoint: str,
    stop_module,
    message: str,
) -> None:
    client = _Client([])
    runtime = _runtime(client)
    runtime.teleop_execution.active = True
    runtime.collection_teleop_active = True
    session = _session()
    session.status = SessionStatus.RUNNING

    def fail_stop(*_args) -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(stop_module, "collect_stop", fail_stop)
    monkeypatch.setattr(teleop, "stop_collection_capture", lambda _runtime: None)

    with pytest.raises(RuntimeError, match=message):
        if entrypoint == "select_mode":
            control.select_mode(SessionMode.SIM, ConfigDict(), session, runtime)
        else:
            app_run.handle_command("web:tab_switch:manual", _config(), runtime, session)

    assert runtime.collection_teleop_active is False
    assert runtime.teleop_execution.active is False
    assert client.closes == 0
    assert runtime.transport.stopped == 1
    assert runtime.transport.published == []
    assert session.last_error == ""


@pytest.mark.parametrize(
    ("command", "reset_message"),
    [
        ("web:collect_arm:off", "ARM OFF must not reset robot"),
        ("web:tab_switch:manual", "tab switch must not reset robot"),
    ],
)
def test_collection_exit_saves_and_stops_teleop_without_robot_reset(
    monkeypatch,
    command: str,
    reset_message: str,
) -> None:
    runtime = _runtime(_Client([]))
    session = _session()
    session.status = SessionStatus.RUNNING
    calls: list[str] = []

    def save(*_args) -> None:
        calls.append("save")
        session.status = SessionStatus.READY

    def stop(*_args) -> None:
        calls.append("stop_teleop")
        runtime.collection_teleop_active = False
        runtime.teleop_execution.active = False

    monkeypatch.setattr(app_run, "collect_stop", save)
    monkeypatch.setattr(app_run, "collect_stop_teleop", stop)
    monkeypatch.setattr(
        app_run,
        "run_reset",
        lambda *_args: (_ for _ in ()).throw(AssertionError(reset_message)),
    )

    app_run.handle_command(command, _config(), runtime, session)

    assert calls == ["save", "stop_teleop"]
    assert runtime.collection_teleop_armed is False
    assert session.mode is SessionMode.SELECT
    assert session.status is SessionStatus.UNSET


def test_collect_home_runs_reset_only_when_disarmed(monkeypatch) -> None:
    runtime = _runtime(_Client([]))
    runtime.collection_teleop_armed = False
    runtime.collection_teleop_active = False
    runtime.teleop_execution.active = False
    session = _session()
    calls: list[str] = []

    monkeypatch.setattr(app_run, "run_reset", lambda *_args: calls.append("run_reset"))

    app_run.handle_command("web:collect_home", _config(), runtime, session)

    assert calls == ["run_reset"]

    runtime.collection_teleop_armed = True
    session.last_error = ""
    app_run.handle_command("web:collect_home", _config(), runtime, session)
    assert calls == ["run_reset"]
    assert session.last_error == "Collection HOME requires ARM OFF"


def test_deactivate_then_reactivate_reuses_input_client(monkeypatch) -> None:
    client = _Client([])
    runtime = _runtime(client)
    session = _session()
    config = _config()
    runtime.teleop_execution.active = True
    runtime.collection_teleop_active = True
    teleop.deactivate_teleop(config, runtime, session)

    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))
    monkeypatch.setattr(
        teleop,
        "solve_canonical_eef",
        lambda *_args, **_kwargs: runtime.transport.qpos.copy(),
    )

    assert runtime.teleop_client is client
    assert client.closes == 0
    assert teleop.activate_teleop(config, runtime, session)
    assert client.starts == 1
    assert runtime.collection_teleop_active is True
    assert runtime.teleop_execution.active is True


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("reset", "client reset failed"),
        ("relay", "relay off failed"),
        ("stop", "stop collection failed"),
    ],
)
def test_deactivate_attempts_all_cleanup_steps(failure, message) -> None:
    client = _Client([])
    runtime = _runtime(client)
    session = _session()
    runtime.last_collection_timestamp = 12.0
    if failure == "reset":
        client.fail_reset = True
    elif failure == "relay":
        runtime.transport.fail_relay = True
    elif failure == "stop":
        runtime.transport.fail_stop = True

    with pytest.raises(RuntimeError, match=message):
        teleop.deactivate_teleop(_config(), runtime, session)

    assert runtime.teleop_execution.active is False
    assert runtime.collection_teleop_active is False
    assert runtime.last_collection_timestamp is None
    assert runtime.teleop_execution.condition is teleop.TeleopExecutionCondition.REJECTED
    assert message in runtime.teleop_execution.last_fault
    assert client.closes == 0
    assert runtime.transport.hil_relay_enabled == [False]
    assert runtime.transport.stopped == 1


@pytest.mark.parametrize("failure", ["start", "reset", "collection"])
def test_activate_failure_keeps_input_client_and_leaves_collection_inactive(
    monkeypatch, failure
) -> None:
    client = _Client([])
    runtime = _runtime(client)
    runtime.collection_teleop_active = False
    runtime.teleop_execution.active = False
    session = _session()
    config = _config()
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))
    monkeypatch.setattr(
        teleop,
        "solve_canonical_eef",
        lambda *_args, **_kwargs: runtime.transport.qpos.copy(),
    )
    if failure == "start":
        client.fail_start = True
        expected = "client start failed"
    elif failure == "reset":
        client.fail_reset = True
        expected = "client reset failed"
    else:
        runtime.transport.fail_start = True
        expected = "start collection failed"

    assert teleop.activate_teleop(config, runtime, session) is False
    assert expected in session.last_error
    assert runtime.collection_teleop_active is False
    assert runtime.teleop_execution.active is False
    assert client.closes == 0
    if failure == "collection":
        assert runtime.transport.stopped == 1
    else:
        assert runtime.transport.stopped == 0


def test_setup_teleop_uses_generic_builder_and_robot_gripper_limits(monkeypatch) -> None:
    client = _Client([])
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    config = _config()
    config.robot = ConfigDict(gripper_open=0.8, gripper_close=0.2)
    captured = {}

    def fake_build_client(client_config, *, arm_group_names):
        captured["config"] = client_config
        captured["arms"] = arm_group_names
        return client

    monkeypatch.setattr(teleop, "build_client", fake_build_client)
    teleop.setup_teleop(config, runtime)

    assert runtime.teleop_client is client
    assert client.starts == 0
    assert captured == {
        "config": config.collection.teleop.client,
        "arms": ("left_arm", "right_arm"),
    }
    assert runtime.teleop_execution.gripper_min == 0.2
    assert runtime.teleop_execution.gripper_max == 0.8


def test_activate_teleop_lazily_builds_and_starts_client(monkeypatch) -> None:
    client = _Client([])
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    config = _config()
    config.robot = ConfigDict(gripper_open=1.0, gripper_close=0.0)
    runtime.collection_teleop_armed = True
    session = _session()
    monkeypatch.setattr(teleop, "build_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))
    monkeypatch.setattr(
        teleop,
        "solve_canonical_eef",
        lambda *_args, **_kwargs: runtime.transport.qpos.copy(),
    )

    assert teleop.activate_teleop(config, runtime, session)

    assert runtime.teleop_client is client
    assert client.starts == 1
    assert client.resets == [True]
    assert runtime.collection_teleop_active is True
    assert session.status is SessionStatus.READY


def test_activate_teleop_rejects_non_neutral_client(monkeypatch) -> None:
    client = _Client([])
    client.neutral = False
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    config = _config()
    config.robot = ConfigDict(gripper_open=1.0, gripper_close=0.0)
    runtime.collection_teleop_armed = True
    session = _session()
    monkeypatch.setattr(teleop, "build_client", lambda *_args, **_kwargs: client)

    assert teleop.activate_teleop(config, runtime, session) is False

    assert client.starts == 1
    assert client.resets == [False]
    assert runtime.collection_teleop_active is False
    assert "neutral before activation" in session.last_error


def test_lazy_client_start_failure_keeps_client_for_retry(monkeypatch) -> None:
    class StartFailClient(_Client):
        def start(self):
            super().start()
            raise RuntimeError("start failed")

    client = StartFailClient([])
    runtime = RuntimeState(robot=_robot(), transport=_Transport())
    config = _config()
    config.robot = ConfigDict(gripper_open=1.0, gripper_close=0.0)
    runtime.collection_teleop_armed = True
    session = _session()
    monkeypatch.setattr(teleop, "build_client", lambda *_args, **_kwargs: client)

    assert teleop.activate_teleop(config, runtime, session) is False

    assert client.starts == 1
    assert client.closes == 0
    assert runtime.teleop_client is client
    assert "start failed" in session.last_error


def test_close_teleop_retains_client_when_shutdown_fails() -> None:
    class CloseFailClient(_Client):
        def close(self):
            super().close()
            raise RuntimeError("worker still running")

    client = CloseFailClient([])
    runtime = _runtime(client)

    teleop.close_teleop(runtime)

    assert client.closes == 1
    assert runtime.teleop_client is client
    assert runtime.teleop_execution.active is False
    assert "worker still running" in runtime.teleop_execution.last_fault


def test_close_teleop_active_collection_stops_transport_once() -> None:
    client = _Client([])
    runtime = _runtime(client)

    teleop.close_teleop(runtime)
    teleop.close_teleop(runtime)

    assert runtime.collection_teleop_active is False
    assert runtime.teleop_execution.active is False
    assert runtime.transport.stopped == 1
    assert runtime.teleop_client is None


class _CollectionLogger:
    is_collection_enabled = True

    def __init__(self, *, fail_stage: str = "") -> None:
        self.fail_stage = fail_stage
        self.has_active_episode = False
        self.cancelled = 0

    def is_queue_full(self) -> bool:
        return False

    def start_episode(
        self,
        *,
        task,
        collection_min_capture_time=None,
        collection_dataset=None,
    ) -> None:
        del task, collection_min_capture_time, collection_dataset
        if self.fail_stage == "start_episode":
            self.has_active_episode = True
            raise RuntimeError("episode open failed")
        self.has_active_episode = True

    def cancel_episode(self, _reason: str) -> None:
        self.cancelled += 1
        self.has_active_episode = False


@pytest.mark.parametrize(
    ("control_source", "fail_stage"),
    [
        ("client", "clear_backlog"),
        ("transport", "start_episode"),
        ("transport", "capture"),
    ],
)
def test_collect_start_failure_cancels_episode_and_deactivates_teleop(
    monkeypatch,
    control_source: str,
    fail_stage: str,
) -> None:
    client = _Client([])
    runtime = _runtime(client)
    logger_obj = _CollectionLogger(fail_stage=fail_stage)
    runtime.episode_logger = logger_obj
    runtime.transport.clear_collection_backlog = lambda: (
        (_ for _ in ()).throw(RuntimeError("backlog failed"))
        if fail_stage == "clear_backlog"
        else 1.0
    )
    runtime.teleop_execution.control_source = control_source
    config = _config()
    config.inference_cfg = ConfigDict(publish_rate=30.0)
    config.collection.teleop.control_source = control_source
    session = _session()
    session.selected_collect_task = "pick"

    monkeypatch.setattr(recording, "activate_teleop", lambda *_args: True)
    if fail_stage == "capture":
        monkeypatch.setattr(
            recording,
            "start_collection_capture",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("capture start failed")),
        )

    assert not recording.collect_start(config, runtime, session)
    assert runtime.collection_teleop_active is False
    assert runtime.teleop_execution.active is False
    assert client.closes == 0
    assert runtime.transport.stopped == 1
    assert logger_obj.has_active_episode is False
    if fail_stage != "clear_backlog":
        assert logger_obj.cancelled == 1
    assert "Collection start failed" in session.last_error
