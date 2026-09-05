from __future__ import annotations

import threading

import numpy as np
import pytest

from core.app.handlers import recording, teleop
from core.app.state import RuntimeState, SessionMode, SessionState
from core.config import ConfigDict
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from teleop_client.base import (
    CanonicalEefCommand,
    QposCommand,
    TeleopInputToken,
    TeleopResult,
    TeleopStatus,
)

pytestmark = pytest.mark.unit


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
    robot = Robot(
        "test",
        (
            ActuatorGroup("left_arm", 2, ("l0", "lg"), gripper_index=1),
            ActuatorGroup("right_arm", 2, ("r0", "rg"), gripper_index=1),
        ),
        np.zeros(4, dtype=np.float32),
        ObservationSchema((CameraSpec("front", "cam"),), ("left_arm", "right_arm")),
    )
    runtime = RuntimeState(robot=robot, transport=_Transport())
    runtime.teleop_client = client
    runtime.teleop_execution = teleop.TeleopExecutionState(
        control_source="client", client_type="test", active=True, max_qpos_step=0.1
    )
    runtime.collection_teleop_armed = True
    runtime.collection_teleop_active = True
    return runtime


def test_handler_dispatches_qpos_and_limits_joint_step(monkeypatch) -> None:
    client = _Client([TeleopResult.from_command(QposCommand(np.asarray([0.3, 0.5, -0.2, -1.0])))])
    runtime = _runtime(client)
    monkeypatch.setattr(teleop, "forward_canonical_eef", lambda *_args: np.zeros(16))

    assert teleop.step_teleop(_config(), runtime, SessionState(mode=SessionMode.COLLECT), now=1.0)
    target, action = runtime.transport.published[0]
    assert target == "real"
    np.testing.assert_allclose(action, [0.1, 0.5, -0.1, 0.0])


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
            teleop.step_teleop(_config(), runtime, SessionState(mode=SessionMode.COLLECT), now=1.0)
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
        teleop.step_teleop(_config(), runtime, SessionState(mode=SessionMode.COLLECT), now=1.0)

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

    assert teleop.step_teleop(_config(), runtime, SessionState(mode=SessionMode.COLLECT), now=1.0)
    assert len(runtime.transport.published) == 1


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
    session = SessionState(mode=SessionMode.COLLECT)
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
