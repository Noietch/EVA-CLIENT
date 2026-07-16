"""Tests for CLI state helpers and rollout intervention state."""

from __future__ import annotations

import logging
import queue

import numpy as np

from core.app import run as app
from core.app.handlers import control, recording
from core.app.state import (
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
    default_inference_strategy_key,
    format_task_label,
    resolve_inference_strategy_label,
)
from core.config import ConfigDict
from core.types import (
    CollectionRawBatch,
    Observation,
    RawCollectionSnapshot,
    RolloutInterventionSegment,
)
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from transport.base import HilStatus


def test_format_task_label_none():
    assert format_task_label(None) == "unset"


def test_format_task_label_empty():
    assert format_task_label("") == "<empty>"


def test_format_task_label_normal():
    assert format_task_label("pick up cup") == "pick up cup"


def test_gripper_command_records_collection_action_target():
    transport = _CollectionTransport()
    transport.latest_qpos = np.array([1.0, 100.0, 2.0, 0.0], dtype=np.float32)
    transport.recorded_gripper_actions = []

    def record_collection_gripper_action(group_name, value):
        transport.recorded_gripper_actions.append((group_name, value))

    transport.record_collection_gripper_action = record_collection_gripper_action
    runtime = RuntimeState(robot=_gripper_robot(), transport=transport)
    session = SessionState()

    ok = control.set_gripper_immediate(
        _gripper_config(),
        runtime,
        session,
        open_gripper=False,
        side="l",
        lock=False,
    )

    assert ok is True
    assert transport.recorded_gripper_actions == [("left_arm", 0.0)]


def _config_with_strategies(strategies):
    return ConfigDict(inference_strategies=strategies)


def test_default_inference_strategy_key_first_entry():
    cfg = _config_with_strategies(
        {"sync": {"type": "sync", "args": {}}, "async": {"type": "async_prefetch", "args": {}}}
    )
    assert default_inference_strategy_key(cfg) == "sync"


def test_resolve_label_none_falls_back_to_default():
    cfg = _config_with_strategies({"async": {"type": "async_prefetch", "args": {}}})
    assert resolve_inference_strategy_label(cfg, None) == "async"


def test_resolve_label_explicit_key_passthrough():
    cfg = _config_with_strategies(
        {"sync": {"type": "sync", "args": {}}, "rtc": {"type": "rtc", "args": {}}}
    )
    assert resolve_inference_strategy_label(cfg, "rtc") == "rtc"


class _CollectionTransport:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.cleared = 0
        self.latest_qpos = np.array([0.1, 0.2], dtype=np.float32)
        self.collection_frames: list[Observation] = []
        self.diagnostics = ""
        self.hil_resets = 0
        self.hil_modes: list[str] = []
        self.hil_relay_enabled: list[bool] = []
        self.hil_supported = True
        self.hil_start_error = ""
        self.published: list[tuple[str, np.ndarray]] = []
        self.sleep_count = 0

    def supports_collection(self) -> bool:
        return True

    def start_collection(self) -> None:
        self.started += 1

    def stop_collection(self) -> None:
        self.stopped += 1

    def clear_collection_backlog(self) -> None:
        self.cleared += 1

    def get_latest_qpos(self) -> np.ndarray | None:
        return self.latest_qpos.copy()

    def get_collection_frame(self) -> Observation | None:
        if not self.collection_frames:
            return None
        return self.collection_frames.pop(0)

    def acquire_collection_raw(self):
        return None

    def collection_diagnostics(self) -> str:
        return self.diagnostics

    def reset_hil_control(self) -> None:
        self.hil_resets += 1

    def set_hil_control_mode(self, mode: str) -> None:
        self.hil_modes.append(mode)

    def set_hil_relay_enabled(self, enabled: bool) -> None:
        self.hil_relay_enabled.append(enabled)

    def hil_status(self) -> HilStatus:
        active = bool(self.hil_relay_enabled and self.hil_relay_enabled[-1])
        return HilStatus(supported=self.hil_supported, active=active)

    def start_hil_control(self, mode: str) -> HilStatus:
        if self.hil_start_error:
            return HilStatus(supported=True, error=self.hil_start_error)
        self.reset_hil_control()
        self.set_hil_control_mode(mode)
        self.set_hil_relay_enabled(True)
        return HilStatus(supported=True, active=True)

    def stop_hil_control(self) -> HilStatus:
        self.set_hil_relay_enabled(False)
        return HilStatus(supported=True, active=False)

    def get_hil_frame(self) -> Observation | None:
        return self.get_collection_frame()

    def has_sim_publishers(self) -> bool:
        return False

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        self.published.append((target, np.asarray(action, dtype=np.float32).copy()))

    def create_rate(self, _hz: float):
        transport = self

        class _Rate:
            def sleep(self) -> None:
                transport.sleep_count += 1

        return _Rate()


class _EmptyCollectionLogger:
    is_collection_enabled = True

    def end_episode(self) -> bool:
        return False


class _ActiveRolloutLogger:
    has_active_episode = True
    active_frame_count = 1

    def cancel_episode(self, _reason: str) -> None:
        raise AssertionError("active rollout should not be cancelled")


class _RolloutLifecycleLogger:
    has_active_episode = False
    active_frame_count = 3

    def __init__(self) -> None:
        self.started_tasks = []
        self.cancelled = []
        self.segments = []
        self.ended = 0

    def start_episode(self, task: str) -> None:
        self.has_active_episode = True
        self.started_tasks.append(task)

    def cancel_episode(self, reason: str) -> None:
        self.has_active_episode = False
        self.cancelled.append(reason)

    def set_rollout_intervention_segments(self, segments) -> None:
        self.segments = list(segments)

    def end_episode(self) -> bool:
        self.has_active_episode = False
        self.ended += 1
        return True


class _RawEpisodeLogger:
    has_active_episode = True
    active_frame_count = 1

    def __init__(self) -> None:
        self.raw: list[tuple[RawCollectionSnapshot, np.ndarray, np.ndarray]] = []

    def ingest_raw_episode_snapshot(
        self,
        snapshot: RawCollectionSnapshot,
        action: np.ndarray,
        state_qpos: np.ndarray,
    ) -> None:
        self.raw.append(
            (
                snapshot,
                np.asarray(action, dtype=np.float32).copy(),
                np.asarray(state_qpos, dtype=np.float32).copy(),
            )
        )

    def record_step(self, _frame: Observation, _action: np.ndarray) -> None:
        raise AssertionError("raw-capable recording must not call record_step")


class _RawTransport:
    def __init__(self) -> None:
        self.snapshot = RawCollectionSnapshot(
            timestamp=1.0,
            decode_raw=lambda: CollectionRawBatch(),
        )
        self.get_frame_calls = 0

    def supports_collection(self) -> bool:
        return True

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        snapshot = self.snapshot
        self.snapshot = None
        return snapshot

    def get_frame(self) -> Observation | None:
        self.get_frame_calls += 1
        raise AssertionError("raw-capable recording must not call get_frame")

    def get_latest_qpos(self) -> np.ndarray:
        return np.array([3.0, 4.0], dtype=np.float32)


class _ResettableStrategy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _robot() -> Robot:
    return Robot(
        name="fake_arm",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam"),),
            state_composition=("arm",),
        ),
    )


def _runtime_and_session() -> tuple[RuntimeState, SessionState]:
    return RuntimeState(robot=_robot(), transport=_CollectionTransport()), SessionState()


def _gripper_robot() -> Robot:
    return Robot(
        name="fake_gripper_robot",
        actuator_groups=(
            ActuatorGroup("left_arm", 2, ("left_j0", "left_gripper"), gripper_index=1),
            ActuatorGroup("right_arm", 2, ("right_j0", "right_gripper"), gripper_index=1),
        ),
        initial_qpos=np.zeros(4, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam"),),
            state_composition=("left_arm", "right_arm"),
        ),
    )


def _gripper_config() -> ConfigDict:
    return ConfigDict(
        robot=ConfigDict(gripper_open=100.0, gripper_close=0.0, gripper_threshold=None),
        inference_cfg=ConfigDict(publish_rate=15),
    )


def _config() -> ConfigDict:
    return ConfigDict(
        collection=ConfigDict(
            schema=ConfigDict(columns={"qpos": "state", "action_qpos": "action"}),
        ),
        rollout=ConfigDict(intervention=ConfigDict(control_mode="relative")),
        inference_cfg=ConfigDict(publish_rate=1000),
    )


def _observation(timestamp: float = 1.0) -> Observation:
    return Observation(
        timestamp=timestamp,
        images={"cam": np.zeros((2, 2, 3), dtype=np.uint8)},
        state_qpos=np.array([0.3, 0.4], dtype=np.float32),
        action_qpos=np.array([0.5, 0.6], dtype=np.float32),
    )


def test_rollout_intervention_segment_tracks_observations():
    segment = RolloutInterventionSegment(
        segment_index=2,
        start_policy_frame_index=5,
        pre_intervention_qpos=np.array([0.1, 0.2], dtype=np.float32),
    )

    segment.frames.append(_observation(1.5))
    segment.resume_policy_frame_index = 8

    assert segment.segment_index == 2
    assert segment.start_policy_frame_index == 5
    assert segment.resume_policy_frame_index == 8
    assert len(segment.frames) == 1


def test_runtime_intervention_buffers_start_empty():
    runtime, _session = _runtime_and_session()

    assert runtime.rollout_intervention_pre_qpos is None
    assert runtime.rollout_intervention_active_segment is None
    assert runtime.rollout_intervention_segments == []
    assert runtime.rollout_intervention_next_segment_index == 0


def test_runtime_defaults_rollout_hil_off():
    runtime, _session = _runtime_and_session()

    assert runtime.rollout_intervention_enabled is False


def test_collect_start_teleop_starts_transport_collection():
    runtime, session = _runtime_and_session()
    runtime.collection_teleop_armed = True

    ok = recording.collect_start_teleop(_config(), runtime, session)

    assert ok is True
    assert runtime.collection_teleop_active is True
    assert runtime.transport.started == 1
    assert session.mode is SessionMode.COLLECT
    assert runtime.transport.hil_resets == 1
    assert runtime.transport.hil_relay_enabled == [True]


def test_collect_start_teleop_refuses_when_not_armed():
    runtime, session = _runtime_and_session()

    ok = recording.collect_start_teleop(_config(), runtime, session)

    assert ok is False
    assert runtime.collection_teleop_active is False
    assert runtime.transport.started == 0
    assert runtime.transport.hil_relay_enabled == []
    assert session.last_error == "Collection teleop requires COLLECT activation"


def test_collect_stop_teleop_stops_transport_collection():
    runtime, session = _runtime_and_session()
    runtime.collection_teleop_active = True
    session.mode = SessionMode.COLLECT

    recording.collect_stop_teleop(_config(), runtime, session)

    assert runtime.collection_teleop_active is False
    assert runtime.transport.stopped == 1
    assert runtime.transport.hil_relay_enabled == [False]


def test_collect_stop_reports_collection_diagnostics(caplog):
    runtime, session = _runtime_and_session()
    runtime.episode_logger = _EmptyCollectionLogger()
    runtime.transport.diagnostics = (
        "missing collection streams: action_qpos_gripper:left_arm "
        "topic=/motion_target/target_position_gripper_left no messages"
    )

    with caplog.at_level(logging.WARNING):
        recording.collect_stop(_config(), runtime, session)

    assert session.status is SessionStatus.READY
    assert session.last_error == (
        "Episode saved 0 frames; missing collection streams: "
        "action_qpos_gripper:left_arm "
        "topic=/motion_target/target_position_gripper_left no messages"
    )
    assert session.last_error in caplog.text


def test_start_rollout_intervention_does_not_restart_transport_collection():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_enabled = True

    assert recording.start_rollout_intervention(_config(), runtime, session) is True
    assert runtime.rollout_intervention_active is True
    assert runtime.transport.started == 0
    assert runtime.transport.cleared == 0
    assert runtime.transport.hil_resets == 1
    assert runtime.transport.hil_relay_enabled == [True]
    np.testing.assert_allclose(runtime.rollout_intervention_pre_qpos, [0.1, 0.2])
    assert runtime.rollout_intervention_active_segment is not None


def test_rollout_episode_starts_collection_before_policy_capture(monkeypatch):
    runtime, session = _runtime_and_session()
    episode_logger = _RolloutLifecycleLogger()
    runtime.rollout_episode_logger = episode_logger
    session.selected_task = "pick"
    monkeypatch.setattr(recording, "maybe_build_rollout_episode_logger", lambda *args: None)

    recording.begin_rollout_save_episode(_config(), runtime, session)

    assert runtime.transport.started == 1
    assert runtime.transport.cleared == 1
    assert episode_logger.started_tasks == ["pick"]
    recording.stop_collection_capture(runtime)


def test_rollout_ui_save_stops_collection_and_persists_whole_episode():
    runtime, session = _runtime_and_session()
    episode_logger = _RolloutLifecycleLogger()
    episode_logger.has_active_episode = True
    runtime.rollout_episode_logger = episode_logger
    runtime.rollout_save_ready = True

    saved = recording.save_rollout_episode(runtime, session)

    assert saved is True
    assert runtime.transport.stopped == 1
    assert episode_logger.ended == 1
    assert runtime.rollout_save_ready is False


def test_rollout_reset_discards_episode_and_stops_collection():
    runtime, _session = _runtime_and_session()
    episode_logger = _RolloutLifecycleLogger()
    episode_logger.has_active_episode = True
    runtime.rollout_episode_logger = episode_logger
    runtime.rollout_save_ready = True

    recording.discard_rollout_episode(runtime)

    assert runtime.transport.stopped == 1
    assert episode_logger.cancelled == ["user reset"]
    assert runtime.rollout_save_ready is False


def test_start_rollout_intervention_refuses_when_hil_off():
    runtime, session = _runtime_and_session()

    assert recording.start_rollout_intervention(_config(), runtime, session) is False

    assert runtime.rollout_intervention_active is False
    assert runtime.transport.started == 0
    assert runtime.transport.hil_relay_enabled == []
    assert session.last_error == "Rollout HIL is off"


def test_start_rollout_intervention_refuses_unsupported_transport():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_enabled = True
    runtime.transport.hil_supported = False

    assert recording.start_rollout_intervention(_config(), runtime, session) is False
    assert runtime.rollout_intervention_active is False
    assert session.last_error == "Transport does not support HIL"


def test_start_rollout_intervention_does_not_allocate_segment_before_ack():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_enabled = True
    runtime.transport.hil_start_error = "leader unavailable"

    assert recording.start_rollout_intervention(_config(), runtime, session) is False
    assert runtime.rollout_intervention_active_segment is None
    assert runtime.rollout_intervention_next_segment_index == 0
    assert session.last_error == "leader unavailable"


def test_start_rollout_intervention_resets_policy_strategy():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_enabled = True
    strategy = _ResettableStrategy()
    runtime.infer_strategy = strategy

    assert recording.start_rollout_intervention(_config(), runtime, session) is True
    assert strategy.resets == 1


def test_start_rollout_intervention_preserves_collection_backlog():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_enabled = True

    assert recording.start_rollout_intervention(_config(), runtime, session) is True
    assert runtime.rollout_intervention_active is True
    assert runtime.transport.started == 0
    assert runtime.transport.cleared == 0
    assert runtime.transport.hil_resets == 1
    assert runtime.rollout_intervention_active_segment is not None


def test_stop_rollout_intervention_disables_only_hil_relay():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_active = True

    assert recording.stop_rollout_intervention(_config(), runtime, session, required=True) is True

    assert runtime.transport.stopped == 0
    assert runtime.rollout_intervention_active is False
    assert runtime.transport.hil_relay_enabled == [False]


def test_record_rollout_intervention_step_appends_observation():
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_active = True
    runtime.rollout_intervention_active_segment = RolloutInterventionSegment(
        segment_index=0,
        start_policy_frame_index=3,
        pre_intervention_qpos=np.array([0.1, 0.2], dtype=np.float32),
    )
    runtime.transport.collection_frames.append(_observation(2.0))

    assert recording.record_rollout_intervention_step(runtime, session) is True

    assert len(runtime.rollout_intervention_active_segment.frames) == 1
    assert runtime.rollout_intervention_active_segment.frames[0].timestamp == 2.0


def test_accept_rollout_intervention_segment_moves_active_to_accepted():
    runtime, session = _runtime_and_session()
    session.step_index = 9
    runtime.rollout_intervention_active_segment = RolloutInterventionSegment(
        segment_index=0,
        start_policy_frame_index=3,
        pre_intervention_qpos=np.array([0.1, 0.2], dtype=np.float32),
        frames=[_observation(2.0)],
    )

    assert recording.accept_rollout_intervention_segment(runtime, session) is True

    assert runtime.rollout_intervention_active_segment is None
    assert len(runtime.rollout_intervention_segments) == 1
    assert runtime.rollout_intervention_segments[0].resume_policy_frame_index == 9


def test_rollback_rollout_intervention_scales_steps_with_joint_distance(monkeypatch):
    from core.app.handlers import imaging

    runtime, session = _runtime_and_session()
    runtime.transport.latest_qpos = np.array([0.1, 0.2], dtype=np.float32)
    runtime.rollout_intervention_pre_qpos = np.array([0.11, 0.2], dtype=np.float32)
    observed_steps = []

    def fake_build_linear_trajectory(current, target, steps):
        observed_steps.append(steps)
        return np.asarray([current, target], dtype=np.float32)

    monkeypatch.setattr(imaging, "build_linear_trajectory", fake_build_linear_trajectory)

    assert recording.rollback_rollout_intervention(_config(), runtime, session) is True

    assert observed_steps == [2]
    assert len(runtime.transport.published) == 1
    np.testing.assert_allclose(runtime.transport.published[0][1], [0.11, 0.2])


def test_console_halt_starts_intervention_without_ending_rollout(monkeypatch):
    calls = []
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.rollout_intervention_enabled = True
    runtime.rollout_save_ready = False

    monkeypatch.setattr(app, "is_replay", lambda runtime: False)
    monkeypatch.setattr(
        app,
        "mark_rollout_save_ready",
        lambda runtime, reason: calls.append(("save_ready", reason)),
    )
    monkeypatch.setattr(
        app,
        "start_rollout_intervention",
        lambda config, runtime, session: calls.append(("intervention", True)) or True,
    )

    app.handle_command("web:halt", _config(), runtime, session)

    assert session.status is SessionStatus.READY
    assert calls == [("save_ready", "stop"), ("intervention", True)]


def test_console_halt_respects_runtime_hil_off(monkeypatch):
    calls = []
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.rollout_intervention_enabled = False

    monkeypatch.setattr(app, "is_replay", lambda runtime: False)
    monkeypatch.setattr(
        app,
        "mark_rollout_save_ready",
        lambda runtime, reason: calls.append(("save_ready", reason)),
    )
    monkeypatch.setattr(
        app,
        "start_rollout_intervention",
        lambda config, runtime, session: calls.append(("intervention", True)) or True,
    )

    app.handle_command("web:halt", _config(), runtime, session)

    assert session.status is SessionStatus.READY
    assert calls == [("save_ready", "stop")]
    assert runtime.rollout_intervention_active is False


def test_rollout_intervention_enabled_command_updates_runtime_flag():
    runtime, session = _runtime_and_session()

    app.handle_command(
        "web:rollout_intervention_enabled:0",
        _config(),
        runtime,
        session,
    )

    assert runtime.rollout_intervention_enabled is False
    assert session.last_error == ""

    app.handle_command(
        "web:rollout_intervention_enabled:1",
        _config(),
        runtime,
        session,
    )

    assert runtime.rollout_intervention_enabled is True
    assert session.last_error == ""


def test_intervention_action_interrupt_respects_runtime_hil_off():
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.rollout_episode_logger = _ActiveRolloutLogger()
    runtime.rollout_intervention_enabled = False
    runtime.command_queue = queue.Queue()
    runtime.command_queue.put("web:operator_action:intervention_toggle:external")

    interrupted = control.poll_motion_commands(
        _config(),
        runtime,
        session,
    )

    assert interrupted is True
    assert session.interrupt_requested is True
    assert runtime.rollout_intervention_active is False
    assert runtime.rollout_save_ready is True
    assert runtime.transport.started == 0


def test_intervention_action_interrupts_sync_motion_and_starts_intervention():
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.rollout_episode_logger = _ActiveRolloutLogger()
    runtime.rollout_intervention_enabled = True
    runtime.command_queue = queue.Queue()
    runtime.command_queue.put("web:operator_action:intervention_toggle:external")

    interrupted = control.poll_motion_commands(
        _config(),
        runtime,
        session,
    )

    assert interrupted is True
    assert session.interrupt_requested is True
    assert runtime.rollout_intervention_active is True
    assert runtime.rollout_save_ready is True
    assert runtime.transport.started == 0
    assert runtime.transport.cleared == 0


def test_rollout_stop_stops_without_starting_new_intervention():
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.rollout_episode_logger = _ActiveRolloutLogger()

    app.handle_command("web:rollout_stop", _config(), runtime, session)

    assert session.status is SessionStatus.READY
    assert runtime.rollout_save_ready is True
    assert runtime.rollout_intervention_active is False
    assert runtime.transport.started == 0


def test_rollout_stop_interrupts_sync_motion_and_marks_save_ready():
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.rollout_episode_logger = _ActiveRolloutLogger()
    runtime.command_queue = queue.Queue()
    runtime.command_queue.put("web:rollout_stop")

    interrupted = control.poll_motion_commands(
        _config(),
        runtime,
        session,
    )

    assert interrupted is True
    assert session.interrupt_requested is True
    assert runtime.rollout_save_ready is True
    assert runtime.rollout_intervention_active is False
    assert runtime.transport.started == 0


def test_rollout_save_rejects_active_intervention(monkeypatch):
    calls = []
    runtime, session = _runtime_and_session()
    runtime.rollout_intervention_active = True

    monkeypatch.setattr(
        app,
        "save_rollout_episode",
        lambda runtime, session: calls.append(("save", True)),
    )

    app.handle_command("web:rollout_save", _config(), runtime, session)

    assert calls == []
    assert session.last_error == "Continue or abandon the active intervention before saving"


def test_rollout_save_button_stops_running_policy_and_saves(monkeypatch):
    calls = []
    runtime, session = _runtime_and_session()
    runtime.rollout_episode_logger = _ActiveRolloutLogger()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    monkeypatch.setattr(
        app,
        "save_rollout_episode",
        lambda runtime, session: calls.append((session.status, runtime.rollout_save_ready)) or True,
    )
    monkeypatch.setattr(
        app,
        "run_reset",
        lambda config, runtime, session: calls.append(("reset", session.status)),
    )

    app.handle_command("web:rollout_save", _config(), runtime, session)

    assert calls == [(SessionStatus.READY, True), ("reset", SessionStatus.READY)]


def test_rollout_save_failure_does_not_reset(monkeypatch):
    calls = []
    runtime, session = _runtime_and_session()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.READY
    monkeypatch.setattr(app, "save_rollout_episode", lambda runtime, session: False)
    monkeypatch.setattr(app, "run_reset", lambda *args: calls.append("reset"))

    app.handle_command("web:rollout_save", _config(), runtime, session)

    assert calls == []


def test_rollout_save_interrupts_sync_policy_and_requeues_save():
    runtime, session = _runtime_and_session()
    runtime.rollout_episode_logger = _ActiveRolloutLogger()
    runtime.command_queue = queue.Queue()
    session.mode = SessionMode.REAL
    session.status = SessionStatus.RUNNING
    runtime.command_queue.put("web:rollout_save")

    assert control.poll_motion_commands(_config(), runtime, session) is True

    assert runtime.rollout_save_ready is True
    assert runtime.command_queue.get_nowait() == "web:rollout_save"


def test_record_executed_action_uses_raw_snapshot_without_get_frame():
    transport = _RawTransport()
    logger = _RawEpisodeLogger()
    runtime = RuntimeState(robot=_robot(), transport=transport)
    runtime.rollout_episode_logger = logger
    action = np.array([1.0, 2.0], dtype=np.float32)

    recording.record_executed_action(_config(), SessionState(), action, runtime)

    assert transport.get_frame_calls == 0
    assert len(logger.raw) == 1
    assert logger.raw[0][0].timestamp == 1.0
    np.testing.assert_allclose(logger.raw[0][1], action)
    np.testing.assert_allclose(logger.raw[0][2], [3.0, 4.0])


def test_record_executed_action_waits_for_raw_snapshot():
    class _DelayedRawTransport(_RawTransport):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
            self.calls += 1
            if self.calls == 1:
                return None
            return super().acquire_collection_raw()

    transport = _DelayedRawTransport()
    logger = _RawEpisodeLogger()
    runtime = RuntimeState(robot=_robot(), transport=transport)
    runtime.rollout_episode_logger = logger

    recording.record_executed_action(_config(), SessionState(), np.array([1.0, 2.0]), runtime)

    assert transport.calls >= 2
    assert len(logger.raw) == 1
