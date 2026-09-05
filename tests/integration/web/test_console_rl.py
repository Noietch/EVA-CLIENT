from __future__ import annotations

import copy
import threading
import time
from unittest.mock import Mock

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.app import run as app
from core.app.console import server as console_server
from core.app.handlers import teleop
from core.app.handlers.recording import (
    begin_rollout_save_episode,
    record_client_rollout_intervention_step,
)
from core.app.operator_control import handle_teleop_operator_event
from core.app.rl import (
    record_rl_sample,
    submit_rl_critic,
)
from core.app.state import SessionStatus
from core.config import ConfigDict
from core.types import Observation
from teleop_client.base import QposCommand, TeleopOperatorEvent, TeleopResult, TeleopStatus

pytestmark = pytest.mark.integration


def _wait_until(predicate, *, timeout_s: float = 2.0, interval_s: float = 0.01) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(interval_s)


def _configure_rl(console, tmp_path) -> None:
    policy_config = copy.deepcopy(console.config)
    policy_config.policy.type = "mock"
    policy_config.policy.backend_options = ConfigDict(chunk_size=4)
    policy_config.rl = None
    policy_config.rl_cfg = None
    rl_cfg = ConfigDict(
        cli_mode="real",
        inference_strategy="sync",
        tasks=["pack the phone"],
        policies=[ConfigDict(name="policy-a", config=policy_config)],
        critics=[
            ConfigDict(
                name="critic-a",
                type="mock",
                host="127.0.0.1",
                port=9100,
                backend_options=ConfigDict(),
            )
        ],
        data=ConfigDict(
            format="lerobot",
            storage=ConfigDict(
                log_dir=str(tmp_path / "rl"),
                fps=15,
                save_queue_max=4,
                async_save=True,
                image_height=64,
                image_width=64,
            ),
        ),
        intervention=ConfigDict(control_mode="relative"),
    )
    console.config.rl = rl_cfg
    console.config.rl_cfg = rl_cfg


def _configure_client_intervention(console, tmp_path, results):
    _configure_rl(console, tmp_path)
    console.config.rl.intervention.source = "teleop_client"
    console.config.rl.data.storage.async_save = False
    console.config.collection.teleop = ConfigDict(
        control_source="client", client=ConfigDict(type="test")
    )
    client = Mock()
    client.poll.side_effect = list(results)
    client.validate_result.return_value = True
    client.status.return_value = TeleopStatus(source_type="test", connected=True, neutral=True)
    console.runtime.teleop_client = client
    console.runtime.teleop_execution = teleop.TeleopExecutionState(
        control_source="client", client_type="vr_webxr"
    )
    for path, body in (
        ("/api/tab_switch", {"tab": "rl"}),
        ("/api/rl/select_task", {"task": "pack the phone"}),
        ("/api/rl/select_policy", {"slot": 0}),
        ("/api/rl/setup", None),
        ("/api/rl/hil_enabled", {"enabled": True}),
        ("/api/rl/run", None),
    ):
        console.do(path, body)
    app.publish_next_action(console.runtime.active_config, console.runtime, console.session)
    return client


def _select_rl_policy(
    console,
    *,
    task: str = "pack the phone",
    policy_slot: int = 0,
    critic_slot: int | None = None,
    setup: bool = False,
) -> None:
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": task})
    console.do("/api/rl/select_policy", {"slot": policy_slot})
    if critic_slot is not None:
        console.do("/api/rl/select_critic", {"slot": critic_slot})
    if setup:
        console.do("/api/rl/setup")


def _publish_until_step_advances(console) -> None:
    def advanced() -> bool:
        app.publish_next_action(
            console.runtime.active_config,
            console.runtime,
            console.session,
        )
        return console.session.step_index > 0

    _wait_until(advanced, timeout_s=2.0, interval_s=0.005)


def test_rl_routes_setup_policy_then_select_optional_critic(console, tmp_path):
    _configure_rl(console, tmp_path)

    cfg = console.get("/api/config").json["rl"]
    assert cfg["backend_ready"] is True
    assert cfg["policies"] == [{"slot": 0, "name": "policy-a"}]
    assert cfg["critics"] == [{"slot": 0, "name": "critic-a", "type": "mock"}]

    _select_rl_policy(console, critic_slot=0)
    selected = console.status()["rl"]
    assert selected["active"] is True
    assert selected["selected_policy_slot"] == 0
    assert selected["selected_critic_slot"] is None

    console.do("/api/rl/setup")
    status = console.status()
    assert status["cli_mode"] == "real"
    assert status["selected_task"] == "pack the phone"
    assert status["selected_strategy"] == "sync"
    assert status["policy_connected"] is True
    assert status["rl"]["critic_connected"] is False
    assert status["is_setup_done"] is True

    console.do("/api/rl/select_critic", {"slot": 0})
    status = console.status()
    assert status["rl"]["selected_critic_slot"] == 0
    assert status["rl"]["critic_connected"] is True


def test_entering_rl_discards_an_active_ordinary_rollout(console, tmp_path):
    normal_dir = tmp_path / "normal"
    console.config.rollout.storage.enabled = True
    console.config.rollout.storage.log_dir = str(normal_dir)
    begin_rollout_save_episode(console.config, console.runtime, console.session)
    ordinary_logger = console.runtime.rollout_episode_logger
    assert ordinary_logger is not None
    assert ordinary_logger.has_active_episode
    assert console.runtime.collection_capture_runner is not None

    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})

    assert not ordinary_logger.has_active_episode
    assert console.runtime.rollout_episode_logger is ordinary_logger
    assert console.runtime.collection_capture_runner is None


def test_leaving_rl_releases_its_rollout_logger(console, tmp_path):
    _configure_rl(console, tmp_path)
    _select_rl_policy(console, setup=True)
    active = console.runtime.active_config
    assert active is not None
    begin_rollout_save_episode(active, console.runtime, console.session)
    assert console.runtime.rollout_episode_logger is not None
    assert console.runtime.collection_capture_runner is not None

    console.do("/api/tab_switch", {"tab": "debug"})

    assert console.runtime.rollout_episode_logger is None
    assert console.runtime.collection_capture_runner is None


def test_rl_async_reset_auto_setup_then_start_publishes_action(console, tmp_path):
    _configure_rl(console, tmp_path)
    policy_config = console.config.rl.policies[0].config
    policy_config.inference_strategies = ConfigDict(
        async_=ConfigDict(
            type="AsyncLinearOverlapInferStrategy",
            args=ConfigDict(latency_k=0),
        )
    )
    console.config.rl.inference_strategy = "async_"

    _select_rl_policy(console, setup=True)
    console.do("/api/rl/run")
    _publish_until_step_advances(console)
    assert console.session.step_index > 0

    console.do("/api/rl/reset")
    status = console.status()
    assert status["is_setup_done"] is True
    assert status["session_status"] == "ready"
    assert status["step_index"] == 0
    assert status["rl"]["selected_policy_slot"] == 0

    console.do("/api/rl/run")
    _publish_until_step_advances(console)

    assert console.session.status is SessionStatus.RUNNING
    assert console.runtime.infer_strategy.is_loop_running()
    assert console.session.step_index > 0


def test_rl_reset_is_allowed_during_rollout_but_rejected_during_intervention(
    console, tmp_path, monkeypatch
):
    _configure_rl(console, tmp_path)
    _select_rl_policy(console, setup=True)

    console.session.status = SessionStatus.RUNNING
    console.do("/api/rl/reset")
    assert console.status()["is_setup_done"] is True

    console.runtime.rollout_intervention_active = True
    reset_calls = 0

    def counted_reset(*_args, **_kwargs):
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(app, "run_reset", counted_reset)
    console.do("/api/rl/reset")
    assert reset_calls == 0
    assert console.runtime.rollout_intervention_active is True
    assert console.status()["last_error"] == (
        "Continue or abandon the active intervention before reset"
    )


def test_rl_control_routes_are_rejected_outside_rl_tab(console, tmp_path):
    _configure_rl(console, tmp_path)

    response = console.post("/api/rl/setup")

    assert response.status == 409
    assert response.json == {"ok": False, "error": "RL tab is not active"}

    series = console.get("/api/rl/series")
    assert series.status == 409
    assert series.json == {"ok": False, "error": "RL tab is not active"}


def test_rl_series_streams_control_source_and_critic_incrementally(console, tmp_path):
    _configure_rl(console, tmp_path)
    _select_rl_policy(console, setup=True)
    console.do("/api/rl/select_critic", {"slot": 0})
    runner = console.runtime.rl_critic_runner
    assert runner is not None
    runner.reset_series()

    state = np.array([0.1, 0.2], dtype=np.float32)
    action = np.array([0.15, 0.25], dtype=np.float32)
    record_rl_sample(console.runtime, state, action, "intervention", 2.5, 3)
    submit_rl_critic(
        console.runtime,
        {"state": state},
        action[None, :],
        "intervention",
        2.5,
    )
    _wait_until(lambda: runner.series()["n"] >= 1)

    series = console.get("/api/rl/series?since=0&critic_since=0").json

    assert series["n"] == 1
    assert series["timestamp"] == [2.5]
    assert series["state"] == [[pytest.approx(0.1), pytest.approx(0.2)]]
    assert series["action"] == [[pytest.approx(0.15), pytest.approx(0.25)]]
    assert series["control_source"] == ["intervention"]
    assert series["intervention"] == [True]
    assert series["intervention_segment_index"] == [3]
    assert series["critic"]["n"] == 1
    assert series["critic"]["source"] == ["intervention"]

    lightweight = console.get("/api/rl/series?samples=0&critic_since=0").json
    assert lightweight["n"] == 1
    assert lightweight["critic"]["n"] == 1
    assert "timestamp" not in lightweight
    assert "state" not in lightweight
    assert "action" not in lightweight

    console.do("/api/tab_switch", {"tab": "replay"})
    assert console.runtime.rl_active is False
    assert console.runtime.rl_critic_runner is None


def test_rl_replay_switch_waits_for_active_critic_frame_builder(console, tmp_path, monkeypatch):
    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})
    read_started = threading.Event()
    allow_read = threading.Event()

    class FakeSource:
        def __init__(self, episode: int):
            self.episode = episode
            self.n_steps = 3
            self.fps = 15
            self.current_task = "pack the phone"
            self.reading = False
            self.closed = False
            self.closed_while_reading = False
            self.actions = np.zeros((self.n_steps, 2), dtype=np.float32)

        def series(self):
            values = [[float(self.episode), 0.0]] * self.n_steps
            return {
                "timestamp": [0.0, 1 / 15, 2 / 15],
                "state": values,
                "action": values,
                "action_names": ["q0", "q1"],
                "state_names": ["q0", "q1"],
                "control_source": ["policy"] * self.n_steps,
                "intervention": [False] * self.n_steps,
                "intervention_segment_index": [-1] * self.n_steps,
            }

        def seek(self, _frame):
            pass

        def get_frame(self):
            self.reading = True
            read_started.set()
            assert allow_read.wait(timeout=2.0)
            self.reading = False
            return object()

        def get_action_trajectory(self):
            return self.actions.copy()

        def close(self):
            self.closed_while_reading = self.reading
            self.closed = True

    class AsyncRunner:
        def __init__(self):
            self.thread = None
            self.error = None

        def reset_series(self):
            pass

        def submit_builder(self, builder, _timestamp, _source):
            def run_builder():
                try:
                    builder()
                except Exception as error:
                    self.error = error

            self.thread = threading.Thread(target=run_builder)
            self.thread.start()

    sources = {}

    def open_episode(_ctx, _dataset_dir, episode):
        source = FakeSource(int(episode))
        sources[source.episode] = source
        return source, {}

    runner = AsyncRunner()
    monkeypatch.setattr(console_server, "_open_review_episode", open_episode)
    monkeypatch.setattr(
        console_server, "build_policy_observation", lambda *_: {"state": np.zeros(2)}
    )
    console.runtime.rl_critic_runner = runner
    console.runtime.rl_critic_action_horizon = 1

    loaded = console.post("/api/rl/review_episode", {"dataset_dir": "rl", "episode": 0})
    assert loaded.status == 200
    critic = console.post("/api/rl/replay_critic", {"frame": 0})
    assert critic.status == 200
    assert read_started.wait(timeout=1.0)

    switched = threading.Event()

    def switch_episode():
        console.post("/api/rl/review_episode", {"dataset_dir": "rl", "episode": 1})
        switched.set()

    switcher = threading.Thread(target=switch_episode)
    switcher.start()
    assert not switched.wait(timeout=0.1)
    assert sources[0].closed is False

    allow_read.set()
    switcher.join(timeout=2.0)
    assert runner.thread is not None
    runner.thread.join(timeout=2.0)
    assert not switcher.is_alive()
    assert not runner.thread.is_alive()
    assert runner.error is None
    assert sources[0].closed is True
    assert sources[0].closed_while_reading is False
    assert console.runtime.rl_replay_source is sources[1]


def test_rl_client_intervention_accept_and_save_persists_rollout(console, tmp_path, monkeypatch):
    client = _configure_client_intervention(
        console,
        tmp_path,
        [TeleopResult.from_command(QposCommand(np.full(14, 0.02, dtype=np.float32)))],
    )
    monkeypatch.setattr(
        teleop,
        "forward_canonical_eef",
        lambda _config, runtime, _qpos: np.zeros(
            8 * len(runtime.robot.arm_groups), dtype=np.float32
        ),
    )
    event = TeleopOperatorEvent("test", 1, "intervention_toggle", 1.0)
    handle_teleop_operator_event(
        event,
        console.runtime.active_config or console.config,
        console.runtime,
        console.session,
        dispatch=app.handle_command,
    )
    client.acknowledge_event.assert_called_with(
        event,
        accepted=True,
        message="RL intervention started",
    )
    assert console.runtime.rollout_intervention_active is True
    monkeypatch.setattr(
        console.runtime.transport,
        "get_frame",
        lambda: Observation(
            timestamp=2.5,
            images={
                "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
                "cam_left_wrist": np.ones((4, 4, 3), dtype=np.uint8),
                "cam_right_wrist": np.full((4, 4, 3), 2, dtype=np.uint8),
            },
            state_qpos=np.full(14, 0.01, dtype=np.float32),
        ),
    )
    published = teleop.step_rollout_teleop(
        console.runtime.active_config,
        console.runtime,
        console.session,
        now=2.5,
    )
    assert published is not None
    assert record_client_rollout_intervention_step(
        console.runtime.active_config,
        console.runtime,
        console.session,
        published,
    )
    console.do("/api/rl/accept")
    assert (
        console.runtime.rollout_intervention_active,
        console.runtime.rollout_intervention_active_segment is None,
        len(console.runtime.rollout_intervention_segments),
    ) == (False, True, 1)
    assert console.do("/api/rl/save").json == {"ok": True}
    rollout_table = pq.read_table(
        str(tmp_path / "rl" / "data" / "chunk-000" / "episode_000000.parquet")
    )
    assert "intervention" in rollout_table.column("control_source").to_pylist()
    assert True in rollout_table.column("intervention").to_pylist()
    assert 0 in rollout_table.column("intervention_segment_index").to_pylist()
    assert client.reset.call_count >= 2


def test_tab_switch_away_from_rl_clears_active_client_intervention(console, tmp_path):
    client = _configure_client_intervention(console, tmp_path, [])
    event = TeleopOperatorEvent("test", 2, "intervention_toggle", 2.0)
    handle_teleop_operator_event(
        event,
        console.runtime.active_config or console.config,
        console.runtime,
        console.session,
        dispatch=app.handle_command,
    )
    assert console.runtime.rollout_intervention_active is True
    console.do("/api/tab_switch", {"tab": "debug"})
    assert (
        console.runtime.rollout_intervention_active,
        console.runtime.rollout_intervention_active_segment is None,
        console.runtime.rl_active,
        console.runtime.teleop_execution.active,
        console.session.status,
        (tmp_path / "rl" / "data" / "chunk-000" / "episode_000000.parquet").exists(),
    ) == (False, True, False, False, SessionStatus.UNSET, False)
    assert client.reset.call_count >= 2
