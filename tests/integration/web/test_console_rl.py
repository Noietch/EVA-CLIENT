from __future__ import annotations

import copy
import json
import threading
import time

import numpy as np
import pytest

from core.app import run as app
from core.app.console import server as console_server
from core.app.rl import build_rl_critic_observation, record_rl_sample, submit_rl_critic
from core.app.state import SessionStatus
from core.config import ConfigDict


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


def test_rl_routes_setup_policy_then_select_optional_critic(console, tmp_path):
    _configure_rl(console, tmp_path)

    cfg = console.get("/api/config").json["rl"]
    assert cfg["backend_ready"] is True
    assert cfg["policies"] == [{"slot": 0, "name": "policy-a"}]
    assert cfg["critics"] == [{"slot": 0, "name": "critic-a", "type": "mock"}]

    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/select_critic", {"slot": 0})
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


def test_rl_saved_data_path_uses_rl_storage_before_setup(console, tmp_path):
    _configure_rl(console, tmp_path)

    console.do("/api/tab_switch", {"tab": "rl"})

    assert console.status()["rollout"]["dataset_dir"] == str(tmp_path / "rl")


def test_rl_setup_and_rollout_do_not_require_a_critic(console, tmp_path):
    _configure_rl(console, tmp_path)

    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/setup")

    status = console.status()
    assert status["policy_connected"] is True
    assert status["is_setup_done"] is True
    assert status["rl"]["critic_connected"] is False
    assert status["last_error"] == ""


def test_rl_duplicate_setup_does_not_repeat_robot_setup(console, tmp_path, monkeypatch):
    _configure_rl(console, tmp_path)
    calls = 0
    original = app.run_setup

    def counted_setup(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(app, "run_setup", counted_setup)
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/setup")
    console.do("/api/rl/setup")

    assert calls == 1
    assert console.status()["is_setup_done"] is True


def test_rl_critic_can_connect_after_policy_setup_without_robot_reset(console, tmp_path):
    _configure_rl(console, tmp_path)

    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/setup")
    assert console.status()["is_setup_done"] is True

    console.do("/api/rl/select_critic", {"slot": 0})
    status = console.status()
    assert status["is_setup_done"] is True
    assert status["rl"]["selected_critic_slot"] == 0
    assert status["rl"]["critic_connected"] is True
    deadline = time.monotonic() + 2.0
    while console.runtime.rl_critic_runner.series()["n"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert console.runtime.rl_critic_runner.series()["n"] == 1


def test_rl_critic_selected_during_setup_is_rejected(console, tmp_path):
    _configure_rl(console, tmp_path)

    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/select_critic", {"slot": 0})
    assert console.status()["rl"]["selected_critic_slot"] is None
    assert console.status()["last_error"] == "RL Critic can only be selected after setup"

    console.do("/api/rl/setup")

    status = console.status()
    assert status["is_setup_done"] is True
    assert status["rl"]["selected_critic_slot"] is None
    assert status["rl"]["critic_connected"] is False


def test_rl_reset_route_clears_setup_and_critic_series(console, tmp_path):
    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/setup")
    console.do("/api/rl/select_critic", {"slot": 0})
    assert console.status()["is_setup_done"] is True

    console.do("/api/rl/reset")
    status = console.status()
    assert status["is_setup_done"] is True
    assert status["rl"]["selected_critic_slot"] is None
    assert status["rl"]["critic_connected"] is False
    assert status["rl"]["critic_samples"] == 0


def test_rl_reset_is_allowed_during_rollout_but_rejected_during_intervention(
    console, tmp_path, monkeypatch
):
    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/setup")

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


def test_rl_saved_episode_qc_reuses_lerobot_qc_route(console, tmp_path):
    dataset_dir = tmp_path / "rl"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_path.write_text(json.dumps({"episode_index": 3, "length": 12}) + "\n")

    response = console.post(
        "/api/qc_mark",
        {
            "dataset_dir": str(dataset_dir),
            "episode": "3",
            "verdict": "pass",
            "note": "three cameras synchronized",
        },
    )

    assert response.status == 200
    assert response.json == {"ok": True, "episode": 3, "verdict": "pass"}
    row = json.loads(episodes_path.read_text().strip())
    assert row["qc_verdict"] == "pass"
    assert row["qc_note"] == "three cameras synchronized"


def test_rl_critic_observation_uses_flat_openpi_image_keys(console):
    images = {
        "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((4, 4, 3), dtype=np.uint8),
        "cam_right_wrist": np.full((4, 4, 3), 2, dtype=np.uint8),
    }
    raw = build_rl_critic_observation(
        console.runtime,
        {"state": np.zeros(14, dtype=np.float32), "images": images, "prompt": "pack the phone"},
    )

    assert "observations.state.qpos" in raw
    assert "observation.images.cam_high" in raw
    assert "observation.images.cam_left_wrist" in raw
    assert "observation.images.cam_right_wrist" in raw


def test_rl_critic_actions_follow_server_horizon(console):
    class RecordingRunner:
        def __init__(self):
            self.actions = None

        def submit(self, _observation, actions, _timestamp, _source):
            self.actions = np.asarray(actions).copy()

    runner = RecordingRunner()
    console.runtime.rl_active = True
    console.runtime.rl_critic_runner = runner
    console.runtime.rl_critic_action_horizon = 4

    intervention = np.array([[0.1, 0.2]], dtype=np.float32)
    submit_rl_critic(console.runtime, {"state": intervention[0]}, intervention, "intervention")
    assert runner.actions.shape == (4, 2)
    np.testing.assert_array_equal(runner.actions, np.repeat(intervention, 4, axis=0))

    policy_chunk = np.arange(8, dtype=np.float32).reshape(4, 2)
    submit_rl_critic(console.runtime, {"state": policy_chunk[0]}, policy_chunk, "policy")
    np.testing.assert_array_equal(runner.actions, policy_chunk)

    oversized_chunk = np.arange(12, dtype=np.float32).reshape(6, 2)
    submit_rl_critic(console.runtime, {"state": oversized_chunk[0]}, oversized_chunk, "policy")
    np.testing.assert_array_equal(runner.actions, oversized_chunk[:4])


def test_rl_critic_without_action_horizon_keeps_original_chunk(console):
    class RecordingRunner:
        def __init__(self):
            self.actions = None

        def submit(self, _observation, actions, _timestamp, _source):
            self.actions = np.asarray(actions).copy()

    runner = RecordingRunner()
    console.runtime.rl_active = True
    console.runtime.rl_critic_runner = runner
    console.runtime.rl_critic_action_horizon = None
    actions = np.array([[0.1, 0.2]], dtype=np.float32)

    submit_rl_critic(console.runtime, {"state": actions[0]}, actions, "intervention")

    np.testing.assert_array_equal(runner.actions, actions)


def test_rl_series_streams_control_source_and_critic_incrementally(console, tmp_path):
    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/setup")
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
    deadline = time.monotonic() + 2.0
    while runner.series()["n"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)

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

    console.do("/api/tab_switch", {"tab": "replay"})
    assert console.runtime.rl_active is False
    assert console.runtime.rl_critic_runner is None


def test_rl_replay_switch_closes_previous_source(console, tmp_path, monkeypatch):
    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/select_critic", {"slot": 0})
    console.do("/api/rl/setup")

    class FakeSource:
        def __init__(self, episode: int):
            self.episode = episode
            self.n_steps = 3
            self.fps = 15
            self.current_task = "pack the phone"
            self.closed = False

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

        def close(self):
            self.closed = True

    sources = {}

    def open_episode(_ctx, _dataset_dir, episode):
        source = FakeSource(int(episode))
        sources[source.episode] = source
        return source, {}

    monkeypatch.setattr(console_server, "_open_review_episode", open_episode)

    first = console.post("/api/rl/review_episode", {"dataset_dir": "rl", "episode": 0})
    second = console.post("/api/rl/review_episode", {"dataset_dir": "rl", "episode": 1})

    assert first.status == 200
    assert second.status == 200
    assert sources[0].closed is True
    assert sources[1].closed is False
    assert console.runtime.rl_replay_source is sources[1]
    assert console.runtime.rl_replay_sources == [sources[1]]


def test_rl_replay_switch_waits_for_active_critic_frame_builder(
    console, tmp_path, monkeypatch
):
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


def test_rl_replay_critic_uses_critic_action_horizon(console, tmp_path, monkeypatch):
    _configure_rl(console, tmp_path)
    console.do("/api/tab_switch", {"tab": "rl"})

    class FakeSource:
        n_steps = 2
        fps = 15
        current_task = "pack the phone"

        def __init__(self):
            self.actions = np.arange(6, dtype=np.float32).reshape(3, 2)

        def series(self):
            values = [[0.0, 0.0]] * self.n_steps
            return {
                "timestamp": [0.0, 1 / 15],
                "state": values,
                "action": values,
                "action_names": ["q0", "q1"],
                "state_names": ["q0", "q1"],
                "control_source": ["policy"] * self.n_steps,
                "intervention": [False] * self.n_steps,
                "intervention_segment_index": [-1] * self.n_steps,
            }

        def seek(self, _frame):
            return None

        def get_frame(self):
            return object()

        def get_action_trajectory(self):
            return self.actions.copy()

        def close(self):
            return None

    class RecordingRunner:
        def __init__(self):
            self.actions = None

        def reset_series(self):
            return None

        def submit_builder(self, builder, _timestamp, _source):
            self.actions = builder()[1]

    source = FakeSource()
    runner = RecordingRunner()
    monkeypatch.setattr(console_server, "_open_review_episode", lambda *_: (source, {}))
    monkeypatch.setattr(
        console_server, "build_policy_observation", lambda *_: {"state": np.zeros(2)}
    )
    console.runtime.rl_critic_runner = runner
    console.runtime.rl_critic_action_horizon = 4

    loaded = console.post("/api/rl/review_episode", {"dataset_dir": "rl", "episode": 0})
    assert loaded.status == 200
    response = console.post("/api/rl/replay_critic", {"frame": 0})

    assert response.status == 200
    assert runner.actions.shape == (4, 2)
    np.testing.assert_array_equal(runner.actions, np.vstack([source.actions, source.actions[-1]]))
