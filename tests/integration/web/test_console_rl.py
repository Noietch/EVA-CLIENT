from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from core.app.console import server as console_server
from core.app.rl import record_rl_sample, submit_rl_critic
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


def test_rl_routes_select_models_and_setup_independent_policy_and_critic(console, tmp_path):
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
    assert selected["selected_critic_slot"] == 0

    console.do("/api/rl/setup")
    status = console.status()
    assert status["cli_mode"] == "real"
    assert status["selected_task"] == "pack the phone"
    assert status["selected_strategy"] == "sync"
    assert status["policy_connected"] is True
    assert status["rl"]["critic_connected"] is True
    assert status["is_setup_done"] is True


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
    console.do("/api/tab_switch", {"tab": "rl"})
    console.do("/api/rl/select_task", {"task": "pack the phone"})
    console.do("/api/rl/select_policy", {"slot": 0})
    console.do("/api/rl/select_critic", {"slot": 0})
    console.do("/api/rl/setup")
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
