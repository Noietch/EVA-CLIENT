"""Verify prediction alignment when execution horizons exceed returned chunks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.config import ConfigDict
from tools import openloop

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_openloop_bounds_steps_by_returned_actions(tmp_path, monkeypatch, horizon):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    output = tmp_path / "output"
    config = ConfigDict(
        transport={
            "dataset_dir": str(dataset),
            "episode_id": 0,
            "dataset_keys": {"action_key": "action"},
        },
        robot={"type": "test"},
        policy={"type": "test"},
        openloop={"output_dir": str(output), "execute_horizon": horizon},
    )
    visited = []

    class Transport:
        n_steps = 5
        current_task = "test"

        def __init__(self, *args):
            self.step = 0
            self.closed = False

        def get_action_trajectory(self):
            return np.arange(5, dtype=np.float32)[:, None]

        def seek(self, step):
            visited.append(step)
            self.step = step

        def get_frame(self):
            return SimpleNamespace(state_qpos=np.array([self.step]), images={})

        def close(self):
            self.closed = True

    class Client:
        metadata = {}

        def __init__(self):
            self.closed = False

        def reset(self):
            pass

        def infer(self, observation):
            step = observation[0]
            return {"actions": np.array([[step], [step + 1]], dtype=np.float32)}

        def close(self):
            self.closed = True

    # Feed bounded predictions through the real reporting path
    transport = Transport()
    client = Client()
    robot = SimpleNamespace(build_observation=lambda images, state, task: state)
    monkeypatch.setattr(openloop, "load_config", lambda path: config)
    monkeypatch.setattr(openloop.ROBOT_REGISTRY, "build", lambda kind: robot)
    monkeypatch.setattr(openloop, "DatasetTransport", lambda *args: transport)
    monkeypatch.setattr(openloop, "_connect_policy", lambda config, process: client)
    openloop.run(tmp_path / "config.py")

    # Every reported step must correspond to one real prediction and target
    summary = json.loads((output / "summary.json").read_text())
    assert summary["steps"] == 5
    assert summary["mae"] == 0.0
    assert visited == (list(range(5)) if horizon == 1 else [0, 2, 4])
    assert len((output / "trajectory.jsonl").read_text().splitlines()) == 5
    assert (output / "open_loop.png").is_file()
    assert transport.closed and client.closed
