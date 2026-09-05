from __future__ import annotations

import json
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import robots  # noqa: F401
from core.app import handlers
from core.app import run as app
from core.app.handlers.space import build_space
from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict
from core.registry import ROBOT_REGISTRY
from core.types import Observation
from tests.integration.web._harness import console_config
from transport.base import TransportBridge
from transport.dataset import DatasetTransport

pytestmark = pytest.mark.integration


def _joint_config(robot_type: str = "agilex_piper") -> ConfigDict:
    """Console config with the default JointState action_space and no gripper snap."""
    cfg = console_config(robot_type=robot_type)
    cfg.robot.gripper_threshold = None
    return cfg


class _NoopRate:
    def sleep(self) -> None:
        return None


class _FakeTransport(TransportBridge):
    def __init__(self) -> None:
        self.published: list[tuple[str, np.ndarray]] = []
        self.latest_qpos: np.ndarray | None = None

    def get_frame(self) -> Observation | None:
        return None

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        self.published.append((target, np.asarray(action, dtype=np.float32).copy()))

    def get_latest_qpos(self) -> np.ndarray | None:
        return None if self.latest_qpos is None else self.latest_qpos.copy()

    def create_rate(self, hz: float) -> _NoopRate:
        _ = hz
        return _NoopRate()

    def close(self) -> None:
        return None


class _ReplaySource:
    episode_id = 0

    def __init__(self, qpos: np.ndarray) -> None:
        self._qpos = qpos
        self.frame_index = 0
        self.n_steps = len(qpos)

    def advance(self) -> bool:
        if self.frame_index >= self.n_steps - 1:
            return False
        self.frame_index += 1
        return True

    def seek(self, index: int) -> None:
        self.frame_index = max(0, min(index, self.n_steps - 1))

    def get_scene_qpos(self, index: int | None = None) -> np.ndarray:
        idx = self.frame_index if index is None else index
        return self._qpos[idx]

    def close(self) -> None:
        return None


def test_replay_run_after_halt_continues_from_current_frame():
    qpos = np.asarray(
        [
            [0.0] * 14,
            [0.1] * 14,
            [0.2] * 14,
            [0.3] * 14,
        ],
        dtype=np.float32,
    )
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=_FakeTransport(),
    )
    runtime.replay_source = cast(DatasetTransport, _ReplaySource(qpos))
    runtime.replay_trajectory = qpos.copy()
    runtime.replay_source.seek(2)
    session = SessionState(
        mode=SessionMode.SIM,
        status=SessionStatus.RUNNING,
        is_setup_done=True,
        selected_task="replay task",
        step_index=2,
    )

    app.handle_command("web:halt", ConfigDict(), runtime, session)
    assert session.status is SessionStatus.READY

    app.handle_command("web:run", ConfigDict(), runtime, session)

    assert session.status is SessionStatus.RUNNING
    assert session.step_index == 2
    assert runtime.replay_source.frame_index == 2


def test_tab_switch_away_from_replay_unmounts_source():
    """Leaving REPLAY for a non-replay tab must drop replay_source.

    Otherwise is_replay() stays true on the next tab and EVAL/DEBUG publish the recorded
    trajectory instead of querying the policy.
    """
    qpos = np.asarray([[0.0] * 14, [0.1] * 14], dtype=np.float32)
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=_FakeTransport(),
    )
    runtime.replay_source = cast(DatasetTransport, _ReplaySource(qpos))
    runtime.replay_trajectory = qpos.copy()
    session = SessionState(mode=SessionMode.SIM, selected_task="replay task")

    app.handle_command("web:tab_switch:eval", _joint_config(), runtime, session)

    assert runtime.replay_source is None
    assert handlers.is_replay(runtime) is False


def test_eef_replay_load_prefers_action_eef_when_action_default_is_joint(tmp_path):
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    info = {
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.qpos": {"dtype": "float32", "shape": [14]},
            "observations.state.eef": {"dtype": "float32", "shape": [16]},
            "action": {"dtype": "float32", "shape": [14]},
            "action_eef": {"dtype": "float32", "shape": [16]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "eef replay"}) + "\n",
        encoding="utf-8",
    )
    action = np.asarray([[1.0] * 14, [2.0] * 14], dtype=np.float32)
    state_eef = np.asarray([[30.0] * 16, [40.0] * 16], dtype=np.float32)
    action_eef = np.asarray([[10.0] * 16, [20.0] * 16], dtype=np.float32)
    qpos = np.asarray([[0.0] * 14, [0.5] * 14], dtype=np.float32)
    table = pa.table(
        {
            "observation.qpos": pa.array(qpos.tolist(), type=pa.list_(pa.float32())),
            "observations.state.eef": pa.array(state_eef.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "action_eef": pa.array(action_eef.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array([0.0, 0.1], type=pa.float32()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=_FakeTransport(),
    )
    session = SessionState()
    config = _joint_config()
    config.inference_cfg.action_space = build_space(
        dict(type="EEFPose", n_arms=2, rotation="quat", include_gripper=True)
    )

    handlers.load_replay_dataset(
        str(tmp_path),
        0,
        config,
        session,
        runtime,
        state_key="observation.qpos",
        action_key="action",
        action_mode="eef",
    )

    assert runtime.replay_trajectory is not None
    np.testing.assert_allclose(runtime.replay_trajectory, action_eef)
    assert runtime.replay_action_key == "action_eef"
    assert runtime.replay_action_mode == "eef"
    assert runtime.replay_source is not None
    np.testing.assert_allclose(runtime.replay_source.get_scene_qpos(1), qpos[1])
    series = runtime.replay_source.series()
    np.testing.assert_allclose(series["state"], state_eef)
    np.testing.assert_allclose(series["action"], action_eef)
    assert series["state_names"] == [
        "left_arm.x",
        "left_arm.y",
        "left_arm.z",
        "left_arm.qw",
        "left_arm.qx",
        "left_arm.qy",
        "left_arm.qz",
        "left_arm.gripper",
        "right_arm.x",
        "right_arm.y",
        "right_arm.z",
        "right_arm.qw",
        "right_arm.qx",
        "right_arm.qy",
        "right_arm.qz",
        "right_arm.gripper",
    ]
    assert series["action_names"] == series["state_names"]
    assert session.last_error == ""


def test_eef_replay_with_joint_config_fills_gripper_and_solves_ik(monkeypatch):
    class _FakeUr5eSolver:
        def __init__(self, initial_qpos_groups, dt: float) -> None:
            _ = initial_qpos_groups, dt
            self.chunks = []

        def solve_chunk(self, chunk: np.ndarray, seed_qpos: np.ndarray | None = None) -> np.ndarray:
            _ = seed_qpos
            self.chunks.append(np.asarray(chunk, dtype=np.float32).copy())
            return np.zeros((1, 7), dtype=np.float32)

    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("ur5e"),
        transport=_FakeTransport(),
    )
    monkeypatch.setattr(runtime.robot, "build_kinematics", lambda **kw: _FakeUr5eSolver(**kw))
    runtime.replay_source = cast(
        DatasetTransport,
        _ReplaySource(np.asarray([[0.1, -0.2, 0.3, -1.0, 0.5, 1.2, 0.75]], dtype=np.float32)),
    )
    runtime.replay_trajectory = np.asarray(
        [[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    runtime.replay_action_key = "action_eef"
    runtime.replay_action_mode = "eef"
    session = SessionState(
        mode=SessionMode.SIM,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="replay task",
    )

    assert handlers.publish_next_action(_joint_config("ur5e"), runtime, session)

    solver = runtime.ik_solver
    assert solver is not None
    np.testing.assert_allclose(
        solver.chunks[0][0],
        [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.75],
    )
