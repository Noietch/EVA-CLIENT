from __future__ import annotations

import json
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from _harness import console_config

import robots  # noqa: F401
from core.app import handlers
from core.app import run as app
from core.app.handlers import control as control_handlers
from core.app.handlers.space import build_space
from core.app.state import RuntimeState, SessionMode, SessionState, SessionStatus
from core.config import ConfigDict
from core.registry import ROBOT_REGISTRY
from core.types import Observation
from transport.base import TransportBridge
from transport.dataset import DatasetTransport


def _eef_config(robot_type: str = "agilex_piper", n_arms: int = 2) -> ConfigDict:
    """Console config whose action_space is an EEFPose (quat + gripper)."""
    cfg = console_config(robot_type=robot_type)
    cfg.robot.gripper_threshold = None
    cfg.inference_cfg.action_space = build_space(
        dict(type="EEFPose", n_arms=n_arms, rotation="quat", include_gripper=True)
    )
    return cfg


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


def test_tab_switch_to_replay_keeps_source_mounted():
    """Switching INTO the REPLAY tab must not unmount a freshly loaded episode."""
    qpos = np.asarray([[0.0] * 14, [0.1] * 14], dtype=np.float32)
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=_FakeTransport(),
    )
    runtime.replay_source = cast(DatasetTransport, _ReplaySource(qpos))
    runtime.replay_trajectory = qpos.copy()
    session = SessionState(mode=SessionMode.SIM, selected_task="replay task")

    app.handle_command("web:tab_switch:replay", _joint_config(), runtime, session)

    assert runtime.replay_source is not None


def test_replay_setup_does_not_connect_policy(monkeypatch):
    qpos = np.asarray([[0.0] * 14, [0.1] * 14], dtype=np.float32)
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=_FakeTransport(),
    )
    runtime.replay_source = cast(DatasetTransport, _ReplaySource(qpos))
    runtime.replay_trajectory = qpos.copy()
    session = SessionState(
        mode=SessionMode.SIM,
        status=SessionStatus.UNSET,
        selected_task="replay task",
    )

    def fail_connect(*_args, **_kwargs):
        raise AssertionError("replay setup must not connect policy")

    monkeypatch.setattr(control_handlers, "connect_policy", fail_connect)

    assert handlers.run_setup(_joint_config(), runtime, session)
    assert session.is_setup_done is True
    assert runtime.policy is None


def test_replay_step_setup_prepares_real_robot_to_first_frame():
    transport = _FakeTransport()
    transport.latest_qpos = np.zeros(14, dtype=np.float32)
    first_frame = np.linspace(0.1, 1.4, 14, dtype=np.float32)
    qpos = np.stack([first_frame, first_frame + 0.1], axis=0)
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=transport,
    )
    runtime.replay_source = cast(DatasetTransport, _ReplaySource(qpos))
    runtime.replay_trajectory = qpos.copy()
    runtime.replay_action_mode = "joint"
    session = SessionState(
        mode=SessionMode.STEP,
        status=SessionStatus.UNSET,
        selected_task="replay task",
    )

    assert handlers.run_setup(_joint_config(), runtime, session)

    real_actions = [action for target, action in transport.published if target == "real"]
    assert real_actions
    np.testing.assert_allclose(real_actions[-1], first_frame, atol=1e-6)
    assert session.is_setup_done is True
    assert session.step_index == 0
    assert runtime.replay_source.frame_index == 0


def test_replay_step_playback_updates_sim_preview_for_each_frame():
    session = SessionState(
        mode=SessionMode.STEP,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="replay task",
    )

    class _InspectingTransport(_FakeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.preview_snapshots: list[np.ndarray | None] = []

        def create_rate(self, hz: float) -> _NoopRate:
            _ = hz
            transport = self

            class _Rate(_NoopRate):
                def sleep(self) -> None:
                    preview = session.sim_preview_qpos
                    transport.preview_snapshots.append(None if preview is None else preview.copy())

            return _Rate()

    transport = _InspectingTransport()
    qpos = np.asarray(
        [
            [0.1] * 14,
            [0.2] * 14,
            [0.3] * 14,
        ],
        dtype=np.float32,
    )
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("agilex_piper"),
        transport=transport,
    )
    runtime.replay_source = cast(DatasetTransport, _ReplaySource(qpos))
    runtime.replay_trajectory = qpos.copy()
    runtime.replay_action_mode = "joint"
    runtime.replay_exec_steps = 3
    config = _joint_config()

    assert handlers.publish_next_action(config, runtime, session)

    assert len(transport.preview_snapshots) == 3
    assert all(snapshot is not None for snapshot in transport.preview_snapshots)
    np.testing.assert_allclose(np.asarray(transport.preview_snapshots), qpos)
    assert session.sim_preview_qpos is not None
    np.testing.assert_allclose(session.sim_preview_qpos, qpos[-1])
    assert session.pending_real_chunk is not None
    np.testing.assert_allclose(session.pending_real_chunk, qpos)


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

    handlers.load_replay_dataset(
        str(tmp_path),
        0,
        _eef_config(),
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


def test_replay_load_resolves_relative_collection_qpos_dataset(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "collection" / "ur5e"
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    info = {
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.qpos": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "collection qc"}) + "\n",
        encoding="utf-8",
    )
    state = np.asarray([[0.0] * 14, [0.5] * 14], dtype=np.float32)
    action = np.asarray([[1.0] * 14, [2.0] * 14], dtype=np.float32)
    table = pa.table(
        {
            "observation.qpos": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
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
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(handlers.utils, "_REPO_ROOT", repo_root)
    monkeypatch.chdir(elsewhere)

    handlers.load_replay_dataset(
        "work_dirs/collection/ur5e", 0, _joint_config("ur5e"), session, runtime
    )

    assert session.last_error == ""
    assert runtime.replay_source is not None
    assert runtime.replay_trajectory is not None
    np.testing.assert_allclose(runtime.replay_trajectory, action)
    np.testing.assert_allclose(runtime.replay_source.get_scene_qpos(1), state[1])


def test_eef_replay_fills_missing_action_gripper_when_mode_is_eef(monkeypatch):
    class _FakeUr5eSolver:
        def __init__(self, initial_qpos_groups, dt: float) -> None:
            _ = initial_qpos_groups, dt
            self.chunks = []

        def solve_chunk(
            self, chunk: np.ndarray, seed_qpos: np.ndarray | None = None
        ) -> np.ndarray:
            _ = seed_qpos
            self.chunks.append(np.asarray(chunk, dtype=np.float32).copy())
            return np.zeros((1, 7), dtype=np.float32)

    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("ur5e"),
        transport=_FakeTransport(),
    )
    monkeypatch.setattr(
        runtime.robot, "build_kinematics", lambda **kw: _FakeUr5eSolver(**kw)
    )
    runtime.replay_source = cast(
        DatasetTransport,
        _ReplaySource(
            np.asarray([[0.1, -0.2, 0.3, -1.0, 0.5, 1.2, 0.75]], dtype=np.float32)
        ),
    )
    runtime.replay_trajectory = np.asarray(
        [[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    runtime.replay_action_key = "action"
    runtime.replay_action_mode = "eef"
    session = SessionState(
        mode=SessionMode.SIM,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="replay task",
    )
    assert handlers.publish_next_action(_eef_config("ur5e", n_arms=1), runtime, session)

    solver = runtime.ik_solver
    assert solver is not None
    np.testing.assert_allclose(
        solver.chunks[0][0],
        [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.75],
    )


def test_eef_replay_with_joint_config_fills_gripper_and_solves_ik(monkeypatch):
    class _FakeUr5eSolver:
        def __init__(self, initial_qpos_groups, dt: float) -> None:
            _ = initial_qpos_groups, dt
            self.chunks = []

        def solve_chunk(
            self, chunk: np.ndarray, seed_qpos: np.ndarray | None = None
        ) -> np.ndarray:
            _ = seed_qpos
            self.chunks.append(np.asarray(chunk, dtype=np.float32).copy())
            return np.zeros((1, 7), dtype=np.float32)

    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("ur5e"),
        transport=_FakeTransport(),
    )
    monkeypatch.setattr(
        runtime.robot, "build_kinematics", lambda **kw: _FakeUr5eSolver(**kw)
    )
    runtime.replay_source = cast(
        DatasetTransport,
        _ReplaySource(
            np.asarray([[0.1, -0.2, 0.3, -1.0, 0.5, 1.2, 0.75]], dtype=np.float32)
        ),
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


def test_joint_replay_does_not_treat_plain_action_as_missing_gripper(monkeypatch):
    class _FakeUr5eSolver:
        def __init__(self, initial_qpos_groups, dt: float) -> None:
            _ = initial_qpos_groups, dt

    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("ur5e"),
        transport=_FakeTransport(),
    )
    monkeypatch.setattr(runtime.robot, "build_kinematics", lambda **kw: _FakeUr5eSolver(**kw))
    runtime.replay_source = cast(
        DatasetTransport, _ReplaySource(np.zeros((1, 7), dtype=np.float32))
    )
    runtime.replay_trajectory = np.zeros((1, 7), dtype=np.float32)
    runtime.replay_action_key = "action"
    runtime.replay_action_mode = "joint"

    action = handlers._replay_action_at(_joint_config("ur5e"), runtime, 0)

    np.testing.assert_allclose(action, np.zeros(7, dtype=np.float32))
    assert runtime.ik_solver is None


def test_joint_replay_snaps_gripper_to_configured_open_value():
    runtime = RuntimeState(
        robot=ROBOT_REGISTRY.build("r1_lite"),
        transport=_FakeTransport(),
    )
    runtime.replay_trajectory = np.asarray(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]],
        dtype=np.float32,
    )
    runtime.replay_action_mode = "joint"
    config = _joint_config("r1_lite")
    config.robot.gripper_threshold = 50.0
    config.robot.gripper_open = 100.0
    config.robot.gripper_close = 0.0

    action = handlers._replay_action_at(config, runtime, 0)

    np.testing.assert_allclose(action[[6, 13]], [100.0, 100.0], atol=1e-6)
