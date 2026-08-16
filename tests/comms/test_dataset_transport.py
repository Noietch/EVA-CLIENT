from __future__ import annotations

import json
import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.config import ConfigDict
from core.datasets import open_dataset
from robots.base import (
    ActuatorGroup,
    CameraSpec,
    ObservationSchema,
    Robot,
)
from transport.dataset import DatasetTransport


def _franka_like_robot() -> Robot:
    joints = tuple(f"j{i}" for i in range(8))
    return Robot(
        name="fake_franka",
        actuator_groups=(
            ActuatorGroup("left_arm", 8, joints, gripper_index=7),
            ActuatorGroup("right_arm", 8, joints, gripper_index=7),
        ),
        initial_qpos=np.asarray(
            [10, 11, 12, 13, 14, 15, 16, 0, 20, 21, 22, 23, 24, 25, 26, 0],
            dtype=np.float32,
        ),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("left_arm", "right_arm"),
        ),
    )


def _dataset_transport_for_disabled_left() -> DatasetTransport:
    robot = _franka_like_robot()
    transport = object.__new__(DatasetTransport)
    transport._robot = robot
    transport._disabled_groups = {"left_arm"}
    transport._enabled_groups = [robot.actuator_groups[1]]
    transport._group_initial_qpos = {
        "left_arm": robot.initial_qpos[:8],
        "right_arm": robot.initial_qpos[8:16],
    }
    transport._robot_qpos_dim = 16
    return transport


def test_expand_robot_qpos_maps_single_arm_to_enabled_right_arm():
    transport = _dataset_transport_for_disabled_left()
    qpos = np.asarray([1, 2, 3, 4, 5, 6, 7, 0.5], dtype=np.float32)

    expanded = transport._expand_robot_qpos(qpos)

    np.testing.assert_allclose(expanded[:8], [10, 11, 12, 13, 14, 15, 16, 0])
    np.testing.assert_allclose(expanded[8:], qpos)


def test_expand_robot_qpos_converts_franka_two_finger_state_to_gripper():
    transport = _dataset_transport_for_disabled_left()
    state = np.asarray([1, 2, 3, 4, 5, 6, 7, 0.04, 0.04], dtype=np.float32)

    expanded = transport._expand_robot_qpos(state)

    np.testing.assert_allclose(expanded[:8], [10, 11, 12, 13, 14, 15, 16, 0])
    np.testing.assert_allclose(expanded[8:15], [1, 2, 3, 4, 5, 6, 7])
    assert expanded[15] == 1.0


# --- real-parquet DatasetTransport round-trips ---


def _write_lerobot_episode(
    dataset_dir,
    states: np.ndarray,
    actions: np.ndarray,
    task: str = "do a thing",
    timestamps: list[float] | None = None,
    fps: float | None = None,
    extra_columns: dict | None = None,
) -> None:
    """Build a minimal LeRobot v2.1 dataset (no video) with one episode at index 0."""
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    data_path_tpl = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_path_tpl = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info = {
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": data_path_tpl,
        "video_path": video_path_tpl,
    }
    if fps is not None:
        info["fps"] = fps
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")

    n = states.shape[0]
    parquet_path = dataset_dir / data_path_tpl.format(episode_chunk=0, episode_index=0)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observations.state.qpos": [row.tolist() for row in states],
        "action": [row.tolist() for row in actions],
        "frame_index": list(range(n)),
        "episode_index": [0] * n,
        "index": list(range(n)),
        "task_index": [0] * n,
    }
    if timestamps is not None:
        columns["timestamp"] = timestamps
    if extra_columns is not None:
        columns.update(extra_columns)
    table = pa.table(columns)
    pq.write_table(table, str(parquet_path))


def _dataset_config() -> ConfigDict:
    # image_* feed get_frame's black-frame fallback; defaults already work here.
    return ConfigDict(
        transport=ConfigDict(
            type="dataset",
            image_height=32,
            image_width=32,
            disabled_cameras=[],
            disabled_groups=[],
            dataset_keys=ConfigDict(
                state_key="observations.state.qpos",
                action_key="action",
                video_keys={},
            ),
        )
    )


def test_dataset_transport_deterministic_replay_from_parquet(tmp_path):
    robot = _franka_like_robot()
    n, dim = 6, 16
    states = (np.arange(n * dim, dtype=np.float32) * 0.1).reshape(n, dim)
    actions = (np.arange(n * dim, dtype=np.float32) * 0.1 + 100.0).reshape(n, dim)
    _write_lerobot_episode(tmp_path, states, actions, task="put the cup on the plate")

    transport = DatasetTransport(_dataset_config(), robot, tmp_path)

    assert transport.n_steps == n
    assert transport.fps == 30
    assert transport.current_task == "put the cup on the plate"
    for i in range(n):
        np.testing.assert_array_equal(transport.get_obs_state(i), states[i])
    np.testing.assert_array_equal(transport.get_action_trajectory(), actions)

    assert transport.frame_index == 0
    for i in range(1, n):
        assert transport.advance() is True
        assert transport.frame_index == i
    assert transport.advance() is False
    assert transport.frame_index == n - 1

    transport.seek(-5)
    assert transport.frame_index == 0
    transport.seek(10_000)
    assert transport.frame_index == n - 1

    other = DatasetTransport(_dataset_config(), robot, tmp_path)
    np.testing.assert_array_equal(other.get_action_trajectory(), transport.get_action_trajectory())


def test_dataset_transport_uses_configured_fallback_fps(tmp_path):
    robot = _franka_like_robot()
    states = np.zeros((2, 16), dtype=np.float32)
    actions = np.ones((2, 16), dtype=np.float32)
    _write_lerobot_episode(tmp_path, states, actions)
    config = _dataset_config()
    config.transport.fps = 24

    transport = DatasetTransport(config, robot, tmp_path)

    assert transport.fps == 24


def test_dataset_transport_uses_recorded_timestamps_for_series_and_fps(tmp_path):
    robot = _franka_like_robot()
    n, dim = 4, 16
    states = (np.arange(n * dim, dtype=np.float32) * 0.1).reshape(n, dim)
    actions = (np.arange(n * dim, dtype=np.float32) * 0.2).reshape(n, dim)
    timestamps = [0.0, 1.0 / 15.0, 2.0 / 15.0, 3.0 / 15.0]
    _write_lerobot_episode(tmp_path, states, actions, timestamps=timestamps, fps=30.0)

    transport = DatasetTransport(_dataset_config(), robot, tmp_path)
    series = transport.series()

    assert transport.fps == 15
    np.testing.assert_allclose(series["timestamp"], timestamps)
    assert series["control_source"] == ["policy"] * n
    assert series["intervention"] == [False] * n
    assert series["intervention_segment_index"] == [-1] * n


def test_dataset_transport_exposes_intervention_series(tmp_path):
    robot = _franka_like_robot()
    n, dim = 4, 16
    states = np.zeros((n, dim), dtype=np.float32)
    actions = np.ones((n, dim), dtype=np.float32)
    _write_lerobot_episode(
        tmp_path,
        states,
        actions,
        extra_columns={
            "control_source": ["policy", "intervention", "intervention", "policy"],
            "intervention": [False, True, True, False],
            "intervention_segment_index": [-1, 0, 0, -1],
        },
    )

    series = DatasetTransport(_dataset_config(), robot, tmp_path).series()

    assert series["control_source"] == ["policy", "intervention", "intervention", "policy"]
    assert series["intervention"] == [False, True, True, False]
    assert series["intervention_segment_index"] == [-1, 0, 0, -1]


def test_dataset_transport_defers_video_decoder_until_frame_read(tmp_path, monkeypatch):
    robot = _franka_like_robot()
    n, dim = 3, 16
    states = (np.arange(n * dim, dtype=np.float32) * 0.1).reshape(n, dim)
    actions = (np.arange(n * dim, dtype=np.float32) * 0.2).reshape(n, dim)
    _write_lerobot_episode(tmp_path, states, actions)
    video_path = (
        tmp_path / "videos" / "chunk-000" / "observation.images.cam_high" / "episode_000000.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"not decoded during mount")
    config = _dataset_config()
    config.transport.dataset_keys.video_keys = {
        "cam_high": "observation.images.cam_high",
    }
    opened = []

    class FakeFrameSource:
        def __init__(self, path):
            opened.append(path)

        def read_at(self, idx):
            return np.full((4, 5, 3), idx, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("transport.dataset._FrameSource", FakeFrameSource)

    transport = DatasetTransport(config, robot, tmp_path)

    assert opened == []
    assert transport.available_camera_keys() == ("cam_high",)
    frame = transport.get_camera_frame("cam_high")
    assert len(opened) == 1
    assert opened[0] == video_path
    assert frame is not None
    assert frame.shape == (4, 5, 3)


def test_dataset_transport_reads_lerobot_v3_shared_shards_and_video_offset(tmp_path, monkeypatch):
    robot = _franka_like_robot()
    meta_dir = tmp_path / "meta"
    episode_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_dir = tmp_path / "data" / "chunk-000"
    video_dir = tmp_path / "videos" / "observation.images.cam_high" / "chunk-000"
    episode_meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 2,
                "fps": 10,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                ),
                "features": {
                    "observations.state.qpos": {"dtype": "float32", "shape": [16]},
                    "action": {"dtype": "float32", "shape": [16]},
                    "observation.images.cam_high": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "tasks": [["zero"], ["one"]],
                "length": [2, 2],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 4],
                "videos/observation.images.cam_high/chunk_index": [0, 0],
                "videos/observation.images.cam_high/file_index": [0, 0],
                "videos/observation.images.cam_high/from_timestamp": [0.0, 2.5],
                "videos/observation.images.cam_high/to_timestamp": [0.2, 2.7],
            }
        ),
        episode_meta_dir / "file-000.parquet",
    )
    states = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    actions = states + 100
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "observations.state.qpos": states.tolist(),
                "action": actions.tolist(),
                "timestamp": [0.0, 0.1, 0.0, 0.1],
            }
        ),
        data_dir / "file-000.parquet",
    )
    video_path = video_dir / "file-000.mp4"
    video_path.write_bytes(b"shared video")
    reads = []

    class FakeFrameSource:
        def __init__(self, path):
            assert path == video_path

        def read_at(self, index):
            reads.append(index)
            return np.full((2, 3, 3), index, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("transport.dataset._FrameSource", FakeFrameSource)
    config = _dataset_config()
    config.transport.dataset_keys.video_keys = {"cam_high": "observation.images.cam_high"}

    transport = DatasetTransport(config, robot, tmp_path, episode_id=1)

    assert transport.data_format == "lerobot_v3"
    assert transport.current_task == "one"
    np.testing.assert_array_equal(transport.get_action_trajectory(), actions[2:])
    assert transport.video_offsets() == {"cam_high": 2.5}
    assert transport.get_camera_frame_at("cam_high", 1) is not None
    assert reads == [26]


def test_dataset_transport_reads_hdf5_embedded_images(tmp_path):
    robot = _franka_like_robot()
    states = np.arange(2 * 16, dtype=np.float32).reshape(2, 16)
    actions = states + 10
    images = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    open_dataset(tmp_path, "hdf5", fps=20).write(
        {"observations.state.qpos": states, "action": actions},
        {"episode_index": 0, "length": 2, "tasks": ["hdf replay"]},
        {"observation.images.cam_high": images},
    )
    config = _dataset_config()
    config.transport.dataset_keys.video_keys = {"cam_high": "observation.images.cam_high"}

    transport = DatasetTransport(config, robot, tmp_path)

    assert transport.data_format == "hdf5"
    assert transport.current_task == "hdf replay"
    assert transport.fps == 20
    assert transport.video_mode == "frames"
    np.testing.assert_array_equal(transport.get_action_trajectory(), actions)
    np.testing.assert_array_equal(transport.get_camera_frame_at("cam_high", 1), images[1])


def test_dataset_transport_close_waits_for_active_video_read(tmp_path, monkeypatch):
    robot = _franka_like_robot()
    states = np.zeros((3, 16), dtype=np.float32)
    actions = np.zeros((3, 16), dtype=np.float32)
    _write_lerobot_episode(tmp_path, states, actions)
    video_path = (
        tmp_path / "videos" / "chunk-000" / "observation.images.cam_high" / "episode_000000.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake video")
    config = _dataset_config()
    config.transport.dataset_keys.video_keys = {
        "cam_high": "observation.images.cam_high",
    }
    read_started = threading.Event()
    allow_read = threading.Event()
    released = threading.Event()

    class BlockingFrameSource:
        def __init__(self, _path):
            pass

        def read_at(self, _idx):
            read_started.set()
            assert allow_read.wait(timeout=2.0)
            return np.zeros((4, 5, 3), dtype=np.uint8)

        def release(self):
            released.set()

    monkeypatch.setattr("transport.dataset._FrameSource", BlockingFrameSource)
    transport = DatasetTransport(config, robot, tmp_path)
    reader = threading.Thread(target=transport.get_camera_frame, args=("cam_high",))
    reader.start()
    assert read_started.wait(timeout=1.0)

    closer = threading.Thread(target=transport.close)
    closer.start()

    assert not released.wait(timeout=0.1)
    allow_read.set()
    reader.join(timeout=1.0)
    closer.join(timeout=1.0)
    assert not reader.is_alive()
    assert not closer.is_alive()
    assert released.is_set()


def test_dataset_transport_publish_records_and_clamps(tmp_path):
    robot = _franka_like_robot()
    n, dim = 4, 16
    states = (np.arange(n * dim, dtype=np.float32) * 0.01).reshape(n, dim)
    actions = (np.arange(n * dim, dtype=np.float32) * 0.01 + 5.0).reshape(n, dim)
    _write_lerobot_episode(tmp_path, states, actions)

    transport = DatasetTransport(_dataset_config(), robot, tmp_path)

    n_published = n + 3
    last_action = None
    for k in range(n_published):
        last_action = np.full(dim, float(k), dtype=np.float32)
        transport.publish_action(last_action)

    assert len(transport.recorded_qpos) == n_published
    assert transport.frame_index == n - 1  # clamped, no IndexError

    latest = transport.get_latest_qpos()
    assert latest is not None and last_action is not None
    np.testing.assert_array_equal(latest, transport._expand_robot_qpos(last_action))

    # recorded_qpos returns a fresh list: appending to it must not grow the buffer.
    snapshot = transport.recorded_qpos
    snapshot.append(np.zeros(dim, dtype=np.float32))
    assert len(transport.recorded_qpos) == n_published


def test_get_action_trajectory_uses_configured_action_key_when_action_eef_exists(tmp_path):
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir.mkdir()
    data_dir.mkdir(parents=True)

    info = {
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observations.state.qpos": {"dtype": "float32", "shape": [16]},
            "action": {"dtype": "float32", "shape": [16]},
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
        json.dumps({"task_index": 0, "task": "test task"}) + "\n",
        encoding="utf-8",
    )

    expected_actions = np.asarray([[1.0] * 16, [2.0] * 16], dtype=np.float32)
    action_eef = np.asarray([[10.0] * 16, [20.0] * 16], dtype=np.float32)
    table = pa.table(
        {
            "observations.state.qpos": pa.array(
                [[0.0] * 16, [0.5] * 16],
                type=pa.list_(pa.float32()),
            ),
            "action": pa.array(expected_actions.tolist(), type=pa.list_(pa.float32())),
            "action_eef": pa.array(action_eef.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array([0.0, 0.1], type=pa.float32()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")

    robot = _franka_like_robot()
    config = ConfigDict(
        transport=ConfigDict(
            type="dataset",
            dataset_dir=str(tmp_path),
            image_height=32,
            image_width=32,
            disabled_cameras=[],
            disabled_groups=[],
            dataset_keys=ConfigDict(
                state_key="observations.state.qpos",
                action_key="action",
                video_keys={},
            ),
        )
    )
    transport = DatasetTransport(config, robot, tmp_path, episode_id=0)

    trajectory = transport.get_action_trajectory()

    np.testing.assert_allclose(trajectory, expected_actions)
    assert not np.array_equal(trajectory, action_eef)
