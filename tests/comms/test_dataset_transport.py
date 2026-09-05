from __future__ import annotations

import json
import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.config import ConfigDict
from robots.base import (
    ActuatorGroup,
    CameraSpec,
    ObservationSchema,
    Robot,
)
from transport.dataset import DatasetTransport

pytestmark = pytest.mark.integration


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
