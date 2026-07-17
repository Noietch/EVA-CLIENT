from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from core.config import ConfigDict
from core.recorder import episode as episode_module
from core.recorder.episode import EpisodeLogger
from core.types import Observation, RolloutInterventionSegment
from robots.base import (
    ActuatorGroup,
    CameraSpec,
    ObservationSchema,
    Robot,
)

_DIM = 3


def _robot() -> Robot:
    joints = tuple(f"j{i}" for i in range(_DIM))
    return Robot(
        name="fake_arm",
        actuator_groups=(ActuatorGroup("arm", _DIM, joints),),
        initial_qpos=np.zeros(_DIM, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )


def _logger(
    log_dir,
    *,
    save_image_height: int | None = None,
    save_image_width: int | None = None,
) -> EpisodeLogger:
    return EpisodeLogger(
        log_dir,
        _robot(),
        fps=10,
        dataset_keys=ConfigDict(
            state_key="observations.state.qpos",
            eef_key="observations.state.eef",
            action_key="action",
            video_keys={},
        ),
        async_save=False,
        save_image_height=save_image_height,
        save_image_width=save_image_width,
    )


def _image(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _obs(state: np.ndarray, image_value: int) -> Observation:
    return Observation(images={"cam_high": _image(image_value)}, state_qpos=state)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_rollout_save_merges_intervention_into_episode_table(tmp_path):
    logger = _logger(tmp_path)
    keys = ConfigDict(state_key="observations.state.qpos", action_key="action")
    segment = RolloutInterventionSegment(
        segment_index=0,
        start_policy_frame_index=1,
        pre_intervention_qpos=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        resume_policy_frame_index=2,
        frames=[
            Observation(
                timestamp=0.2,
                images={"cam_high": _image(20)},
                state_qpos=np.array([1.0, 1.1, 1.2], dtype=np.float32),
                action_qpos=np.array([2.0, 2.1, 2.2], dtype=np.float32),
            ),
            Observation(
                timestamp=0.3,
                images={"cam_high": _image(30)},
                state_qpos=np.array([3.0, 3.1, 3.2], dtype=np.float32),
                action_qpos=np.array([4.0, 4.1, 4.2], dtype=np.float32),
            ),
        ],
    )

    logger.start_episode("pick")
    logger.record_step(
        _obs(np.array([0.0, 0.0, 0.0], dtype=np.float32), 1),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
        timestamp=0.0,
    )
    logger.record_step(
        _obs(np.array([0.1, 0.1, 0.1], dtype=np.float32), 2),
        np.array([0.6, 0.6, 0.6], dtype=np.float32),
        timestamp=0.1,
    )
    logger.record_step(
        _obs(np.array([0.7, 0.7, 0.7], dtype=np.float32), 4),
        np.array([0.8, 0.8, 0.8], dtype=np.float32),
        timestamp=0.4,
    )
    logger.set_rollout_intervention_segments([segment])

    assert logger.end_episode() is True

    rollout_table = pq.read_table(str(tmp_path / "data" / "chunk-000" / "episode_000000.parquet"))
    assert rollout_table.num_rows == 5
    np.testing.assert_allclose(
        np.array(rollout_table.column(keys.state_key).to_pylist(), dtype=np.float32),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [1.0, 1.1, 1.2],
                [3.0, 3.1, 3.2],
                [0.7, 0.7, 0.7],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        np.array(rollout_table.column(keys.action_key).to_pylist(), dtype=np.float32),
        np.array(
            [
                [0.5, 0.5, 0.5],
                [0.6, 0.6, 0.6],
                [2.0, 2.1, 2.2],
                [4.0, 4.1, 4.2],
                [0.8, 0.8, 0.8],
            ],
            dtype=np.float32,
        ),
    )
    assert rollout_table.column("intervention").to_pylist() == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert rollout_table.column("intervention_segment_index").to_pylist() == [
        -1,
        -1,
        0,
        0,
        -1,
    ]
    assert rollout_table.column("control_source").to_pylist() == [
        "policy",
        "policy",
        "intervention",
        "intervention",
        "policy",
    ]
    assert not (tmp_path / "interventions").exists()

    episodes = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")
    assert episodes[0]["has_intervention"] is True
    assert episodes[0]["intervention_segments"] == 1
    assert episodes[0]["intervention_frames"] == 2
    assert episodes[0]["intervention_ranges"] == [
        {"segment_index": 0, "start_frame": 2, "end_frame": 3}
    ]


def test_rollout_intervention_video_uses_episode_writer_when_size_set(tmp_path, monkeypatch):
    frame_shapes: dict[str, list[tuple[int, int, int]]] = {}

    class _Writer:
        def __init__(self, path: str) -> None:
            self.path = path
            frame_shapes[self.path] = []

        def append_data(self, frame: np.ndarray) -> None:
            frame_shapes[self.path].append(frame.shape)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        episode_module.imageio, "get_writer", lambda path, *a, **k: _Writer(str(path))
    )
    logger = _logger(tmp_path, save_image_height=120, save_image_width=160)
    state = np.zeros(_DIM, dtype=np.float32)
    segment = RolloutInterventionSegment(
        segment_index=0,
        start_policy_frame_index=1,
        pre_intervention_qpos=state,
        frames=[
            Observation(
                timestamp=10.0,
                images={"cam_high": np.zeros((480, 640, 3), dtype=np.uint8)},
                state_qpos=state,
                action_qpos=state,
            ),
            Observation(
                timestamp=10.1,
                images={"cam_high": np.zeros((480, 640, 3), dtype=np.uint8)},
                state_qpos=state,
                action_qpos=state,
            ),
        ],
    )

    logger.start_episode("pick")
    logger.record_step(_obs(state, 1), state, timestamp=9.9)
    logger.set_rollout_intervention_segments([segment])

    assert logger.end_episode() is True
    episode_shapes = [
        shapes for path, shapes in frame_shapes.items() if "/videos/chunk-000/" in path
    ]
    assert episode_shapes == [[(120, 160, 3), (120, 160, 3), (120, 160, 3)]]
    intervention_dir = "/" + "inter" + "ventions/"
    assert not any(intervention_dir in path for path in frame_shapes)


def test_rollout_save_removes_explicit_policy_warmup_interval(tmp_path, monkeypatch):
    written_frames: list[int] = []

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            written_frames.append(int(frame[0, 0, 0]))

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *args, **kwargs: _Writer())
    logger = _logger(tmp_path)
    state = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("pick")
    for frame_index in range(6):
        logger.record_step(
            _obs(state + frame_index, frame_index),
            state + frame_index,
            timestamp=frame_index / 10.0,
        )
    logger.set_episode_meta(
        excluded_ranges=[
            {
                "reason": "policy_warmup",
                "start_time": 0.1,
                "end_time": 0.4,
            }
        ]
    )

    assert logger.end_episode() is True

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == 4
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [0.0, 0.1, 0.4, 0.5])
    np.testing.assert_allclose(table.column("timestamp").to_pylist(), [0.0, 0.1, 0.2, 0.3])
    assert written_frames == [0, 1, 4, 5]
    episode = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")[0]
    assert episode["excluded_frames"] == 2
    assert episode["excluded_ranges"] == [
        {
            "reason": "policy_warmup",
            "start_time": 0.1,
            "end_time": 0.4,
            "duration_sec": 0.3,
            "removed_frames": 2,
        }
    ]
