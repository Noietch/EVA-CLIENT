"""Tests for EpisodeLogger (core.recorder.episode) — synchronous parquet/meta
writing, multi-episode lifecycle continuation, and input-copy / empty-episode
data integrity.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.config import ConfigDict
from core.recorder import collection as collection_module
from core.recorder import episode as episode_module
from core.recorder.episode import EpisodeLogger, sanitize_path_component
from core.types import (
    CollectionRawBatch,
    CollectionRawImage,
    CollectionRawSample,
    Observation,
    RawCollectionSnapshot,
)
from robots.base import (
    ActuatorGroup,
    CameraSpec,
    ObservationSchema,
    Robot,
)

_DIM = 6


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
    save_video: bool = False,
    fps: int = 30,
    async_save: bool = False,
    recording_space: str = "qpos",
) -> EpisodeLogger:
    return EpisodeLogger(
        log_dir,
        _robot(),
        fps=fps,
        dataset_keys=ConfigDict(
            state_key="observations.state.qpos",
            eef_key="observations.state.eef",
            action_key="action",
            video_keys={},
        ),
        async_save=async_save,
        recording_space=recording_space,
    )


def _rollout_logger(
    log_dir,
    *,
    save_image_height: int | None = None,
    save_image_width: int | None = None,
) -> EpisodeLogger:
    return EpisodeLogger(
        log_dir,
        _robot(),
        fps=30,
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


def _collection_logger(
    log_dir,
    *,
    save_video: bool = False,
    save_image_height: int | None = None,
    save_image_width: int | None = None,
    async_save: bool = False,
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
        collection=ConfigDict(
            enabled=True,
            schema=ConfigDict(
                robot_type="fake_arm",
                min_episode_frames=1,
                arms={"left_arm": "L"},
                cameras={
                    "cam_high": "observation.images.cam_high",
                },
                columns={
                    "qpos": "observation.qpos",
                    "action_qpos": "action",
                },
            ),
        ),
        async_save=async_save,
        save_image_height=save_image_height,
        save_image_width=save_image_width,
    )


def _collection_task_dir(root, task: str = "t"):
    return root / sanitize_path_component(task)


def _obs(state: np.ndarray, image: np.ndarray | None = None) -> Observation:
    images = {}
    if image is not None:
        images["cam_high"] = image
    return Observation(images=images, state_qpos=state)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _collection_raw_snapshot(
    timestamp: float,
    *,
    image: np.ndarray | None = None,
    state: np.ndarray | None = None,
    action: np.ndarray | None = None,
    batch: CollectionRawBatch | None = None,
) -> RawCollectionSnapshot:
    if batch is None:
        image_value = np.zeros((8, 8, 3), dtype=np.uint8) if image is None else np.asarray(image)
        state_value = (
            np.zeros(_DIM, dtype=np.float32)
            if state is None
            else np.asarray(state, dtype=np.float32)
        )
        action_value = state_value if action is None else np.asarray(action, dtype=np.float32)
        batch = CollectionRawBatch(
            images={"cam_high": [CollectionRawSample(timestamp, image_value)]},
            vectors={
                "state_qpos": [CollectionRawSample(timestamp, state_value)],
                "action_qpos": [CollectionRawSample(timestamp, action_value)],
            },
        )
    return RawCollectionSnapshot(timestamp=timestamp, decode_raw=lambda: batch)


def _collection_raw_batch(
    timestamps: tuple[float, ...],
    *,
    image: np.ndarray | None = None,
    state: np.ndarray | None = None,
    action: np.ndarray | None = None,
) -> CollectionRawBatch:
    image_value = np.zeros((8, 8, 3), dtype=np.uint8) if image is None else image
    state_value = (
        np.zeros(_DIM, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32)
    )
    action_value = state_value if action is None else np.asarray(action, dtype=np.float32)
    return CollectionRawBatch(
        images={"cam_high": [CollectionRawSample(ts, image_value) for ts in timestamps]},
        vectors={
            "state_qpos": [CollectionRawSample(ts, state_value) for ts in timestamps],
            "action_qpos": [CollectionRawSample(ts, action_value) for ts in timestamps],
        },
    )


def test_raw_collection_snapshot_rejects_decoded_frame_path():
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        RawCollectionSnapshot(
            timestamp=0.0,
            decode=lambda: Observation(
                timestamp=0.0,
                images={"cam_high": image},
                state_qpos=qpos,
                action_qpos=qpos,
            ),
        )


def test_collection_logger_has_no_decoded_frame_entrypoint(tmp_path):
    logger = _collection_logger(tmp_path)
    assert not hasattr(logger, "record_collection_frame")


def test_live_series_streams_in_flight_buffer(tmp_path):
    # live_series exposes the in-memory step buffer (state/action/timestamp) so the
    # console can chart a rollout before it is ever saved. Shape mirrors
    # load_episode_series; active flag tracks the open episode.
    logger = _logger(tmp_path)
    empty = logger.live_series()
    assert empty == {"active": False, "n": 0, "timestamp": [], "state": [], "action": []}

    n = 4
    states = (np.arange(n * _DIM, dtype=np.float32)).reshape(n, _DIM)
    actions = states + 100.0
    logger.start_episode("t")
    for i in range(n):
        logger.record_step(_obs(states[i]), actions[i], timestamp=float(i))

    full = logger.live_series()
    assert full["active"] is True
    assert full["n"] == n
    assert full["timestamp"] == [0.0, 1.0, 2.0, 3.0]
    assert full["state"][0] == states[0].tolist()
    assert full["action"][-1] == actions[-1].tolist()

    # since= returns only the tail for incremental polling, but n stays the total.
    tail = logger.live_series(since=2)
    assert tail["n"] == n
    assert len(tail["timestamp"]) == 2
    assert tail["timestamp"] == [2.0, 3.0]
    assert tail["state"][0] == states[2].tolist()

    # after the episode ends the buffer is no longer active.
    logger.end_episode()
    assert logger.live_series()["active"] is False


def test_raw_snapshot_updates_live_series_without_decoding(tmp_path):
    logger = _logger(tmp_path)
    state = np.arange(_DIM, dtype=np.float32)
    action = state + 100.0

    def reject_decode() -> CollectionRawBatch:
        raise AssertionError("live series must not decode the raw snapshot")

    logger.start_episode("t")
    logger.ingest_raw_episode_snapshot(
        RawCollectionSnapshot(timestamp=1.25, decode_raw=reject_decode),
        action,
        state,
    )

    assert logger.live_series() == {
        "active": True,
        "n": 1,
        "timestamp": [1.25],
        "state": [state.tolist()],
        "action": [action.tolist()],
    }


def test_rollout_resizes_saved_frames_when_size_set(tmp_path, monkeypatch):
    frame_shapes = []

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            frame_shapes.append(frame.shape)

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *a, **k: _Writer())
    logger = _rollout_logger(tmp_path, save_image_height=120, save_image_width=160)
    state = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("t")
    for _ in range(3):
        logger.record_step(_obs(state, np.zeros((480, 640, 3), dtype=np.uint8)), state)

    assert logger.end_episode() is True
    assert frame_shapes == [(120, 160, 3)] * 3
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.cam_high"]["shape"] == [120, 160, 3]


def test_episode_logger_writes_faststart_mp4s(tmp_path, monkeypatch):
    writer_kwargs = []

    class _Writer:
        def append_data(self, _frame: np.ndarray) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_get_writer(*_args, **kwargs):
        writer_kwargs.append(kwargs)
        return _Writer()

    monkeypatch.setattr(episode_module.imageio, "get_writer", fake_get_writer)
    logger = _rollout_logger(tmp_path)
    state = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("t")
    logger.record_step(_obs(state, np.zeros((8, 8, 3), dtype=np.uint8)), state)

    assert logger.end_episode() is True
    assert writer_kwargs
    assert writer_kwargs[0]["ffmpeg_params"] == [
        "-preset",
        "ultrafast",
        "-movflags",
        "+faststart",
    ]


def test_episode_logger_sync_save_writes_n_rows(tmp_path):
    keys = ConfigDict(
        state_key="observations.state.qpos",
        eef_key="observations.state.eef",
        action_key="action.qpos",
        video_keys={},
    )
    logger = _logger(tmp_path)
    n = 5
    states = (np.arange(n * _DIM, dtype=np.float32) * 0.5).reshape(n, _DIM)
    actions = (np.arange(n * _DIM, dtype=np.float32) * 0.5 + 7.0).reshape(n, _DIM)

    logger.start_episode("t")
    for i in range(n):
        logger.record_step(_obs(states[i]), actions[i])
    logger.end_episode()

    parquet_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(str(parquet_path))
    assert table.num_rows == n
    assert table.column("frame_index").to_pylist() == list(range(n))
    # Sync end_episode advances _global_index before writing, so the first episode's
    # global index block starts at n (not 0); assert contiguous & strictly monotonic.
    index_col = table.column("index").to_pylist()
    assert index_col == list(range(index_col[0], index_col[0] + n))
    assert all(b - a == 1 for a, b in zip(index_col, index_col[1:], strict=False))
    np.testing.assert_allclose(
        np.array(table.column(keys.state_key).to_pylist(), dtype=np.float32), states
    )
    np.testing.assert_allclose(
        np.array(table.column(keys.action_key).to_pylist(), dtype=np.float32), actions
    )

    episodes = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")
    assert len(episodes) == 1
    assert episodes[0]["length"] == n
    assert episodes[0]["tasks"] == ["t"]


def test_episode_logger_start_stop_start_lifecycle(tmp_path):
    logger = _logger(tmp_path)
    k1, k2 = 3, 4

    logger.start_episode("a")
    for i in range(k1):
        logger.record_step(_obs(np.full(_DIM, i, dtype=np.float32)), np.zeros(_DIM, np.float32))
    logger.end_episode()

    logger.start_episode("b")
    for i in range(k2):
        logger.record_step(_obs(np.full(_DIM, i, dtype=np.float32)), np.zeros(_DIM, np.float32))
    logger.end_episode()

    data_dir = tmp_path / "data" / "chunk-000"
    assert (data_dir / "episode_000000.parquet").exists()
    assert (data_dir / "episode_000001.parquet").exists()

    episodes = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")
    assert len(episodes) == 2
    assert [e["episode_index"] for e in episodes] == [0, 1]
    assert [e["length"] for e in episodes] == [k1, k2]

    ep1 = pq.read_table(str(data_dir / "episode_000001.parquet"))
    ep0 = pq.read_table(str(data_dir / "episode_000000.parquet"))
    # global 'index' strictly increases across episodes (ep1 starts after ep0's block).
    assert ep1.column("index").to_pylist()[0] > ep0.column("index").to_pylist()[-1]

    # A fresh logger over the same dir resumes without clobbering existing episodes.
    resumed = _logger(tmp_path)
    assert resumed.current_episode_index == 2
    resumed.start_episode("c")
    resumed.record_step(_obs(np.ones(_DIM, np.float32)), np.zeros(_DIM, np.float32))
    resumed.end_episode()
    assert (data_dir / "episode_000002.parquet").exists()
    assert (data_dir / "episode_000000.parquet").exists()


def test_episode_logger_copies_inputs_and_skips_empty_episode(tmp_path):
    keys = ConfigDict(
        state_key="observations.state.qpos",
        eef_key="observations.state.eef",
        action_key="action.qpos",
        video_keys={},
    )

    # (a) record_step snapshots inputs: mutating after the call must not alter the row.
    logger = _logger(tmp_path)
    logger.start_episode("t")
    state = np.ones(_DIM, dtype=np.float32)
    action = np.full(_DIM, 2.0, dtype=np.float32)
    logger.record_step(_obs(state), action)
    state[:] = -1.0
    action[:] = -1.0
    logger.end_episode()

    table = pq.read_table(str(tmp_path / "data" / "chunk-000" / "episode_000000.parquet"))
    np.testing.assert_allclose(table.column(keys.state_key).to_pylist()[0], np.ones(_DIM))
    np.testing.assert_allclose(table.column(keys.action_key).to_pylist()[0], np.full(_DIM, 2.0))

    # (b) an empty episode writes nothing and leaves the next index unchanged.
    empty_dir = tmp_path / "empty"
    logger2 = _logger(empty_dir)
    assert logger2.current_episode_index == 0
    logger2.start_episode("t")
    logger2.end_episode()
    assert not (empty_dir / "data").exists()
    assert not (empty_dir / "meta" / "episodes.jsonl").exists()
    assert logger2.current_episode_index == 0

    # a subsequent one-step episode still lands at episode_000000.
    logger2.start_episode("t")
    logger2.record_step(_obs(np.zeros(_DIM, np.float32)), np.zeros(_DIM, np.float32))
    logger2.end_episode()
    assert (empty_dir / "data" / "chunk-000" / "episode_000000.parquet").exists()


def test_collection_timestamp_is_ideal_grid_capture_time_keeps_raw(tmp_path):
    # The fatal bug: parquet timestamp stored raw hardware uptime with jitter. LeRobot
    # requires timestamp[i] == i/fps (tol 1e-4); the raw capture time must move aside
    # into capture_time so it no longer pollutes the validated column.
    logger = _collection_logger(tmp_path)  # fps=10
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    raw = [73529.6, 73529.7, 73529.8, 73529.9]
    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        _collection_raw_snapshot(
            raw[-1],
            batch=_collection_raw_batch(tuple(raw), image=image, state=qpos, action=qpos),
        )
    )
    assert logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    timestamps = table.column("timestamp").to_pylist()
    expected = [i / 10.0 for i in range(len(raw))]
    np.testing.assert_allclose(timestamps, expected, atol=1e-4)
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), raw)


def test_collection_raw_batches_align_to_fixed_grid_before_save(tmp_path):
    logger = _collection_logger(tmp_path)  # fps=10
    image0 = np.full((8, 8, 3), 10, dtype=np.uint8)
    image1 = np.full((8, 8, 3), 20, dtype=np.uint8)
    image2 = np.full((8, 8, 3), 30, dtype=np.uint8)

    batch = CollectionRawBatch(
        images={
            "cam_high": [
                CollectionRawSample(1.00, image0),
                CollectionRawSample(1.10, image1),
                CollectionRawSample(1.20, image2),
            ]
        },
        vectors={
            "state_qpos": [
                CollectionRawSample(1.00, np.zeros(_DIM, dtype=np.float32)),
                CollectionRawSample(1.20, np.full(_DIM, 2.0, dtype=np.float32)),
            ],
            "action_qpos": [
                CollectionRawSample(1.00, np.full(_DIM, 10.0, dtype=np.float32)),
                CollectionRawSample(1.20, np.full(_DIM, 12.0, dtype=np.float32)),
            ],
        },
    )

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(
            timestamp=1.2,
            decode_raw=lambda: batch,
        )
    )
    assert logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    np.testing.assert_allclose(table.column("timestamp").to_pylist(), [0.0, 0.1, 0.2])
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [1.0, 1.1, 1.2])
    np.testing.assert_allclose(table.column("observation.qpos").to_pylist()[1], [1.0] * _DIM)
    np.testing.assert_allclose(table.column("action").to_pylist()[1], [11.0] * _DIM)

    episode = _read_jsonl(task_dir / "meta" / "episodes.jsonl")[0]
    assert episode["quality"] == "green"
    assert episode["alignment_fps"] == 10.0
    assert episode["alignment_image_skew_tolerance_sec"] == 1.0 / 18.0
    assert episode["alignment_image_stream_stats"]["cam_high"]["sample_count"] == 3
    assert episode["alignment_image_stream_stats"]["cam_high"]["max_gap_sec"] == 0.1


def test_record_step_batches_align_to_fixed_grid_before_save(tmp_path):
    logger = _logger(tmp_path, fps=10)
    state0 = np.zeros(_DIM, dtype=np.float32)
    state1 = np.full(_DIM, 2.0, dtype=np.float32)
    action0 = np.full(_DIM, 10.0, dtype=np.float32)
    action1 = np.full(_DIM, 12.0, dtype=np.float32)

    logger.start_episode("t")
    logger.record_step(_obs(state0), action0, timestamp=1.0)
    logger.record_step(_obs(state1), action1, timestamp=1.2)
    assert logger.end_episode()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    np.testing.assert_allclose(table.column("timestamp").to_pylist(), [0.0, 0.1, 0.2])
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [1.0, 1.1, 1.2])
    np.testing.assert_allclose(table.column("observations.state.qpos").to_pylist()[1], [1.0] * _DIM)
    np.testing.assert_allclose(table.column("action.qpos").to_pylist()[1], [11.0] * _DIM)

    episode = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")[0]
    assert episode["quality"] == "green"
    assert episode["alignment_fps"] == 10.0


def test_raw_episode_snapshots_decode_on_async_save_worker(tmp_path):
    logger = _logger(tmp_path, fps=10, async_save=True)
    raw_decodes = 0
    image_decodes = 0

    def snapshot(timestamp: float, state: np.ndarray) -> RawCollectionSnapshot:
        def decode_raw() -> CollectionRawBatch:
            nonlocal raw_decodes
            raw_decodes += 1

            def decode_image() -> np.ndarray:
                nonlocal image_decodes
                image_decodes += 1
                return np.zeros((2, 2, 3), dtype=np.uint8)

            return CollectionRawBatch(
                images={
                    "cam_high": [
                        CollectionRawSample(timestamp, CollectionRawImage(decode_image)),
                    ],
                },
                vectors={
                    "state_qpos": [CollectionRawSample(timestamp, state.copy())],
                },
            )

        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=decode_raw)

    logger.start_episode("t")
    logger.ingest_raw_episode_snapshot(
        snapshot(1.0, np.zeros(_DIM, dtype=np.float32)),
        np.full(_DIM, 10.0, dtype=np.float32),
    )
    logger.ingest_raw_episode_snapshot(
        snapshot(1.2, np.full(_DIM, 2.0, dtype=np.float32)),
        np.full(_DIM, 12.0, dtype=np.float32),
    )
    logger._start_save_worker = lambda: None

    assert logger.end_episode()
    assert raw_decodes == 0
    assert image_decodes == 0

    logger._save_queue_worker()

    assert raw_decodes == 2
    assert image_decodes == 2
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    np.testing.assert_allclose(table.column("timestamp").to_pylist(), [0.0, 0.1, 0.2])
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [1.0, 1.1, 1.2])
    np.testing.assert_allclose(table.column("observations.state.qpos").to_pylist()[1], [1.0] * _DIM)
    np.testing.assert_allclose(table.column("action.qpos").to_pylist()[1], [11.0] * _DIM)
    episode = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")[0]
    assert episode["alignment_image_skew_tolerance_sec"] == 1.0 / 18.0


def test_raw_episode_action_column_follows_recording_space(tmp_path):
    qpos_logger = _logger(tmp_path / "qpos", fps=10, recording_space="qpos")
    qpos = np.zeros(_DIM, dtype=np.float32)
    qpos_logger.start_episode("t")
    qpos_logger.record_step(_obs(qpos), qpos + 1.0, timestamp=0.0)
    qpos_logger.record_step(_obs(qpos), qpos + 2.0, timestamp=0.1)

    assert qpos_logger.end_episode()

    qpos_table = pq.read_table(
        tmp_path / "qpos" / "data" / "chunk-000" / "episode_000000.parquet"
    )
    assert "action.qpos" in qpos_table.column_names
    assert "action" not in qpos_table.column_names

    eef_logger = _logger(tmp_path / "eef", fps=10, recording_space="eef")
    action_eef = np.arange(8, dtype=np.float32)
    eef_logger.start_episode("t")
    for timestamp in (0.0, 0.1):
        eef_logger.record_step(
            Observation(
                images={},
                state_qpos=qpos,
                state_eef=np.zeros(8, dtype=np.float32),
                action_eef=action_eef,
            ),
            qpos,
            timestamp=timestamp,
        )

    assert eef_logger.end_episode()

    eef_table = pq.read_table(
        tmp_path / "eef" / "data" / "chunk-000" / "episode_000000.parquet"
    )
    assert "action.eef" in eef_table.column_names
    np.testing.assert_allclose(eef_table.column("action.eef").to_pylist(), [action_eef] * 2)


def test_raw_episode_save_refreshes_info_and_stats(tmp_path):
    logger = _logger(tmp_path, fps=10)
    qpos = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("t")
    logger.record_step(_obs(qpos), qpos + 1.0, timestamp=0.0)
    logger.record_step(_obs(qpos), qpos + 2.0, timestamp=0.1)

    assert logger.end_episode()

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert info["splits"] == {"train": "0:1"}
    assert "action.qpos" in info["features"]
    assert set(stats) == {"observations.state.qpos", "action.qpos"}


def test_raw_episode_video_writer_holds_only_one_decoded_image(tmp_path, monkeypatch):
    raw_images: list[CollectionRawImage] = []
    written_values: list[int] = []

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            assert sum(image._decoded is not None for image in raw_images) == 1
            written_values.append(int(frame[0, 0, 0]))

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *args, **kwargs: _Writer())
    logger = _logger(tmp_path, fps=10, async_save=True)

    def snapshot(timestamp: float, value: int) -> RawCollectionSnapshot:
        image = CollectionRawImage(
            lambda value=value: np.full((2, 2, 3), value, dtype=np.uint8)
        )
        raw_images.append(image)

        def decode_raw() -> CollectionRawBatch:
            state = np.full(_DIM, timestamp, dtype=np.float32)
            return CollectionRawBatch(
                images={"cam_high": [CollectionRawSample(timestamp, image)]},
                vectors={"state_qpos": [CollectionRawSample(timestamp, state)]},
            )

        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=decode_raw)

    logger.start_episode("t")
    logger.ingest_raw_episode_snapshot(
        snapshot(1.0, 10),
        np.zeros(_DIM, dtype=np.float32),
    )
    logger.ingest_raw_episode_snapshot(
        snapshot(1.1, 20),
        np.ones(_DIM, dtype=np.float32),
    )
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)

    assert logger.end_episode()
    assert all(image._decoded is None for image in raw_images)

    logger._save_queue_worker()

    assert written_values == [10, 20]
    assert all(image._decoded is None for image in raw_images)


def test_pending_score_patch_applies_to_async_save_job(tmp_path):
    logger = _logger(tmp_path, fps=10, async_save=True)

    def snapshot(timestamp: float, state: np.ndarray) -> RawCollectionSnapshot:
        def decode_raw() -> CollectionRawBatch:
            return CollectionRawBatch(
                images={
                    "cam_high": [
                        CollectionRawSample(
                            timestamp,
                            np.zeros((2, 2, 3), dtype=np.uint8),
                        ),
                    ],
                },
                vectors={
                    "state_qpos": [CollectionRawSample(timestamp, state.copy())],
                },
            )

        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=decode_raw)

    logger.start_episode("t")
    logger.set_episode_meta(clip_id="c1", prompt="p", trial=1)
    logger.ingest_raw_episode_snapshot(
        snapshot(1.0, np.zeros(_DIM, dtype=np.float32)),
        np.full(_DIM, 10.0, dtype=np.float32),
    )
    logger.ingest_raw_episode_snapshot(
        snapshot(1.2, np.full(_DIM, 2.0, dtype=np.float32)),
        np.full(_DIM, 12.0, dtype=np.float32),
    )
    logger._start_save_worker = lambda: None

    assert logger.end_episode()
    episode_index = logger.patch_episode_meta_by_clip(
        "c1",
        score=2,
        max_score=4,
        milestones={"grasp": True},
        note="ok",
    )

    assert episode_index == 0

    logger._save_queue_worker()

    row = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")[0]
    assert row["clip_id"] == "c1"
    assert row["score"] == 2
    assert row["max_score"] == 4
    assert row["milestones"] == {"grasp": True}
    assert row["note"] == "ok"


def test_record_step_uses_observation_timestamp_when_present(tmp_path):
    logger = _logger(tmp_path, fps=10)
    state0 = np.zeros(_DIM, dtype=np.float32)
    state1 = np.full(_DIM, 2.0, dtype=np.float32)

    logger.start_episode("t")
    logger.record_step(Observation(images={}, state_qpos=state0, timestamp=5.0), state0)
    logger.record_step(Observation(images={}, state_qpos=state1, timestamp=5.2), state1)
    assert logger.end_episode()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [5.0, 5.1, 5.2])


def test_collection_raw_batches_report_image_skew_qc(tmp_path):
    logger = _collection_logger(tmp_path)  # fps=10, skew tolerance = 1/(2*9)
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                CollectionRawSample(0.0, np.zeros((8, 8, 3), dtype=np.uint8)),
                CollectionRawSample(0.2, np.ones((8, 8, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                CollectionRawSample(0.0, np.zeros(_DIM, dtype=np.float32)),
                CollectionRawSample(0.2, np.ones(_DIM, dtype=np.float32)),
            ],
            "action_qpos": [
                CollectionRawSample(0.0, np.zeros(_DIM, dtype=np.float32)),
                CollectionRawSample(0.2, np.ones(_DIM, dtype=np.float32)),
            ],
        },
    )

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(
            timestamp=0.2,
            decode_raw=lambda: batch,
        )
    )
    assert logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    episode = _read_jsonl(task_dir / "meta" / "episodes.jsonl")[0]
    assert episode["quality"] == "red"
    assert episode["alignment_image_max_skew_sec"]["cam_high"] == 0.1
    assert "image_skew_exceeded" in {issue["code"] for issue in episode["quality_issues"]}


def test_collection_raw_end_episode_defers_alignment_and_video_preprocess(tmp_path, monkeypatch):
    logger = _collection_logger(
        tmp_path,
        save_video=True,
        save_image_height=4,
        save_image_width=6,
        async_save=True,
    )
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)
    monkeypatch.setattr(
        collection_module,
        "align_collection_samples",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("alignment must run in the save worker")
        ),
    )
    monkeypatch.setattr(
        collection_module,
        "resize_direct",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("video preprocessing must run in the save worker")
        ),
    )
    decode_calls = []

    def decode_raw() -> CollectionRawBatch:
        decode_calls.append("decode")
        raise AssertionError("raw decode must run in the save worker")

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(
            timestamp=0.1,
            decode_raw=decode_raw,
        )
    )

    assert logger.end_episode()
    assert len(logger._save_jobs) == 1
    assert logger._save_jobs[0].collection_columns is None
    assert decode_calls == []


def test_collection_raw_save_worker_aligns_and_prepares_video(tmp_path, monkeypatch):
    frame_shapes = []

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            frame_shapes.append(frame.shape)

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *a, **k: _Writer())
    logger = _collection_logger(
        tmp_path,
        save_video=True,
        save_image_height=4,
        save_image_width=6,
        async_save=True,
    )
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                CollectionRawSample(1.0, np.full((8, 8, 3), 10, dtype=np.uint8)),
                CollectionRawSample(1.1, np.full((8, 8, 3), 20, dtype=np.uint8)),
                CollectionRawSample(1.2, np.full((8, 8, 3), 30, dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos": [
                CollectionRawSample(1.0, np.zeros(_DIM, dtype=np.float32)),
                CollectionRawSample(1.2, np.full(_DIM, 2.0, dtype=np.float32)),
            ],
            "action_qpos": [
                CollectionRawSample(1.0, np.full(_DIM, 10.0, dtype=np.float32)),
                CollectionRawSample(1.2, np.full(_DIM, 12.0, dtype=np.float32)),
            ],
        },
    )
    decode_calls = []

    def decode_raw() -> CollectionRawBatch:
        decode_calls.append("decode")
        return batch

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(
            timestamp=1.2,
            decode_raw=decode_raw,
        )
    )
    assert logger.end_episode()
    job = logger._save_jobs[0]
    assert job.collection_columns is None
    assert decode_calls == []

    logger._write_job(job)

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [1.0, 1.1, 1.2])
    np.testing.assert_allclose(table.column("observation.qpos").to_pylist()[1], [1.0] * _DIM)
    np.testing.assert_allclose(table.column("action").to_pylist()[1], [11.0] * _DIM)
    assert frame_shapes == [(4, 6, 3)] * 3
    assert decode_calls == ["decode"]
    episode = _read_jsonl(task_dir / "meta" / "episodes.jsonl")[0]
    assert episode["length"] == 3
    assert episode["quality"] == "green"


def test_collection_save_error_reports_missing_required_stream(tmp_path):
    logger = _collection_logger(tmp_path)
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                CollectionRawSample(0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                CollectionRawSample(0.1, np.zeros((2, 2, 3), dtype=np.uint8)),
            ]
        },
        vectors={
            "state_qpos:arm": [
                CollectionRawSample(0.0, np.zeros(_DIM, dtype=np.float32)),
                CollectionRawSample(0.1, np.ones(_DIM, dtype=np.float32)),
            ],
        },
    )

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(
            timestamp=0.1,
            decode_raw=lambda: batch,
        )
    )

    with pytest.raises(ValueError, match="missing_vector_stream.*action_qpos:arm"):
        logger.end_episode()


def test_collection_flags_frame_count_mismatch_when_camera_frame_dropped(tmp_path):
    # An invalid raw camera sample still occupies an aligned parquet row but is skipped
    # in the mp4, so video frames < data rows. end_episode must flag it red.
    logger = _collection_logger(tmp_path, save_video=True)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    invalid_image = np.zeros((8, 8), dtype=np.uint8)

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        _collection_raw_snapshot(
            0.2,
            batch=CollectionRawBatch(
                images={
                    "cam_high": [
                        CollectionRawSample(0.0, image),
                        CollectionRawSample(0.1, invalid_image),
                        CollectionRawSample(0.2, image),
                    ]
                },
                vectors={
                    "state_qpos": [
                        CollectionRawSample(0.0, qpos),
                        CollectionRawSample(0.2, qpos),
                    ],
                    "action_qpos": [
                        CollectionRawSample(0.0, qpos),
                        CollectionRawSample(0.2, qpos),
                    ],
                },
            ),
        )
    )
    assert logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    episode = _read_jsonl(task_dir / "meta" / "episodes.jsonl")[0]
    assert episode["length"] == 3
    assert episode["quality"] == "red"
    codes = {issue["code"] for issue in episode["quality_issues"]}
    assert "frame_count_mismatch" in codes


def test_collection_prepares_video_frames_during_save(tmp_path, monkeypatch):
    calls = []

    def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
        calls.append((image.shape, height, width))
        return np.zeros((height, width, 3), dtype=image.dtype)

    monkeypatch.setattr(collection_module, "resize_direct", resize)
    logger = _collection_logger(tmp_path, save_video=True, save_image_height=4, save_image_width=6)
    qpos = np.zeros(_DIM, dtype=np.float32)

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        _collection_raw_snapshot(
            0.0,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            state=qpos,
            action=qpos,
        )
    )

    assert calls == []
    assert logger.end_episode()
    assert calls == [((8, 8, 3), 4, 6)]


def test_collection_skips_resize_when_saved_size_matches_frame(tmp_path, monkeypatch):
    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            assert frame.shape == (8, 8, 3)

        def close(self) -> None:
            pass

    def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
        raise AssertionError("resize should be skipped for matching collection frame size")

    monkeypatch.setattr(collection_module, "resize_direct", resize)
    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *a, **k: _Writer())
    logger = _collection_logger(tmp_path, save_video=True, save_image_height=8, save_image_width=8)
    qpos = np.zeros(_DIM, dtype=np.float32)

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        _collection_raw_snapshot(
            0.0,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            state=qpos,
            action=qpos,
        )
    )
    assert logger.end_episode()


def test_collection_qc_updates_saved_episode_while_another_episode_is_queued(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path, async_save=True)
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, image=image, state=qpos))
    assert logger.end_episode()
    first_job = logger._save_jobs[0]
    first_job.status = "saving"
    logger._write_job(first_job)

    assert logger.status_snapshot("t")["episodes"] == []
    assert logger.mark_collection_qc("t", 0, "fail", "still saving") is False

    with logger._lock:
        logger._save_jobs.remove(first_job)

    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1, image=image, state=qpos))
    assert logger.end_episode()

    assert logger.mark_collection_qc("t", 0, "fail", "bad contact") is True
    assert logger.mark_collection_qc("t", 1, "pass", "not saved") is False

    episode = logger.status_snapshot("t")["episodes"][0]
    assert episode["qc_verdict"] == "fail"
    assert episode["qc_note"] == "bad contact"


def test_collection_qc_note_only_preserves_existing_verdict(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, image=image, state=qpos))
    logger.end_episode()

    assert logger.mark_collection_qc("t", 0, "pass", "initial") is True
    assert logger.mark_collection_qc("t", 0, "", "updated note") is True

    episode = logger.status_snapshot("t")["episodes"][0]
    assert episode["qc_verdict"] == "pass"
    assert episode["qc_note"] == "updated note"


def test_collection_qc_and_save_append_share_metadata_lock(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, image=image, state=qpos))
    assert logger.end_episode()

    qc_reading = threading.Event()
    release_qc = threading.Event()
    original_read_jsonl = episode_module._read_jsonl

    def blocking_read_jsonl(path):
        if threading.current_thread().name == "collection_qc":
            qc_reading.set()
            assert release_qc.wait(timeout=5.0)
        return original_read_jsonl(path)

    monkeypatch.setattr(episode_module, "_read_jsonl", blocking_read_jsonl)
    qc_result = []

    def mark_qc():
        qc_result.append(logger.mark_collection_qc("t", 0, "pass", "approved"))

    def save_next_episode():
        logger.start_episode("t")
        logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1, image=image, state=qpos))
        assert logger.end_episode()

    qc_thread = threading.Thread(target=mark_qc, name="collection_qc")
    save_thread = threading.Thread(target=save_next_episode, name="collection_save")
    qc_thread.start()
    assert qc_reading.wait(timeout=5.0)
    save_thread.start()
    release_qc.set()
    qc_thread.join(timeout=5.0)
    save_thread.join(timeout=5.0)

    assert not qc_thread.is_alive()
    assert not save_thread.is_alive()
    assert qc_result == [True]
    episodes = _read_jsonl(_collection_task_dir(tmp_path) / "meta" / "episodes.jsonl")
    assert [episode["episode_index"] for episode in episodes] == [0, 1]
    assert episodes[0]["qc_verdict"] == "pass"


def test_collection_qc_rewrite_is_atomic_for_concurrent_readers(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for timestamp in (0.0, 0.1):
        logger.start_episode("t")
        logger.ingest_collection_snapshot(
            _collection_raw_snapshot(timestamp, image=image, state=qpos)
        )
        assert logger.end_episode()

    second_row_started = threading.Event()
    release_write = threading.Event()
    original_dumps = episode_module.json.dumps
    qc_dumps = 0

    def blocking_dumps(value, *args, **kwargs):
        nonlocal qc_dumps
        if threading.current_thread().name == "collection_qc":
            qc_dumps += 1
            if qc_dumps == 2:
                second_row_started.set()
                assert release_write.wait(timeout=5.0)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(episode_module.json, "dumps", blocking_dumps)
    qc_thread = threading.Thread(
        target=lambda: logger.mark_collection_qc("t", 0, "pass", "approved"),
        name="collection_qc",
    )
    qc_thread.start()
    assert second_row_started.wait(timeout=5.0)

    path = _collection_task_dir(tmp_path) / "meta" / "episodes.jsonl"
    visible_episodes = _read_jsonl(path)
    release_write.set()
    qc_thread.join(timeout=5.0)

    assert not qc_thread.is_alive()
    assert [episode["episode_index"] for episode in visible_episodes] == [0, 1]
    assert _read_jsonl(path)[0]["qc_verdict"] == "pass"


def test_collection_saves_each_task_in_own_dataset_dir(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for task, value in (("pick up cup", 1.0), ("place cup", 2.0)):
        logger.start_episode(task)
        logger.ingest_collection_snapshot(
            _collection_raw_snapshot(
                0.0,
                image=image,
                state=qpos + value,
                action=qpos + value,
            )
        )
        assert logger.end_episode()

    pick_dir = _collection_task_dir(tmp_path, "pick up cup")
    place_dir = _collection_task_dir(tmp_path, "place cup")
    assert pick_dir == tmp_path / "pick_up_cup"
    assert place_dir == tmp_path / "place_cup"
    assert (pick_dir / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (place_dir / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert not (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()

    pick_episode = _read_jsonl(pick_dir / "meta" / "episodes.jsonl")[0]
    place_episode = _read_jsonl(place_dir / "meta" / "episodes.jsonl")[0]
    assert pick_episode["tasks"] == ["pick up cup"]
    assert place_episode["tasks"] == ["place cup"]
    assert logger.status_snapshot("pick up cup")["dataset_dir"] == str(pick_dir)
    assert logger.status_snapshot("pick up cup")["episodes"][0]["episode_index"] == 0
    assert logger.status_snapshot("place cup")["dataset_dir"] == str(place_dir)
    assert logger.status_snapshot("place cup")["episodes"][0]["episode_index"] == 0


def test_collection_status_queue_is_scoped_to_selected_task_dir(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path, async_save=True)
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for task in ("pick up cup", "place cup"):
        logger.start_episode(task)
        logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, image=image, state=qpos))
        assert logger.end_episode()

    pick_status = logger.status_snapshot("pick up cup")
    place_status = logger.status_snapshot("place cup")
    assert pick_status["dataset_dir"] == str(_collection_task_dir(tmp_path, "pick up cup"))
    assert place_status["dataset_dir"] == str(_collection_task_dir(tmp_path, "place cup"))
    assert [item["episode_index"] for item in pick_status["queue"]] == [0]
    assert [item["episode_index"] for item in place_status["queue"]] == [0]
    assert pick_status["episodes"] == []
    assert place_status["episodes"] == []


def test_collection_status_counts_raw_frames_before_save_worker_decodes(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    release_raw = threading.Event()

    def raw_snapshot(timestamp: float) -> RawCollectionSnapshot:
        def decode_raw() -> CollectionRawBatch:
            release_raw.wait(timeout=5.0)
            return _collection_raw_batch((timestamp,), image=image, state=qpos, action=qpos)

        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=decode_raw)

    logger.start_episode("t")
    for i in range(3):
        logger.ingest_collection_snapshot(raw_snapshot(float(i) / 10.0))

    status = logger.status_snapshot("t")
    assert status["current_episode_frames"] == 3
    assert status["current_episode_recorded_frames"] == 0
    assert status["current_episode_pending_frames"] == 3

    release_raw.set()
    assert logger.end_episode()


def test_collection_start_cutoff_drops_cached_frames(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    def snapshot(timestamp: float) -> RawCollectionSnapshot:
        return _collection_raw_snapshot(
            timestamp,
            image=image,
            state=qpos + timestamp,
            action=qpos + timestamp,
        )

    logger.start_episode("t", collection_min_capture_time=10.0)
    logger.ingest_collection_snapshot(snapshot(9.0))
    logger.ingest_collection_snapshot(snapshot(10.0))
    logger.ingest_collection_snapshot(snapshot(10.1))

    status = logger.status_snapshot("t")
    assert status["current_episode_frames"] == 1
    assert status["current_episode_skipped_before_start"] == 2
    assert logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == 1
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), [10.1])


def test_collection_save_worker_decodes_snapshots_after_end_preserving_order(tmp_path):
    logger = _collection_logger(tmp_path, async_save=True)
    logger._start_save_worker = lambda: None
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    decode_order = []

    def decode_raw_first() -> CollectionRawBatch:
        decode_order.append("first")
        return _collection_raw_batch((0.0,), image=image, state=qpos, action=qpos)

    def decode_raw_second() -> CollectionRawBatch:
        decode_order.append("second")
        return _collection_raw_batch((0.1,), image=image, state=qpos, action=qpos)

    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(timestamp=0.0, decode_raw=decode_raw_first)
    )
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(timestamp=0.1, decode_raw=decode_raw_second)
    )

    assert logger.end_episode()
    assert decode_order == []
    job = logger._save_jobs[0]
    logger._write_job(job)

    assert decode_order == ["first", "second"]
    table = pq.read_table(_collection_task_dir(tmp_path) / "data/chunk-000/episode_000000.parquet")
    assert table.column("capture_time").to_pylist() == [0.0, 0.1]


def test_collection_info_fps_uses_target_fps_not_measured(tmp_path, monkeypatch):
    writer_fps = []

    class _Writer:
        def append_data(self, _frame: np.ndarray) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_get_writer(*_args, **kwargs):
        writer_fps.append(kwargs["fps"])
        return _Writer()

    monkeypatch.setattr(episode_module.imageio, "get_writer", fake_get_writer)
    logger = _collection_logger(tmp_path, save_video=True)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for timestamps in ((0.0, 0.1, 0.2), (1.0, 1.1, 1.2)):
        logger.start_episode("t")
        logger.ingest_collection_snapshot(
            _collection_raw_snapshot(
                timestamps[-1],
                batch=_collection_raw_batch(timestamps, image=image, state=qpos, action=qpos),
            )
        )
        assert logger.end_episode()
    logger.finalize()

    task_dir = _collection_task_dir(tmp_path)
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    video_info = info["features"]["observation.images.cam_high"]["info"]
    # info.fps and every mp4 use the integer target fps (10), not the jittery measured
    # mean, so timestamp[i] == i/fps holds for LeRobot validation.
    np.testing.assert_allclose(info["fps"], 10.0)
    np.testing.assert_allclose(video_info["video.fps"], 10.0)
    np.testing.assert_allclose(writer_fps, [10.0, 10.0])


def test_image_episode_stats_does_not_stack_all_frames_as_float64(monkeypatch):
    original_asarray = episode_module.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None and args:
            dtype = args[0]
        if isinstance(value, list) and np.dtype(dtype) == np.dtype(np.float64):
            raise AssertionError("image stats must stream frames instead of stacking them")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(episode_module.np, "asarray", guarded_asarray)

    frames = [
        np.array([[[0, 10, 20], [30, 40, 50]]], dtype=np.uint8),
        np.array([[[60, 70, 80], [90, 100, 110]]], dtype=np.uint8),
    ]

    stats = episode_module._image_episode_stats(frames)

    expected = original_asarray(frames, dtype=np.float64).reshape(-1, 3) / 255.0
    expected_mean = expected.mean(axis=0)
    expected_std = expected.std(axis=0)
    np.testing.assert_allclose(stats["min"], [[[float(x)]] for x in expected.min(axis=0)])
    np.testing.assert_allclose(stats["max"], [[[float(x)]] for x in expected.max(axis=0)])
    np.testing.assert_allclose(stats["mean"], [[[float(x)]] for x in expected_mean])
    np.testing.assert_allclose(stats["std"], [[[float(x)]] for x in expected_std])
    assert stats["count"] == [2]


def _record_one_collection_episode(logger, image):
    qpos = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("t")
    logger.ingest_collection_snapshot(
        _collection_raw_snapshot(
            0.2,
            batch=_collection_raw_batch((0.0, 0.1, 0.2), image=image, state=qpos, action=qpos),
        )
    )
    assert logger.end_episode()
    logger.finalize()


def test_collection_resizes_saved_frames_when_size_set(tmp_path, monkeypatch):
    frame_shapes = []

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            frame_shapes.append(frame.shape)

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *a, **k: _Writer())
    logger = _collection_logger(
        tmp_path, save_video=True, save_image_height=120, save_image_width=160
    )
    _record_one_collection_episode(logger, np.zeros((480, 640, 3), dtype=np.uint8))

    assert frame_shapes == [(120, 160, 3)] * 3
    task_dir = _collection_task_dir(tmp_path)
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.cam_high"]["shape"] == [120, 160, 3]


def test_collection_keeps_native_size_when_size_unset(tmp_path, monkeypatch):
    frame_shapes = []

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            frame_shapes.append(frame.shape)

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *a, **k: _Writer())
    logger = _collection_logger(tmp_path, save_video=True)
    _record_one_collection_episode(logger, np.zeros((480, 640, 3), dtype=np.uint8))

    assert frame_shapes == [(480, 640, 3)] * 3
    task_dir = _collection_task_dir(tmp_path)
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.cam_high"]["shape"] == [480, 640, 3]


def test_collection_stats_json_uses_episode_stats_without_parquet_rescan(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for offset in (0.0, 10.0):
        qpos = np.arange(_DIM, dtype=np.float32) + offset
        logger.start_episode("t")
        logger.ingest_collection_snapshot(
            _collection_raw_snapshot(
                0.2,
                batch=_collection_raw_batch(
                    (0.0, 0.1, 0.2),
                    image=image,
                    state=qpos,
                    action=qpos + 1.0,
                ),
            )
        )
        assert logger.end_episode()

    def fail_read_table(*_args, **_kwargs):
        raise AssertionError("stats.json should use episodes_stats.jsonl")

    monkeypatch.setattr(collection_module.pq, "read_table", fail_read_table)
    logger.finalize()

    task_dir = _collection_task_dir(tmp_path)
    stats = json.loads((task_dir / "meta" / "stats.json").read_text())
    expected_qpos = np.concatenate(
        [
            np.tile(np.arange(_DIM, dtype=np.float64), (3, 1)),
            np.tile(np.arange(_DIM, dtype=np.float64) + 10.0, (3, 1)),
        ],
        axis=0,
    )
    np.testing.assert_allclose(stats["observation.qpos"]["mean"], expected_qpos.mean(axis=0))
    np.testing.assert_allclose(stats["observation.qpos"]["std"], expected_qpos.std(axis=0))


def test_record_step_timestamp_is_ideal_grid_capture_time_keeps_raw(tmp_path):
    # The eval/inference path (record_step) had the same fatal bug as collection:
    # parquet timestamp stored raw wall-clock (absolute epoch seconds), which breaks
    # LeRobot's timestamp[i] == i/fps (tol 1e-4) AND the replay clock — the viewer
    # seeks the mp4 by timestamp, so ~1.7e9 values land far past the clip and it shows
    # 0s. Synthesize the grid from target fps; raw time moves aside into capture_time.
    logger = _logger(tmp_path)  # fps=30
    state = np.zeros(_DIM, dtype=np.float32)

    raw = [1717000000.0, 1717000000.034, 1717000000.067, 1717000000.099]
    logger.start_episode("t")
    for ts in raw:
        logger.record_step(_obs(state), state, timestamp=ts)
    logger.end_episode()

    table = pq.read_table(str(tmp_path / "data" / "chunk-000" / "episode_000000.parquet"))
    expected = [i / 30.0 for i in range(len(raw))]
    np.testing.assert_allclose(table.column("timestamp").to_pylist(), expected, atol=1e-4)
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), raw)


def test_episode_logger_info_and_video_fps_use_target_fps(tmp_path, monkeypatch):
    writer_fps = []
    writer_frames = []

    class _Writer:
        def __init__(self) -> None:
            self.frames = 0

        def append_data(self, _frame: np.ndarray) -> None:
            self.frames += 1

        def close(self) -> None:
            writer_frames.append(self.frames)

    def fake_get_writer(*_args, **kwargs):
        writer_fps.append(kwargs["fps"])
        return _Writer()

    monkeypatch.setattr(episode_module.imageio, "get_writer", fake_get_writer)
    logger = _logger(tmp_path, save_video=True)  # fps=30
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for timestamps in ((0.0, 0.1, 0.2), (1.0, 1.05, 1.1)):
        logger.start_episode("t")
        for ts in timestamps:
            logger.record_step(
                _obs(np.zeros(_DIM, dtype=np.float32), image),
                np.zeros(_DIM, np.float32),
                timestamp=ts,
            )
        logger.end_episode()
    logger.finalize()

    episodes = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    video_info = info["features"]["observation.images.cam_high"]["info"]
    # measured per-episode fps stays in episodes.jsonl as a jitter diagnostic, but
    # info.fps and every mp4 use the integer target fps (30) — so timestamp[i] == i/fps
    # holds for LeRobot validation, mirroring the collection path.
    np.testing.assert_allclose([e["episode_fps"] for e in episodes], [10.0, 20.0])
    np.testing.assert_allclose(info["fps"], 30.0)
    np.testing.assert_allclose(video_info["video.fps"], 30.0)
    np.testing.assert_allclose(writer_fps, [30.0, 30.0])
    assert writer_frames == [7, 4]
    assert "capture_time" in info["features"]


def test_image_episode_stats_matches_stacked_formula():
    frames = [
        np.array(
            [
                [[0, 10, 20], [30, 40, 50]],
                [[60, 70, 80], [90, 100, 110]],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [[120, 130, 140], [150, 160, 170]],
                [[180, 190, 200], [210, 220, 230]],
            ],
            dtype=np.uint8,
        ),
    ]
    frames.append(frames[0][:, :, ::-1])
    stats = episode_module._image_episode_stats(frames)

    stacked = np.asarray(frames, dtype=np.float64) / 255.0
    per_channel = stacked.reshape(-1, 3)

    def flat(values):
        return np.asarray(values, dtype=np.float64).reshape(3)

    np.testing.assert_allclose(flat(stats["min"]), per_channel.min(axis=0))
    np.testing.assert_allclose(flat(stats["max"]), per_channel.max(axis=0))
    np.testing.assert_allclose(flat(stats["mean"]), per_channel.mean(axis=0))
    np.testing.assert_allclose(flat(stats["std"]), per_channel.std(axis=0))
    assert stats["count"] == [3]
