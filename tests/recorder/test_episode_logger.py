"""Tests for EpisodeLogger (core.recorder.episode) — synchronous parquet/meta
writing, multi-episode lifecycle continuation, and input-copy / empty-episode
data integrity.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pyarrow.parquet as pq

from core.config import ConfigDict
from core.recorder import collection as collection_module
from core.recorder import episode as episode_module
from core.recorder.episode import EpisodeLogger, sanitize_path_component
from core.types import Observation, RawCollectionSnapshot
from core.utils.lerobot import LeRobotDatasetIO
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


def _logger(log_dir, *, save_video: bool = False) -> EpisodeLogger:
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


def test_episode_logger_sync_save_writes_n_rows(tmp_path):
    keys = ConfigDict(
        state_key="observations.state.qpos",
        eef_key="observations.state.eef",
        action_key="action",
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
        action_key="action",
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

    raw = [73529.6, 73529.633, 73529.667, 73529.702]
    logger.start_episode("t")
    for ts in raw:
        logger.record_collection_frame(
            Observation(
                timestamp=ts,
                images={"cam_high": image},
                state_qpos=qpos,
                action_qpos=qpos,
            )
        )
    logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    timestamps = table.column("timestamp").to_pylist()
    expected = [i / 10.0 for i in range(len(raw))]
    np.testing.assert_allclose(timestamps, expected, atol=1e-4)
    np.testing.assert_allclose(table.column("capture_time").to_pylist(), raw)


def test_collection_flags_frame_count_mismatch_when_camera_frame_dropped(tmp_path):
    # A frame that drops a camera image still occupies a parquet row but is skipped in
    # the mp4 (_snapshot_videos), so video frames < data rows — the exact divergence that
    # crashes LeRobot on load. end_episode must flag it red as frame_count_mismatch.
    logger = _collection_logger(tmp_path, save_video=True)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    logger.start_episode("t")
    for i in range(3):
        images = {} if i == 1 else {"cam_high": image}  # drop the camera on frame 1
        logger.record_collection_frame(
            Observation(timestamp=i / 10.0, images=images, state_qpos=qpos, action_qpos=qpos)
        )
    logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    episode = _read_jsonl(task_dir / "meta" / "episodes.jsonl")[0]
    assert episode["length"] == 3
    assert episode["quality"] == "red"
    codes = {issue["code"] for issue in episode["quality_issues"]}
    assert "frame_count_mismatch" in codes


def test_collection_status_reflects_qc_written_after_save(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    logger.start_episode("t")
    logger.record_collection_frame(
        Observation(
            timestamp=0.0,
            images={"cam_high": image},
            state_qpos=qpos,
            action_qpos=qpos,
        )
    )
    logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    assert LeRobotDatasetIO(task_dir).mark_qc(0, "fail", "bad contact") is True

    episode = logger.status_snapshot("t")["episodes"][0]
    assert episode["qc_verdict"] == "fail"
    assert episode["qc_note"] == "bad contact"


def test_collection_saves_each_task_in_own_dataset_dir(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for task, value in (("pick up cup", 1.0), ("place cup", 2.0)):
        logger.start_episode(task)
        logger.record_collection_frame(
            Observation(
                timestamp=0.0,
                images={"cam_high": image},
                state_qpos=qpos + value,
                action_qpos=qpos + value,
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
        logger.record_collection_frame(
            Observation(
                timestamp=0.0,
                images={"cam_high": image},
                state_qpos=qpos,
                action_qpos=qpos,
            )
        )
        assert logger.end_episode()

    pick_status = logger.status_snapshot("pick up cup")
    place_status = logger.status_snapshot("place cup")
    assert pick_status["dataset_dir"] == str(_collection_task_dir(tmp_path, "pick up cup"))
    assert place_status["dataset_dir"] == str(_collection_task_dir(tmp_path, "place cup"))
    assert [item["episode_index"] for item in pick_status["queue"]] == [0]
    assert [item["episode_index"] for item in place_status["queue"]] == [0]
    assert pick_status["episodes"] == []
    assert place_status["episodes"] == []


def test_collection_status_counts_raw_frames_before_ingest_catches_up(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    release_decode = threading.Event()

    def decode() -> Observation:
        release_decode.wait(timeout=5.0)
        return Observation(
            timestamp=0.0,
            images={"cam_high": image},
            state_qpos=qpos,
            action_qpos=qpos,
        )

    logger.start_episode("t")
    for _ in range(3):
        logger.ingest_collection_snapshot(RawCollectionSnapshot(timestamp=0.0, decode=decode))

    status = logger.status_snapshot("t")
    assert status["current_episode_frames"] == 3
    assert status["current_episode_recorded_frames"] == 0
    assert status["current_episode_pending_frames"] == 3

    release_decode.set()
    assert logger.end_episode()


def test_collection_start_cutoff_drops_cached_frames(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    def snapshot(timestamp: float) -> RawCollectionSnapshot:
        def decode() -> Observation:
            return Observation(
                timestamp=timestamp,
                images={"cam_high": image},
                state_qpos=qpos + timestamp,
                action_qpos=qpos + timestamp,
            )

        return RawCollectionSnapshot(timestamp=timestamp, decode=decode)

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

    for timestamps in ((0.0, 0.1, 0.2), (1.0, 1.05, 1.1)):
        logger.start_episode("t")
        for ts in timestamps:
            logger.record_collection_frame(
                Observation(
                    timestamp=ts,
                    images={"cam_high": image},
                    state_qpos=qpos,
                    action_qpos=qpos,
                )
            )
        logger.end_episode()
    logger.finalize()

    task_dir = _collection_task_dir(tmp_path)
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    video_info = info["features"]["observation.images.cam_high"]["info"]
    # info.fps and every mp4 use the integer target fps (10), not the jittery measured
    # mean, so timestamp[i] == i/fps holds for LeRobot validation.
    np.testing.assert_allclose(info["fps"], 10.0)
    np.testing.assert_allclose(video_info["video.fps"], 10.0)
    np.testing.assert_allclose(writer_fps, [10.0, 10.0])


def _record_one_collection_episode(logger, image):
    qpos = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("t")
    for ts in (0.0, 0.1, 0.2):
        logger.record_collection_frame(
            Observation(timestamp=ts, images={"cam_high": image}, state_qpos=qpos, action_qpos=qpos)
        )
    logger.end_episode()
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
        for ts in (0.0, 0.1, 0.2):
            logger.record_collection_frame(
                Observation(
                    timestamp=ts,
                    images={"cam_high": image},
                    state_qpos=qpos,
                    action_qpos=qpos + 1.0,
                )
            )
        logger.end_episode()

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
    assert writer_frames == [3, 3]
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


def test_collection_writes_eef_uv_column_when_frames_carry_it(tmp_path):
    """When an Observation carries eef_uv dict, per-camera uv columns land in parquet."""
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    logger.start_episode("t")
    for i in range(3):
        logger.record_collection_frame(
            Observation(
                timestamp=i / 10.0,
                images={"cam_high": image},
                state_qpos=qpos,
                action_qpos=qpos,
                eef_uv={
                    "cam_high": np.array([[100.0 + i, 200.0 + i]], dtype=np.float32),
                },
            )
        )
    logger.end_episode()

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert "observation.eef_uv.cam_high" in table.column_names
    rows = table.column("observation.eef_uv.cam_high").to_pylist()
    assert rows == [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]]


def test_collection_omits_eef_uv_column_when_frames_lack_it(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    logger.start_episode("t")
    for i in range(2):
        logger.record_collection_frame(
            Observation(
                timestamp=i / 10.0,
                images={"cam_high": image},
                state_qpos=qpos,
                action_qpos=qpos,
            )
        )
    logger.end_episode()
    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    uv_cols = [c for c in table.column_names if c.startswith("observation.eef_uv")]
    assert uv_cols == []
