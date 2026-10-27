"""Collection persistence across LeRobot storage chunks."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.recorder import episode as episode_module
from core.types import CollectionRawBatch, CollectionRawSample, RawCollectionSnapshot
from core.utils.lerobot import LeRobotDatasetIO
from tests.integration.test_dataset_concurrency import _record_episode
from tests.integration.test_episode_logger import (
    _collection_logger,
    _collection_task_dir,
    _read_jsonl,
)

pytestmark = pytest.mark.integration

_DIM = 6
_TASK = "t"
_VIDEO_KEYS = {"cam_high": "observation.images.cam_high"}
_DEFAULT_LAYOUT = {
    "chunks_size": 1000,
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _seed_dataset(root: Path, episode_index: int, *, task: str = _TASK) -> Path:
    """Create one valid prior episode so a real logger starts at a boundary."""
    dataset_dir = _collection_task_dir(root, task)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    row = {
        "episode_index": episode_index,
        "tasks": [task],
        "length": 1,
        "global_index_start": 0,
        "quality": "green",
        "video_keys": [],
        "total_videos": 0,
    }
    (dataset_dir / "meta" / "episodes.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    (dataset_dir / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8"
    )
    _write_json(dataset_dir / "meta" / "info.json", _DEFAULT_LAYOUT)
    table = pa.table(
        {
            "observation.qpos": [[-1.0] * _DIM],
            "action": [[-1.0] * _DIM],
            "timestamp": [0.0],
            "capture_time": [0.0],
            "frame_index": [0],
            "episode_index": [episode_index],
            "index": [0],
            "task_index": [0],
        }
    )
    pq.write_table(
        table,
        dataset_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
    )
    return dataset_dir


def _snapshot(value: float, start: float = 1.0) -> RawCollectionSnapshot:
    """Build a small changing RGB/vector stream for one collection episode."""
    times = [start, start + 0.1, start + 0.2]
    frames = [np.full((8, 8, 3), int(value + step), dtype=np.uint8) for step in range(3)]
    vectors = [np.full(_DIM, value + step, dtype=np.float32) for step in range(3)]
    batch = CollectionRawBatch(
        images={
            "cam_high": [
                CollectionRawSample(t, frame) for t, frame in zip(times, frames, strict=True)
            ]
        },
        vectors={
            "state_qpos": [
                CollectionRawSample(t, vector) for t, vector in zip(times, vectors, strict=True)
            ],
            "action_qpos": [
                CollectionRawSample(t, vector + 10.0)
                for t, vector in zip(times, vectors, strict=True)
            ],
        },
    )
    return RawCollectionSnapshot(timestamp=times[-1], decode_raw=lambda: batch)


def _save_episode(logger, value: float, slot: str, start: float) -> int:
    logger.start_episode(_TASK)
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_snapshot(value, start))
    assert logger.end_episode()
    logger.wait_for_saves(timeout=20, fail_on_error=True)
    rows = _read_jsonl(_collection_task_dir(logger._log_dir) / "meta/episodes.jsonl")
    return max(int(row["episode_index"]) for row in rows if row.get("slot_id") == slot)


def _assert_episode_files(io: LeRobotDatasetIO, episode: int, expected_parent: str) -> None:
    parquet = io.episode_parquet(episode)
    assert parquet.exists()
    assert parquet.parent.name == expected_parent
    videos = io.episode_video_paths(episode, _VIDEO_KEYS)
    assert set(videos) == {"cam_high"}
    assert videos["cam_high"].exists()


def test_collection_round_trip_retake_and_restart_across_chunk_boundary(tmp_path):
    dataset_dir = _seed_dataset(tmp_path, 998)
    logger = _collection_logger(tmp_path)
    try:
        assert _save_episode(logger, 1.0, "slot-999", 1.0) == 999
        assert _save_episode(logger, 2.0, "slot-1000", 2.0) == 1000
        assert _save_episode(logger, 3.0, "slot-1001", 3.0) == 1001
        io = LeRobotDatasetIO(dataset_dir)
        for episode in (999, 1000, 1001):
            _assert_episode_files(io, episode, "chunk-000" if episode == 999 else "chunk-001")
            actions, task = io.load_trajectory(episode, "action")
            assert task == _TASK
            assert actions.shape == (3, _DIM)

        assert logger.mark_collection_qc(_TASK, 999, "pass", "boundary saved")
        assert logger.mark_collection_qc(_TASK, 1000, "pass", "boundary saved")
        assert logger.mark_collection_qc(_TASK, 1001, "pass", "boundary saved")

        assert _save_episode(logger, 20.0, "slot-1000", 4.0) == 1000
        assert logger.mark_collection_qc(_TASK, 1000, "pass", "retake saved")
        rows = _read_jsonl(dataset_dir / "meta/episodes.jsonl")
        assert sorted(int(row["episode_index"]) for row in rows) == [998, 999, 1000, 1001]
        retake, _ = io.load_trajectory(1000, "action")
        np.testing.assert_allclose(retake[0], np.full(_DIM, 30.0, dtype=np.float32))
    finally:
        logger.finalize()

    resumed = _collection_logger(tmp_path)
    try:
        assert _save_episode(resumed, 4.0, "slot-1002", 5.0) == 1002
    finally:
        resumed.finalize()

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 5
    assert info["total_chunks"] == 2
    assert info["chunks_size"] == 1000
    stats = json.loads((dataset_dir / "meta" / "stats.json").read_text())
    assert stats["observation.qpos"]["max"][0] == 22.0
    qc = {int(row["episode_index"]): row for row in _read_jsonl(dataset_dir / "meta/qc.jsonl")}
    assert all(qc[index]["qc_verdict"] == "pass" for index in (999, 1000, 1001))


def test_collection_custom_layout_is_preserved_and_readable(tmp_path):
    dataset_dir = _collection_task_dir(tmp_path)
    custom = {
        "chunks_size": 2,
        "data_path": "tables/part-{episode_chunk:02d}/row-{episode_index:06d}.parquet",
        "video_path": "media/part-{episode_chunk:02d}/{video_key}/row-{episode_index:06d}.mp4",
    }
    _write_json(dataset_dir / "meta" / "info.json", custom)
    logger = _collection_logger(tmp_path)
    try:
        for episode, value in enumerate((1.0, 2.0, 3.0)):
            assert _save_episode(logger, value, f"custom-{episode}", 10.0 + episode) == episode
    finally:
        logger.finalize()

    io = LeRobotDatasetIO(dataset_dir)
    for episode in range(3):
        assert io.episode_parquet(episode).parent.name == f"part-{episode // 2:02d}"
        _assert_episode_files(io, episode, f"part-{episode // 2:02d}")
        actions, task = io.load_trajectory(episode, "action")
        assert task == _TASK
        np.testing.assert_allclose(actions[0], np.full(_DIM, episode + 11.0))
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert {key: info[key] for key in custom} == custom
    assert info["total_chunks"] == 2


def test_custom_layout_resume_reads_frame_indices_without_start_metadata(tmp_path):
    dataset_dir = _collection_task_dir(tmp_path)
    custom = {
        "chunks_size": 2,
        "data_path": "tables/part-{episode_chunk:02d}/row-{episode_index:06d}.parquet",
        "video_path": "media/part-{episode_chunk:02d}/{video_key}/row-{episode_index:06d}.mp4",
    }
    _write_json(dataset_dir / "meta/info.json", custom)
    row = {"episode_index": 0, "tasks": [_TASK], "length": 1, "video_keys": []}
    (dataset_dir / "meta/episodes.jsonl").write_text(json.dumps(row) + "\n")
    path = LeRobotDatasetIO(dataset_dir).episode_path(0, "data_path")
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation.qpos": [[0.0] * _DIM],
                "action": [[0.0] * _DIM],
                "index": [99],
            }
        ),
        path,
    )

    logger = _collection_logger(tmp_path)
    try:
        assert _save_episode(logger, 1.0, "resumed", 1.0) == 1
    finally:
        logger.finalize()

    saved = LeRobotDatasetIO(dataset_dir).episode_parquet(1)
    assert pq.read_table(saved, columns=["index"]).column("index").to_pylist() == [100, 101, 102]


def test_collection_retake_preserves_custom_episode_file_stem(tmp_path):
    dataset_dir = _collection_task_dir(tmp_path)
    custom = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/{episode_file}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/{episode_file}.mp4",
    }
    _write_json(dataset_dir / "meta/info.json", custom)
    logger = _collection_logger(tmp_path)
    try:
        assert _save_episode(logger, 1.0, "named-take", 1.0) == 0
        logger._collection_qc.wait(timeout=10)
        io = LeRobotDatasetIO(dataset_dir)
        paths = [io.episode_parquet(0), *io.episode_video_paths(0, _VIDEO_KEYS).values()]
        for path in paths:
            path.rename(path.with_name("named-take" + path.suffix))
        metadata = dataset_dir / "meta/episodes.jsonl"
        rows = _read_jsonl(metadata)
        rows[0]["file_stem"] = "named-take"
        metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))

        assert _save_episode(logger, 10.0, "named-take", 2.0) == 0
        assert io.episode_parquet(0).name == "named-take.parquet"
        assert io.episode_video_paths(0, _VIDEO_KEYS)["cam_high"].name == "named-take.mp4"
        actions, _ = io.load_trajectory(0, "action")
        np.testing.assert_allclose(actions[0], np.full(_DIM, 20.0))
    finally:
        logger.finalize()


def test_saved_episode_survives_qc_submit_failure_and_still_notifies_render(
    tmp_path, monkeypatch
):
    dataset_dir = _seed_dataset(tmp_path, 998)
    logger = _collection_logger(tmp_path, async_save=True)
    logger._collection["render_service"] = {"url": "http://render.test"}
    notifications = []
    monkeypatch.setattr(
        episode_module,
        "notify_render_service",
        lambda url, path, episode: notifications.append((url, Path(path), episode)),
    )
    monkeypatch.setattr(
        logger._collection_qc,
        "submit",
        lambda *_args: (_ for _ in ()).throw(OSError("qc unavailable")),
    )
    try:
        assert _save_episode(logger, 5.0, "qc-error", 20.0) == 999
    finally:
        logger.finalize()

    assert notifications == [("http://render.test", dataset_dir, 999)]
    assert LeRobotDatasetIO(dataset_dir).episode_parquet(999).exists()
    qc = _read_jsonl(dataset_dir / "meta/qc.jsonl")
    row = next(item for item in qc if int(item["episode_index"]) == 999)
    assert row["qc_auto"] is True
    assert row["qc_note"] == "qc unavailable"
    assert row["qc_reason"] == "auto_qc_error"


def _record_boundary_station(root: Path, task: str, barrier) -> None:
    logger = _collection_logger(root, shared_dataset=True)
    barrier.wait(20.0)
    _record_episode(logger, task, 30.0 if task == "a" else 40.0, 0)
    logger.finalize()


def test_two_collection_writers_allocate_both_sides_of_chunk_boundary(tmp_path):
    dataset_dir = _seed_dataset(tmp_path, 999, task="shared")
    context = mp.get_context("fork")
    barrier = context.Barrier(3)
    stations = [
        context.Process(target=_record_boundary_station, args=(tmp_path, task, barrier))
        for task in ("a", "b")
    ]
    for station in stations:
        station.start()
    barrier.wait(30.0)
    for station in stations:
        station.join(60.0)
        assert station.exitcode == 0

    rows = _read_jsonl(dataset_dir / "meta/episodes.jsonl")
    assert sorted(int(row["episode_index"]) for row in rows) == [999, 1000, 1001]
    io = LeRobotDatasetIO(dataset_dir)
    for episode in (1000, 1001):
        _assert_episode_files(io, episode, "chunk-001")
        table = pq.read_table(io.episode_parquet(episode))
        assert set(table.column("episode_index").to_pylist()) == {episode}
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert info["total_chunks"] == 2
