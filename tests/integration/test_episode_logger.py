"""Tests for EpisodeLogger (core.recorder.episode) — synchronous parquet/meta
writing, multi-episode lifecycle continuation, and input-copy / empty-episode
data integrity.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.config import ConfigDict
from core.recorder import episode as episode_module
from core.recorder.episode import EpisodeLogger, sanitize_path_component
from core.types import (
    CollectionRawBatch,
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

pytestmark = pytest.mark.integration

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
    fps: int = 30,
    async_save: bool = False,
    save_queue_max: int = 15,
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
        save_queue_max=save_queue_max,
    )


def _record_single_step(logger: EpisodeLogger, *, timestamp: float) -> None:
    state = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("task")
    logger.record_step(_obs(state), state, timestamp=timestamp)
    assert logger.end_episode()


def test_failed_async_save_does_not_consume_queue_capacity(tmp_path, monkeypatch):
    logger = _logger(tmp_path, async_save=True, save_queue_max=1)
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)
    _record_single_step(logger, timestamp=1.0)
    monkeypatch.setattr(
        logger,
        "_write_job",
        lambda _job: (_ for _ in ()).throw(OSError("broken save")),
    )
    logger._save_queue_worker()

    status = logger.status_snapshot()
    assert status["pipeline_state"] == "IDLE"
    assert status["save_queue_size"] == 0
    assert status["queue"][0]["status"] == "failed"
    assert status["queue"][0]["error"] == "broken save"
    assert logger.wait_for_saves()
    assert not logger.is_queue_full()

    _record_single_step(logger, timestamp=2.0)
    assert logger.is_queue_full()
    status = logger.status_snapshot()
    assert status["pipeline_state"] == "QUEUE_FULL"
    assert status["save_queue_size"] == 1
    assert [item["status"] for item in status["queue"]] == ["failed", "queued"]


def _collection_logger(
    log_dir,
    *,
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
            storage=ConfigDict(data_root=str(_collection_data_root(log_dir))),
            tasks={},
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
    )


def _collection_data_root(root):
    return root / "data"


def _collection_task_dir(root, task: str = "t"):
    """The dataset's one directory: <data_root>/datasets/real_robot/<robot>/<task>."""
    return (
        _collection_data_root(root)
        / "datasets"
        / "real_robot"
        / "fake_arm"
        / sanitize_path_component(task)
    )


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
) -> RawCollectionSnapshot:
    image_value = np.zeros((8, 8, 3), dtype=np.uint8) if image is None else np.asarray(image)
    state_value = (
        np.zeros(_DIM, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32)
    )
    batch = CollectionRawBatch(
        images={"cam_high": [CollectionRawSample(timestamp, image_value)]},
        vectors={
            "state_qpos": [CollectionRawSample(timestamp, state_value)],
            "action_qpos": [CollectionRawSample(timestamp, state_value)],
        },
    )
    return RawCollectionSnapshot(timestamp=timestamp, decode_raw=lambda: batch)


def test_episode_logger_resume_skips_gaps_without_reusing_persisted_indices(tmp_path):
    logger = _logger(tmp_path)
    state = np.zeros(_DIM, dtype=np.float32)
    logger.start_episode("a")
    logger.record_step(_obs(state), state)
    logger.end_episode()

    episodes_path = tmp_path / "meta" / "episodes.jsonl"
    episodes = _read_jsonl(episodes_path)
    episodes.append({"episode_index": 7, "tasks": ["a"], "length": 1})
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in episodes))

    original = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet")
    orphan = original.set_column(
        original.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array([9], type=pa.int64()),
    ).set_column(
        original.schema.get_field_index("index"),
        "index",
        pa.array([99], type=pa.int64()),
    )
    pq.write_table(orphan, tmp_path / "data/chunk-000/episode_000009.parquet")

    resumed = _logger(tmp_path)
    assert resumed.current_episode_index == 10
    resumed.start_episode("b")
    resumed.record_step(_obs(state), state)
    resumed.end_episode()

    saved = pq.read_table(tmp_path / "data/chunk-000/episode_000010.parquet")
    assert saved.column("index").to_pylist() == [100]


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


def test_failed_wire_decode_releases_capture_journal(tmp_path, monkeypatch):
    from transport.zmq import _WireCaptureJournal

    logger = _collection_logger(tmp_path, async_save=True)
    monkeypatch.setattr(logger, "_start_save_worker", lambda: None)
    monkeypatch.setattr(episode_module.logger, "exception", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(episode_module, "_trim_process_allocator", lambda: None)

    journal = _WireCaptureJournal(tmp_path)
    entry = journal.append(b"wire payload")
    journal.finish()
    journal._writer.join(timeout=1)
    assert not journal._writer.is_alive()
    journal_ref = weakref.ref(journal)
    file_ref = weakref.ref(journal._file)
    file_descriptor = journal._file.fileno()

    def decode_raw(entry=entry) -> CollectionRawBatch:
        entry.read()
        try:
            raise ValueError("invalid msgpack")
        except ValueError as cause:
            raise OSError("wire decode failed") from cause

    snapshot = RawCollectionSnapshot(timestamp=0.1, decode_raw=decode_raw)
    logger.start_episode("t")
    logger.ingest_collection_snapshot(snapshot)
    assert logger.end_episode()

    del decode_raw, entry, journal, snapshot
    logger._save_queue_worker()
    gc.collect()

    job = logger._save_jobs[0]
    assert job.status == "failed"
    assert job.error == "wire decode failed"
    assert journal_ref() is None
    assert file_ref() is None
    with pytest.raises(OSError):
        os.fstat(file_descriptor)


def test_collection_qc_lands_in_the_ledger_next_to_a_save(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, image=image, state=qpos))
    assert logger.end_episode()

    qc_results = []

    def mark_qc():
        qc_results.append(logger.mark_collection_qc("t", 0, "pass", "approved"))

    qc_thread = threading.Thread(target=mark_qc, name="collection_qc")
    qc_thread.start()
    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1, image=image, state=qpos))
    assert logger.end_episode()
    qc_thread.join(timeout=5.0)

    assert not qc_thread.is_alive()
    assert qc_results == [True]
    dataset_dir = _collection_task_dir(tmp_path)
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    assert [episode["episode_index"] for episode in episodes] == [0, 1]
    # The verdict lives only in the ledger, never on the recording rows.
    assert all("qc_verdict" not in episode for episode in episodes)
    ledger = _read_jsonl(dataset_dir / "meta" / "qc.jsonl")
    assert [(row["episode_index"], row["qc_verdict"]) for row in ledger] == [(0, "pass")]


def test_collection_qc_failure_leaves_the_ledger_intact(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for timestamp in (0.0, 0.1):
        logger.start_episode("t")
        logger.ingest_collection_snapshot(
            _collection_raw_snapshot(timestamp, image=image, state=qpos)
        )
        assert logger.end_episode()
    assert logger.mark_collection_qc("t", 0, "pass", "approved")
    ledger = _collection_task_dir(tmp_path) / "meta" / "qc.jsonl"
    before = ledger.read_bytes()

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(ledger), "write_text", explode)
    with pytest.raises(OSError):
        logger.mark_collection_qc("t", 1, "fail", "camera offline")

    assert ledger.read_bytes() == before


@pytest.mark.parametrize("failure", ["frozen"])
def test_camera_failure_is_persisted_as_red_quality(tmp_path, failure):
    logger = _collection_logger(tmp_path)
    times = [index / 10 for index in range(21)]
    batch = CollectionRawBatch(
        start_time=0,
        end_time=2,
        images={}
        if failure == "missing"
        else {
            "cam_high": [
                CollectionRawSample(
                    t,
                    np.full((8, 8, 3), 0 if failure == "frozen" else round(t * 10), dtype=np.uint8),
                )
                for t in times
            ]
        },
        vectors={
            field: [CollectionRawSample(t, np.full(_DIM, t, dtype=np.float32)) for t in times]
            for field in ("state_qpos", "action_qpos")
        },
    )
    logger.start_episode("t")
    logger.ingest_collection_snapshot(RawCollectionSnapshot(timestamp=2, decode_raw=lambda: batch))
    assert logger.end_episode()
    episode = _read_jsonl(_collection_task_dir(tmp_path) / "meta" / "episodes.jsonl")[0]
    assert episode["quality"] == "red"
    expected = {
        "frozen": "frozen_camera",
        "missing": "missing_camera_stream",
    }[failure]
    assert expected in {issue["code"] for issue in episode["quality_issues"]}
