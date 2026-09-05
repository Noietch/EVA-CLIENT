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
    save_video: bool = False,
    fps: int = 30,
    async_save: bool = False,
    recording_space: str = "qpos",
    eval_mode: bool = False,
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
        recording_space=recording_space,
        eval_mode=eval_mode,
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
    save_video: bool = False,
    save_image_height: int | None = None,
    save_image_width: int | None = None,
    async_save: bool = False,
    image_skew_tolerance_sec: float | None = None,
    tasks: dict[str, list[tuple[str, int]]] | None = None,
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
            tasks=tasks or {},
            storage=ConfigDict(image_skew_tolerance_sec=image_skew_tolerance_sec),
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
    return root / sanitize_path_component(task) / "raw"


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
    assert logger.status_snapshot()["queue"][0]["length"] == 2

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


def test_raw_episode_recomputes_image_skew_after_excluding_warmup(tmp_path):
    logger = _logger(tmp_path, fps=10)

    def snapshot(timestamp: float, image_timestamp: float) -> RawCollectionSnapshot:
        state = np.full(_DIM, timestamp, dtype=np.float32)
        batch = CollectionRawBatch(
            images={
                "cam_high": [
                    CollectionRawSample(
                        image_timestamp,
                        np.zeros((8, 8, 3), dtype=np.uint8),
                    )
                ]
            },
            vectors={"state_qpos": [CollectionRawSample(timestamp, state)]},
        )
        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=lambda: batch)

    logger.start_episode("t")
    for timestamp, image_timestamp in (
        (0.0, 0.0),
        (0.1, 0.1),
        (0.2, 0.26),
        (0.3, 0.3),
        (0.4, 0.4),
    ):
        logger.ingest_raw_episode_snapshot(
            snapshot(timestamp, image_timestamp),
            np.full(_DIM, timestamp, dtype=np.float32),
        )
    logger.set_episode_meta(
        excluded_ranges=[
            {
                "reason": "policy_warmup",
                "start_time": 0.15,
                "end_time": 0.25,
            }
        ]
    )

    assert logger.end_episode()

    episode = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")[0]
    assert episode["length"] == 4
    assert episode["quality"] == "green"
    assert episode["quality_issues"] == []
    assert episode["alignment_image_max_skew_sec"]["cam_high"] == pytest.approx(0.0)


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


def test_collection_saves_multiple_prompts_in_one_dataset_set(tmp_path):
    logger = _collection_logger(
        tmp_path,
        tasks={"cup_set": [("pick up cup", 10), ("place cup", -1)]},
    )
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

    dataset_dir = tmp_path / "cup_set" / "raw"
    assert (dataset_dir / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (dataset_dir / "data" / "chunk-000" / "episode_000001.parquet").exists()
    assert not (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()

    pick_episode, place_episode = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    assert pick_episode["tasks"] == ["pick up cup"]
    assert place_episode["tasks"] == ["place cup"]
    assert _read_jsonl(dataset_dir / "meta" / "tasks.jsonl") == [
        {"task_index": 0, "task": "pick up cup", "required_episodes": 10},
        {"task_index": 1, "task": "place cup", "required_episodes": -1},
    ]
    first_table = pq.read_table(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet")
    second_table = pq.read_table(dataset_dir / "data" / "chunk-000" / "episode_000001.parquet")
    assert set(first_table["task_index"].to_pylist()) == {0}
    assert set(second_table["task_index"].to_pylist()) == {1}
    assert logger.status_snapshot("pick up cup")["dataset_dir"] == str(dataset_dir)
    assert logger.status_snapshot("pick up cup")["episodes"][0]["episode_index"] == 0
    assert logger.status_snapshot("place cup")["dataset_dir"] == str(dataset_dir)
    assert logger.status_snapshot("place cup")["episodes"][1]["episode_index"] == 1


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
