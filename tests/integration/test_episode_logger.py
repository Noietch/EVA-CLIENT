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
from core.utils.lerobot import LeRobotDatasetIO
from core.utils.qc import qc_state, replay_validation
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


def test_wait_for_saves_checks_only_the_requested_job(tmp_path):
    logger = _logger(tmp_path, async_save=True)
    failed_history = type("Job", (), {"status": "failed"})()
    current = type("Job", (), {"status": "saved"})()
    logger._save_jobs = [failed_history]

    assert logger.wait_for_saves(fail_on_error=True, job=current)


def _collection_logger(
    log_dir,
    *,
    async_save: bool = False,
    shared_dataset: bool = False,
) -> EpisodeLogger:
    """A collection logger over ``log_dir``.

    With ``shared_dataset`` the logger is bound to a mounted task set, so every
    prompt records into one dataset directory instead of one directory per prompt —
    the way two stations collecting the same task set share their output.
    """
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
            task_set_dir=str(log_dir / "task_sets") if shared_dataset else "",
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
    assert [(row["episode_index"], row["qc_verdict"]) for row in ledger] == [
        (0, "pass"),
        (1, "unreviewed"),
    ]


def test_a_retake_replaces_the_slots_previous_episode(tmp_path):
    """A second take for one slot overwrites that episode instead of appending."""
    logger = _collection_logger(tmp_path)
    dataset_dir = _collection_task_dir(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    slot = "TASK-1:SC-1:0"

    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, state=qpos))
    assert logger.end_episode()
    assert logger.mark_collection_qc("t", 0, "fail", "first take")

    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    for timestamp in (0.1, 0.2, 0.3):
        logger.ingest_collection_snapshot(_collection_raw_snapshot(timestamp, state=qpos))
    assert logger.end_episode()

    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    assert [(row["episode_index"], row["slot_id"]) for row in episodes] == [(0, slot)]
    assert episodes[0]["length"] == 3
    stats = _read_jsonl(dataset_dir / "meta" / "episodes_stats.jsonl")
    assert [row["episode_index"] for row in stats] == [0]
    saved = pq.read_table(dataset_dir / "data/chunk-000/episode_000000.parquet")
    assert saved.num_rows == 3
    # The replaced take's verdict dies with it: the new recording is unreviewed.
    ledger = _read_jsonl(dataset_dir / "meta" / "qc.jsonl")
    assert [(row["episode_index"], row["qc_verdict"]) for row in ledger] == [(0, "unreviewed")]


def test_sim_retake_invalidates_the_previous_replay_report(tmp_path):
    logger = _collection_logger(tmp_path)
    dataset_dir = _collection_task_dir(tmp_path)
    slot = "TASK-1:SC-1:0"
    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot, sim_metadata={"schema_version": 1})
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0))
    assert logger.end_episode()
    logger._collection_qc.wait(timeout=10)

    report_paths = (
        dataset_dir / "meta/render_reports/episode_000000.json",
        dataset_dir / "render/reports/episode_000000.json",
    )
    for report_path in report_paths:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"task_evaluation": {"success": True, "error": None}}),
            encoding="utf-8",
        )
    previous = _read_jsonl(dataset_dir / "meta/episodes.jsonl")[0]
    assert replay_validation(dataset_dir, previous) == (True, "")

    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot, sim_metadata={"schema_version": 1})
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1))
    assert logger.end_episode()
    logger._collection_qc.wait(timeout=10)
    assert logger.mark_collection_qc("t", 0, "pass", "current take")
    current = _read_jsonl(dataset_dir / "meta/episodes.jsonl")[0]
    verified, reason = replay_validation(dataset_dir, current)
    assert all(not path.exists() for path in report_paths)
    assert (verified, reason) == (None, "replay_report_pending")
    assert qc_state("pass", "green", verified, True) == "passed"
    logger.finalize()


@pytest.mark.parametrize("secondary_finding", [False, True])
def test_retake_enters_new_qc_without_inheriting_machine_failure(
    tmp_path, monkeypatch, secondary_finding
):
    from tools.datasets.auto_qc import EpisodeAutoQc

    logger = _collection_logger(tmp_path)
    dataset_dir = _collection_task_dir(tmp_path)
    slot = "TASK-1:SC-1:0"
    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0))
    assert logger.end_episode()
    logger._collection_qc.wait(timeout=10)
    io = LeRobotDatasetIO(dataset_dir)
    if secondary_finding:
        assert io.mark_qc(0, "fail", "old human review")
    assert io.mark_qc(0, "fail", "old machine finding", "joint_limit_exceeded", auto=True)

    checking = threading.Event()
    release = threading.Event()
    check = EpisodeAutoQc.check

    def paused_check(self, row, fence=None):
        checking.set()
        assert release.wait(10)
        return check(self, row, fence)

    monkeypatch.setattr(EpisodeAutoQc, "check", paused_check)
    try:
        logger.start_episode("t")
        logger.set_episode_meta(slot_id=slot)
        for index, timestamp in enumerate((0.1, 0.2, 0.3)):
            logger.ingest_collection_snapshot(
                _collection_raw_snapshot(timestamp, image=np.full((8, 8, 3), index * 60, np.uint8))
            )
        assert logger.end_episode()
        assert checking.wait(10)
        ledger = _read_jsonl(dataset_dir / "meta/qc.jsonl")[0]
        assert ledger["qc_verdict"] == "unreviewed"
        assert ledger["qc_note"] == ledger["qc_reason"] == ""
        assert not any(key.startswith("qc_auto_") for key in ledger)
        assert qc_state(ledger["qc_verdict"], "green") == "unreviewed"
    finally:
        release.set()
        logger._collection_qc.wait(timeout=10)
    ledger = _read_jsonl(dataset_dir / "meta/qc.jsonl")[0]
    assert ledger["qc_verdict"] == "unreviewed"
    assert ledger["qc_note"] == ledger["qc_reason"] == ""
    logger.finalize()


def test_older_qc_cannot_write_its_finding_into_a_new_retake(tmp_path, monkeypatch):
    from tools.datasets.auto_qc import EpisodeAutoQc

    logger = _collection_logger(tmp_path)
    dataset_dir = _collection_task_dir(tmp_path)
    slot = "TASK-1:SC-1:0"
    checked = threading.Event()
    release = threading.Event()
    check = EpisodeAutoQc.check
    first = True

    def delayed_result(self, row, fence=None):
        nonlocal first
        issues = check(self, row, fence)
        if first:
            first = False
            assert issues  # The first take has a genuinely frozen camera.
            checked.set()
            assert release.wait(10)
        return issues

    monkeypatch.setattr(EpisodeAutoQc, "check", delayed_result)
    try:
        logger.start_episode("t")
        logger.set_episode_meta(slot_id=slot)
        for timestamp in (0.0, 0.1, 0.2):
            logger.ingest_collection_snapshot(_collection_raw_snapshot(timestamp))
        assert logger.end_episode()
        assert checked.wait(10)
        logger.start_episode("t")
        logger.set_episode_meta(slot_id=slot)
        for index, timestamp in enumerate((0.3, 0.4, 0.5)):
            logger.ingest_collection_snapshot(
                _collection_raw_snapshot(timestamp, image=np.full((8, 8, 3), index * 60, np.uint8))
            )
        assert logger.end_episode()
    finally:
        release.set()
        logger._collection_qc.wait(timeout=10)
    ledger = _read_jsonl(dataset_dir / "meta/qc.jsonl")[0]
    assert ledger["qc_verdict"] == "unreviewed"
    assert ledger["qc_note"] == ledger["qc_reason"] == ""
    assert _read_jsonl(dataset_dir / "meta/episodes.jsonl")[0]["quality"] == "green"
    logger.finalize()


def test_retake_removes_duplicate_episode_and_stats_rows(tmp_path):
    logger = _collection_logger(tmp_path)
    dataset_dir = _collection_task_dir(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    slot = "TASK-1:SC-1:0"

    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, state=qpos))
    assert logger.end_episode()

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    episodes_path.write_text(
        episodes_path.read_text() + episodes_path.read_text(), encoding="utf-8"
    )
    stats_path.write_text(stats_path.read_text() + stats_path.read_text(), encoding="utf-8")

    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1, state=qpos))
    assert logger.end_episode()

    episodes = _read_jsonl(episodes_path)
    stats = _read_jsonl(stats_path)
    assert [row["episode_index"] for row in episodes] == [0]
    assert [row["episode_index"] for row in stats] == [0]


def test_qc_drop_canonicalizes_duplicate_rows(tmp_path):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    for timestamp in (0.0, 0.1, 0.2):
        logger.start_episode("t")
        logger.ingest_collection_snapshot(_collection_raw_snapshot(timestamp, state=qpos))
        assert logger.end_episode()
    dataset_dir = _collection_task_dir(tmp_path)
    qc_path = dataset_dir / "meta" / "qc.jsonl"
    rows = _read_jsonl(qc_path)
    qc_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [rows[2], rows[0], rows[1], rows[1]]),
        encoding="utf-8",
    )

    assert LeRobotDatasetIO(dataset_dir).drop_qc(1)
    rows = _read_jsonl(qc_path)
    assert [row["episode_index"] for row in rows] == [0, 2]


def test_collection_save_rolls_back_when_qc_write_fails(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path)
    qpos = np.zeros(_DIM, dtype=np.float32)
    slot = "TASK-1:SC-1:0"
    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0, state=qpos))
    assert logger.end_episode()
    dataset_dir = _collection_task_dir(tmp_path)
    old_parquet = (dataset_dir / "data/chunk-000/episode_000000.parquet").read_bytes()
    logger._collection_qc.wait(timeout=10)
    report_paths = (
        dataset_dir / "meta/render_reports/episode_000000.json",
        dataset_dir / "render/reports/episode_000000.json",
    )
    old_reports = {}
    for report_path in report_paths:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        old_reports[report_path] = json.dumps(
            {"task_evaluation": {"success": True, "error": None}}
        )
        report_path.write_text(old_reports[report_path], encoding="utf-8")

    original = LeRobotDatasetIO.reset_qc

    def fail_qc(self, *args, **kwargs):
        if args and int(args[0]) == 0:
            raise OSError("qc disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LeRobotDatasetIO, "reset_qc", fail_qc)
    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    for timestamp in (0.1, 0.2):
        logger.ingest_collection_snapshot(_collection_raw_snapshot(timestamp, state=qpos))
    with pytest.raises(OSError, match="qc disk full"):
        logger.end_episode()

    episodes = _read_jsonl(dataset_dir / "meta/episodes.jsonl")
    assert [(row["episode_index"], row["length"]) for row in episodes] == [(0, 1)]
    assert (dataset_dir / "data/chunk-000/episode_000000.parquet").read_bytes() == old_parquet
    assert {
        path: path.read_text(encoding="utf-8") for path in report_paths
    } == old_reports
    logger.finalize()


@pytest.mark.parametrize("failed_name", ["episodes_stats.jsonl", "episode_000000.parquet"])
def test_failed_backup_preserves_every_existing_artifact(tmp_path, monkeypatch, failed_name):
    logger = _collection_logger(tmp_path)
    slot = "TASK-1:SC-1:0"
    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0))
    assert logger.end_episode()
    logger._collection_qc.wait(timeout=10)

    dataset_dir = _collection_task_dir(tmp_path)
    paths = [
        dataset_dir / "data/chunk-000/episode_000000.parquet",
        *(
            dataset_dir / "meta" / name
            for name in (
                "episodes.jsonl",
                "episodes_stats.jsonl",
                "tasks.jsonl",
                "qc.jsonl",
                "info.json",
                "stats.json",
            )
        ),
    ]
    before = {path: path.read_bytes() for path in paths}
    copy = episode_module.shutil.copy2

    def fail_backup(source, destination):
        if source.name == failed_name:
            raise OSError("backup disk full")
        return copy(source, destination)

    monkeypatch.setattr(episode_module.shutil, "copy2", fail_backup)
    logger.start_episode("t")
    logger.set_episode_meta(slot_id=slot)
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1))
    with pytest.raises(OSError, match="backup disk full"):
        logger.end_episode()

    assert {path: path.read_bytes() for path in paths} == before
    logger.finalize()


def test_failed_parquet_staging_leaves_no_empty_episode(tmp_path, monkeypatch):
    logger = _collection_logger(tmp_path)

    def fail_write(_table, _path):
        raise OSError("parquet disk full")

    with monkeypatch.context() as patch:
        patch.setattr(episode_module.pq, "write_table", fail_write)
        logger.start_episode("t")
        logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0))
        with pytest.raises(OSError, match="parquet disk full"):
            logger.end_episode()

    dataset_dir = _collection_task_dir(tmp_path)
    assert not list((dataset_dir / "data/chunk-000").glob("episode_*.parquet"))
    logger.start_episode("t")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.1))
    assert logger.end_episode()
    rows = _read_jsonl(dataset_dir / "meta/episodes.jsonl")
    assert len(rows) == 1
    saved = pq.ParquetFile(logger._parquet_path(rows[0]["episode_index"], dataset_dir))
    assert saved.metadata.num_rows == 1
    logger.finalize()


def test_episode_metadata_cannot_be_reassigned_to_another_slot(tmp_path):
    logger = _collection_logger(tmp_path)
    logger.start_episode("t")
    logger.set_episode_meta(slot_id="TASK-1:SC-1:0")
    logger.ingest_collection_snapshot(_collection_raw_snapshot(0.0))
    assert logger.end_episode()
    logger._collection_qc.wait(timeout=10)
    dataset_dir = _collection_task_dir(tmp_path)
    path = dataset_dir / "meta/episodes.jsonl"
    before = path.read_bytes()
    row = {**_read_jsonl(path)[0], "slot_id": "TASK-2:SC-2:0"}
    with pytest.raises(ValueError, match="belongs to slot"):
        logger._upsert_episode_dict_row_locked(row, dataset_dir)
    assert path.read_bytes() == before
    logger.finalize()


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
