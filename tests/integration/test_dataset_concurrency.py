"""Two stations writing one collection dataset directory at the same time.

The stations are separate processes over a shared dataset, so these tests fork real
processes and assert the dataset keeps both sides' episodes, task rows and QC
verdicts instead of the last writer winning.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.types import CollectionRawBatch, CollectionRawSample, RawCollectionSnapshot
from core.utils.lerobot import LeRobotDatasetIO
from tests.integration.test_episode_logger import (
    _collection_logger,
    _collection_task_dir,
    _read_jsonl,
)

pytestmark = pytest.mark.integration

_DIM = 6
_SHARED_DATASET = "shared"
_STATION_START_SEC = {"a": 1.0, "b": 5.0}


def _record_episode(logger, task: str, timestamp: float, index: int) -> None:
    """Record one short collection episode at ``timestamp`` on the 10 fps grid."""
    frame = np.full((8, 8, 3), index, dtype=np.uint8)
    value = np.full(_DIM, float(index), dtype=np.float32)
    batch = CollectionRawBatch(
        images={"cam_high": [CollectionRawSample(timestamp, frame)]},
        vectors={
            "state_qpos": [CollectionRawSample(timestamp, value)],
            "action_qpos": [CollectionRawSample(timestamp, value)],
        },
    )
    logger.start_episode(task, collection_dataset=_SHARED_DATASET)
    logger.ingest_collection_snapshot(
        RawCollectionSnapshot(timestamp=timestamp, decode_raw=lambda: batch)
    )
    assert logger.end_episode()


def _record_station(root: Path, task: str, count: int, barrier) -> None:
    """One station's session: ``count`` episodes, started together with its peer."""
    logger = _collection_logger(root, shared_dataset=True)
    barrier.wait(30.0)
    start = _STATION_START_SEC[task]
    for index in range(count):
        _record_episode(logger, task, start + index * 0.3, index)


def _index_ranges(root) -> dict[int, tuple[int, int]]:
    """``{episode_index: (first, last)}`` of the dataset's parquet frame indices."""
    dataset_dir = _collection_task_dir(root, _SHARED_DATASET)
    ranges = {}
    for path in sorted((dataset_dir / "data" / "chunk-000").glob("episode_*.parquet")):
        values = pq.read_table(path, columns=["index"]).column("index").to_pylist()
        ranges[int(path.stem.rsplit("_", 1)[1])] = (min(values), max(values))
    return ranges


def test_two_processes_recording_one_dataset_keep_every_episode(tmp_path) -> None:
    context = mp.get_context("fork")
    barrier = context.Barrier(3)
    stations = [
        context.Process(target=_record_station, args=(tmp_path, task, 3, barrier))
        for task in ("a", "b")
    ]
    for station in stations:
        station.start()
    barrier.wait(60.0)
    for station in stations:
        station.join(120.0)
        assert station.exitcode == 0

    dataset_dir = _collection_task_dir(tmp_path, _SHARED_DATASET)
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    indices = sorted(int(row["episode_index"]) for row in episodes)
    assert len(indices) == len(set(indices)) == 6
    assert {row["tasks"][0] for row in episodes} == {"a", "b"}

    task_rows = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    tasks = {row["task"]: int(row["task_index"]) for row in task_rows}
    assert set(tasks) == {"a", "b"}
    assert len(set(tasks.values())) == 2

    qc_rows = _read_jsonl(dataset_dir / "meta" / "qc.jsonl")
    assert {int(row["episode_index"]) for row in qc_rows} == set(indices)

    frame_ranges = _index_ranges(tmp_path)
    assert sorted(frame_ranges) == indices
    ordered = sorted(frame_ranges.values())
    for (_, previous_end), (next_start, _) in pairwise(ordered):
        assert next_start > previous_end
    starts = {start for start, _ in ordered}
    for row in episodes:
        assert row["global_index_start"] in starts


def test_a_second_logger_places_frames_after_the_existing_episode(tmp_path) -> None:
    first = _collection_logger(tmp_path, shared_dataset=True)
    _record_episode(first, "a", 1.0, 0)
    second = _collection_logger(tmp_path, shared_dataset=True)
    _record_episode(second, "a", 5.0, 1)
    _record_episode(first, "a", 9.0, 2)

    dataset_dir = _collection_task_dir(tmp_path, _SHARED_DATASET)
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    assert sorted(int(row["episode_index"]) for row in episodes) == [0, 1, 2]
    ranges = _index_ranges(tmp_path)
    assert len(ranges) == 3
    assert min(start for start, _ in ranges.values()) == 0
    assert max(end for _, end in ranges.values()) == 2
    assert len({index for start, end in ranges.values() for index in range(start, end + 1)}) == 3


def _mark_verdict(root: Path, episode: int, verdict: str, barrier) -> None:
    dataset_dir = _collection_task_dir(root, _SHARED_DATASET)
    barrier.wait(30.0)
    assert LeRobotDatasetIO(dataset_dir).mark_qc(episode, verdict, f"{verdict} on station")


def test_two_processes_reviewing_one_dataset_keep_both_verdicts(tmp_path) -> None:
    dataset_dir = _collection_task_dir(tmp_path, _SHARED_DATASET)
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["a"], "length": 1}) + "\n"
            for index in range(2)
        )
    )

    context = mp.get_context("fork")
    barrier = context.Barrier(3)
    reviewers = [
        context.Process(target=_mark_verdict, args=(tmp_path, index, verdict, barrier))
        for index, verdict in ((0, "pass"), (1, "fail"))
    ]
    for reviewer in reviewers:
        reviewer.start()
    barrier.wait(60.0)
    for reviewer in reviewers:
        reviewer.join(60.0)
        assert reviewer.exitcode == 0

    verdicts = {
        int(row["episode_index"]): row["qc_verdict"]
        for row in _read_jsonl(dataset_dir / "meta" / "qc.jsonl")
    }
    assert verdicts == {0: "pass", 1: "fail"}


def _annotate(root: Path, episode: int, annotation: str, barrier) -> None:
    dataset_dir = _collection_task_dir(root, _SHARED_DATASET)
    barrier.wait(30.0)
    assert LeRobotDatasetIO(dataset_dir).annotate_language(episode, annotation)


def test_two_processes_annotating_one_dataset_keep_both_annotations(tmp_path) -> None:
    dataset_dir = _collection_task_dir(tmp_path, _SHARED_DATASET)
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["a"], "length": 1}) + "\n"
            for index in range(2)
        )
    )

    context = mp.get_context("fork")
    barrier = context.Barrier(3)
    annotators = [
        context.Process(target=_annotate, args=(tmp_path, index, text, barrier))
        for index, text in ((0, "grasp the cup"), (1, "place the cup"))
    ]
    for annotator in annotators:
        annotator.start()
    barrier.wait(60.0)
    for annotator in annotators:
        annotator.join(60.0)
        assert annotator.exitcode == 0

    rows = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    annotations = {int(row["episode_index"]): row["language_annotation"] for row in rows}
    assert annotations == {0: "grasp the cup", 1: "place the cup"}
