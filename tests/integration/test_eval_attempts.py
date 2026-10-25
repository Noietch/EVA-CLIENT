"""Evaluation recordings preserve attempts and restore their scene identity."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.app.console.episode_preview import read_result_rows
from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger
from core.types import Observation
from robots.base import ActuatorGroup, ObservationSchema, Robot

pytestmark = pytest.mark.integration


@pytest.fixture(params=[False, True], ids=["sync-save", "async-save"])
def recorder(tmp_path, request):
    robot = Robot(
        name="eval_arm",
        actuator_groups=(ActuatorGroup("arm", 1, ("j0",)),),
        initial_qpos=np.zeros(1, dtype=np.float32),
        observation_schema=ObservationSchema(cameras=(), state_composition=("arm",)),
    )
    logger = EpisodeLogger(
        tmp_path,
        robot,
        fps=10,
        dataset_keys=ConfigDict(
            state_key="observations.state.qpos",
            eef_key="observations.state.eef",
            video_keys={},
        ),
        eval_mode=True,
        async_save=request.param,
    )
    yield logger
    logger.wait_for_saves(fail_on_error=True)


def _record_attempt(recorder, clip_id, scene_id, value):
    recorder.start_episode("pick up the cup")
    recorder.set_episode_meta(
        task_set="eval-cups",
        task_id="PICK-CUP",
        scene_id=scene_id,
        prompt="pick up the cup",
        trial=1,
        clip_id=clip_id,
        sim_metadata={"scene": {"scene_id": scene_id}},
        scene_map={"scene_id": scene_id, "placements": [{"object_id": "CUP"}]},
    )
    state = np.array([value], dtype=np.float32)
    for frame in range(3):
        recorder.record_step(
            Observation(state_qpos=state, images={}),
            state,
            timestamp=100.0 + frame / 10,
        )
    assert recorder.end_episode()


def _rows(root):
    return [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]


def test_different_scenes_preserve_both_recordings(recorder, tmp_path):
    _record_attempt(recorder, "attempt-scene-1", "SC-1", 1.0)
    _record_attempt(recorder, "attempt-scene-2", "SC-2", 2.0)
    recorder.wait_for_saves(fail_on_error=True)

    rows = _rows(tmp_path)
    assert [(row["episode_index"], row["clip_id"], row["scene_id"]) for row in rows] == [
        (0, "attempt-scene-1", "SC-1"),
        (1, "attempt-scene-2", "SC-2"),
    ]
    for index, expected in enumerate((1.0, 2.0)):
        table = pq.read_table(tmp_path / f"data/chunk-000/episode_{index:06d}.parquet")
        assert table["action.qpos"].to_pylist() == [[expected]] * 3
        assert table["index"].to_pylist() == list(range(index * 3, index * 3 + 3))


def test_rerun_keeps_attempts_and_scores_only_the_selected_clip(recorder, tmp_path):
    _record_attempt(recorder, "first-attempt", "SC-1", 1.0)
    _record_attempt(recorder, "second-attempt", "SC-1", 2.0)
    recorder.wait_for_saves(fail_on_error=True)

    assert recorder.patch_episode_meta_by_clip("first-attempt", score=1, max_score=2) == 0
    assert recorder.patch_episode_meta_by_clip("second-attempt", score=2, max_score=2) == 1
    assert recorder.patch_episode_meta_by_clip("unknown-attempt", score=0) is None
    assert [(row["clip_id"], row["score"]) for row in _rows(tmp_path)] == [
        ("first-attempt", 1),
        ("second-attempt", 2),
    ]
    assert [row["clip_id"] for row in read_result_rows(tmp_path)] == [
        "first-attempt",
        "second-attempt",
    ]
    for index, expected in enumerate((1.0, 2.0)):
        table = pq.read_table(tmp_path / f"data/chunk-000/episode_{index:06d}.parquet")
        assert table["action.qpos"].to_pylist() == [[expected]] * 3


def test_result_reload_preserves_scene_identity_and_snapshot(recorder, tmp_path):
    _record_attempt(recorder, "scene-attempt", "SC-1", 1.0)
    recorder.wait_for_saves(fail_on_error=True)

    recorded = _rows(tmp_path)[0]
    result = read_result_rows(tmp_path)[0]
    for field in (
        "task_set",
        "task_id",
        "scene_id",
        "prompt",
        "trial",
        "sim_metadata",
        "scene_map",
    ):
        assert result[field] == recorded[field]
    assert recorder.patch_episode_meta_by_clip("scene-attempt", score=2) == 0
    refreshed = read_result_rows(tmp_path)[0]
    assert refreshed["score"] == 2
    assert refreshed["scene_map"] == recorded["scene_map"]


def test_scoring_does_not_drop_an_attempt_saved_concurrently(recorder, tmp_path, monkeypatch):
    _record_attempt(recorder, "first-attempt", "SC-1", 1.0)
    assert recorder.wait_for_saves(timeout=5, fail_on_error=True)
    save_ready = threading.Event()
    score_ready = threading.Event()
    release_save = threading.Event()
    release_score = threading.Event()
    appended = threading.Event()
    write_videos = recorder._write_raw_sample_videos
    write_jsonl = recorder._write_jsonl_atomic
    append_row = recorder._append_episode_dict_row_locked

    def pause_before_append(*args):
        write_videos(*args)
        save_ready.set()
        assert release_save.wait(timeout=5)

    def pause_score_commit(path, rows):
        if path.name == "episodes.jsonl":
            score_ready.set()
            assert release_score.wait(timeout=5)
        write_jsonl(path, rows)

    def track_append(row, dataset_dir=None):
        appended.set()
        append_row(row, dataset_dir)

    monkeypatch.setattr(recorder, "_write_raw_sample_videos", pause_before_append)
    monkeypatch.setattr(recorder, "_write_jsonl_atomic", pause_score_commit)
    monkeypatch.setattr(recorder, "_append_episode_dict_row_locked", track_append)
    with ThreadPoolExecutor(max_workers=2) as workers:
        saving = workers.submit(_record_attempt, recorder, "second-attempt", "SC-2", 2.0)
        try:
            assert save_ready.wait(timeout=5)
            scoring = workers.submit(recorder.patch_episode_meta_by_clip, "first-attempt", score=2)
            assert score_ready.wait(timeout=5)
            release_save.set()
            append_during_score = appended.wait(timeout=0.2)
        finally:
            release_save.set()
            release_score.set()
        assert scoring.result(timeout=5) == 0
        saving.result(timeout=5)
        assert recorder.wait_for_saves(timeout=5, fail_on_error=True)

    assert not append_during_score
    rows = _rows(tmp_path)
    assert [(row["clip_id"], row.get("score")) for row in rows] == [
        ("first-attempt", 2),
        ("second-attempt", None),
    ]
    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["eval"]["scored_episodes"] == 1
