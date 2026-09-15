"""Saved collection QC uses dataset/episode identity across prompt edits."""

import json

import pytest

from core.recorder.episode import EpisodeLogger
from tests.integration.web._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def test_qc_survives_renamed_task_and_preserves_dataset_boundary(tmp_path):
    config = console_config(collection={"enabled": True})
    config.collection.schema.columns = {"qpos": "observation.qpos", "action_qpos": "action"}
    root = tmp_path / "collection"
    meta = root / "cup_set" / "raw" / "meta" / "episodes.jsonl"
    meta.parent.mkdir(parents=True)
    original = {
        "episode_index": 44,
        "tasks": ["old zipper closed wording"],
        "task_id": "TASK-071",
        "length": 20,
        "quality": "green",
    }
    meta.write_text(json.dumps(original) + "\n")
    with serve_console(config) as console:
        logger = EpisodeLogger(
            root,
            console.runtime.robot,
            fps=30,
            dataset_keys=config.transport.dataset_keys,
            collection=config.collection,
        )
        console.runtime.episode_logger = logger
        try:
            body = {
                "dataset": "cup_set",
                "task": original["tasks"][0],
                "episode": "44",
                "verdict": "fail",
                "note": "retake",
                "dataset_dir": str(tmp_path / "untrusted-path"),
            }
            response = console.post("/api/collect_qc_mark", body)
            assert response.status == 200
            assert json.loads(meta.read_text()) == {
                **original,
                "qc_verdict": "fail",
                "qc_note": "retake",
            }
            saved = meta.read_bytes()
            for overrides in (
                {"dataset": "missing"},
                {"dataset": "pouring_set"},
                {"dataset": ""},
                {"episode": "999"},
            ):
                response = console.post("/api/collect_qc_mark", {**body, **overrides})
                assert response.status == 409
                assert meta.read_bytes() == saved
            response = console.post("/api/collect_qc_mark", {**body, "verdict": "pass"})
            assert response.status == 200
            assert json.loads(meta.read_text())["qc_verdict"] == "pass"
        finally:
            logger.finalize()
