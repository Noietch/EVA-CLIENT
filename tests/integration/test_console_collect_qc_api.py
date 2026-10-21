"""Saved collection QC uses dataset/episode identity across prompt edits."""

import json
from pathlib import Path

import pytest

from core.app.console.collection_slots import build_collection_slots
from core.recorder.episode import EpisodeLogger
from core.utils.collection_paths import dataset_location
from core.utils.lerobot import LeRobotDatasetIO
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def test_qc_survives_renamed_task_and_preserves_dataset_boundary(tmp_path):
    config = console_config(collection={"enabled": True})
    config.collection.schema.columns = {"qpos": "observation.qpos", "action_qpos": "action"}
    root = tmp_path / "collection"
    dataset_dir = (
        Path(config.collection.storage.data_root)
        / "datasets/real_robot"
        / config.robot.type
        / "cup_set"
    )
    meta = dataset_dir / "meta" / "episodes.jsonl"
    ledger = dataset_dir / "meta" / "qc.jsonl"
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
            # The recording row keeps its capture metadata; the verdict lives in
            # the dataset's QC ledger.
            assert json.loads(meta.read_text()) == original
            saved = meta.read_bytes()
            row = json.loads(ledger.read_text())
            assert (row["episode_index"], row["qc_verdict"]) == (44, "fail")
            assert row["qc_note"] == "retake"
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
            assert json.loads(ledger.read_text())["qc_verdict"] == "pass"
            response = console.post("/api/collect_qc_mark", {**body, "verdict": "unreviewed"})
            assert response.status == 200
            assert json.loads(ledger.read_text())["qc_verdict"] == "unreviewed"
            response = console.post(
                "/api/collect_qc_mark", {**body, "verdict": "", "note": "keep review pending"}
            )
            assert response.status == 200
            assert json.loads(ledger.read_text())["qc_verdict"] == "unreviewed"
        finally:
            logger.finalize()


@pytest.mark.parametrize("replay_success", [None, False, True])
def test_sim_human_pass_survives_console_refresh_and_preserves_findings(tmp_path, replay_success):
    config = console_config(collection={"enabled": True})
    config.collection.storage.data_root = str(tmp_path)
    config.collection.schema.columns = {"qpos": "observation.qpos", "action_qpos": "action"}
    slot_id = build_collection_slots(config, {}, "cup_set")[0].slot_id
    dataset_dir, _ = dataset_location(config, "cup_set")
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    episode_path = meta / "episodes.jsonl"
    episode_path.write_text(
        json.dumps({
            "episode_index": 44,
            "tasks": ["pick up cup"],
            "slot_id": slot_id,
            "length": 30,
            "quality": "red",
            "sim_metadata": {"schema_version": 1},
        }) + "\n"
    )
    original = episode_path.read_bytes()
    report_path = meta / "render_reports/episode_000044.json"
    replay_failure = "第1次回放有问题：任务未成功"
    if replay_success is not None:
        report_path.parent.mkdir()
        report_path.write_text(
            json.dumps({"task_evaluation": {
                "success": replay_success,
                "error": {"type": "ReplayFailed", "message": replay_failure}
                if replay_success is False else None,
            }})
        )
    io = LeRobotDatasetIO(dataset_dir)
    assert io.mark_qc(44, "fail", "joint4 exceeds its limit", "joint_limit_exceeded", auto=True)
    with serve_console(config) as console:
        logger = EpisodeLogger(
            tmp_path / "collection",
            console.runtime.robot,
            fps=30,
            dataset_keys=config.transport.dataset_keys,
            collection=config.collection,
        )
        console.runtime.episode_logger = logger
        try:
            body = {"dataset": "cup_set", "episode": "44", "verdict": "pass", "note": "reviewed"}
            response = console.post("/api/collect_qc_mark", body)
            assert response.status == 200
            slots = console.get("/api/collection_slots?dataset=cup_set&all=1").json
            slot = next(row for row in slots["slots"] if row["slot_id"] == slot_id)
            assert slot["qc_state"] == "passed"
            assert slots["counts"]["passed"] == 1
            history = console.get(
                "/api/episodes?scope=collect&set=cup_set&task=pick%20up%20cup"
            ).json
            saved = history["episodes"][0]
            assert saved["qc_verdict"] == "pass"
            assert saved["replay_verified"] is replay_success
            expected_reason = (
                "replay_report_pending" if replay_success is None
                else replay_failure if replay_success is False else ""
            )
            assert saved["replay_qc_reason"] == expected_reason
            assert saved["qc_auto_verdict"] == "fail"
            assert saved["qc_auto_reason"] == "joint_limit_exceeded"
            assert saved["qc_auto_note"] == "joint4 exceeds its limit"
            assert saved["qc_note"] == "reviewed"
            if replay_success is False:
                report = json.loads(report_path.read_text())
                report["task_evaluation"]["error"]["message"] = "更新后的回放诊断"
                report_path.write_text(json.dumps(report))
                refreshed = console.get(
                    "/api/episodes?scope=collect&set=cup_set&task=pick%20up%20cup"
                ).json["episodes"][0]
                assert refreshed["replay_qc_reason"] == "更新后的回放诊断"
                assert refreshed["qc_note"] == "reviewed"
            assert episode_path.read_bytes() == original
            response = console.post("/api/collect_qc_mark", {**body, "verdict": "unreviewed"})
            assert response.status == 200
            slots = console.get("/api/collection_slots?dataset=cup_set&all=1").json
            slot = next(row for row in slots["slots"] if row["slot_id"] == slot_id)
            assert slot["qc_state"] == "failed"
        finally:
            logger.finalize()
