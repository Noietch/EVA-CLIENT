"""HTTP contract tests for history split from the high-frequency status poll."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pytest

from core.app.console.collection_slots import build_collection_slots
from core.app.console.dashboard import _load_dataset_records
from core.app.handlers.recording import load_episode_history
from core.config import _normalize_collection_task_set
from core.utils.collection_paths import dataset_location
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("report_ref", [None, "custom-report.json", "absolute"])
def test_history_refreshes_when_the_actual_replay_report_changes(tmp_path, report_ref):
    meta = tmp_path / "meta"
    meta.mkdir()
    report_path = meta / "render_reports" / "episode_000000.json"
    episode = {
        "episode_index": 0,
        "tasks": ["task"],
        "length": 30,
        "sim_metadata": {"schema_version": 1},
        "qc_verdict": "pass",
    }
    if report_ref is not None:
        report_path = tmp_path / "custom-report.json"
        episode["render_report"] = str(report_path) if report_ref == "absolute" else report_ref
    (meta / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
    pending = load_episode_history(tmp_path)
    assert pending["episodes"][0]["replay_verified"] is None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"task_evaluation": {"success": True, "error": None}}))
    passed = load_episode_history(tmp_path)
    assert passed["episodes"][0]["replay_verified"] is True
    assert passed["version"] != pending["version"]
    report_path.write_text(json.dumps({"task_evaluation": {"success": False, "error": None}}))
    failed = load_episode_history(tmp_path)
    assert failed["episodes"][0]["replay_verified"] is False
    assert failed["version"] != passed["version"]


@pytest.mark.parametrize("verdict", ["unreviewed", "pass"])
def test_dashboard_refreshes_replay_failures_without_rewriting_episode_metadata(tmp_path, verdict):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": 30,
                "sim_metadata": {},
                "qc_verdict": verdict,
            }
        )
        + "\n"
    )
    report_dir = meta / "render_reports"
    report_dir.mkdir()
    report = report_dir / "episode_000000.json"
    for success in [True, False, True]:
        report.write_text(json.dumps({"task_evaluation": {"success": success, "error": None}}))
        _, _, records = _load_dataset_records(tmp_path, "collection")
        assert records[0]["valid"] is (verdict == "pass" or success)


@pytest.mark.parametrize("machine_in_source", [False, True])
def test_machine_findings_reach_history_and_color_unreviewed_slots(tmp_path, machine_in_source):
    config = console_config(collection={"enabled": True})
    config.collection.storage.data_root = str(tmp_path)
    plan_slots = build_collection_slots(config, {}, "cup_set")
    dataset_dir, _ = dataset_location(config, "cup_set")
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    episodes = [
        {
            "episode_index": index,
            "slot_id": plan_slots[index].slot_id,
            "tasks": ["pick up cup"],
            "length": 30,
            "quality": "green",
        }
        for index in range(3)
    ]
    qc = [
        {
            "episode_index": index,
            "qc_verdict": verdict,
            "qc_note": "",
            "qc_auto": False,
            "qc_auto_verdict": auto_verdict,
            "qc_auto_reason": "joint_limit_exceeded" if auto_verdict else "",
            "qc_auto_note": "right_arm.joint4 exceeds its limit" if auto_verdict else "",
        }
        for index, (verdict, auto_verdict) in enumerate(
            [("unreviewed", "fail"), ("pass", "fail"), ("unreviewed", "")]
        )
    ]
    expected_note = qc[0]["qc_auto_note"]
    if machine_in_source:
        for episode, verdict in zip(episodes, qc, strict=True):
            for key in ("qc_auto", "qc_auto_verdict", "qc_auto_reason", "qc_auto_note"):
                episode[key] = verdict.pop(key)
    episode_path = meta / "episodes.jsonl"
    qc_path = meta / "qc.jsonl"
    episode_path.write_text("".join(json.dumps(row) + "\n" for row in episodes))
    qc_path.write_text("".join(json.dumps(row) + "\n" for row in qc))
    originals = (episode_path.read_bytes(), qc_path.read_bytes())
    with serve_console(config) as console:
        console.runtime.episode_logger = _CollectionHistoryLogger(dataset_dir, [])
        history = console.get("/api/episodes?scope=collect&set=cup_set").json
        slots = console.get("/api/collection_slots?dataset=cup_set&all=1").json
    row = history["episodes"][0]
    assert row["qc_verdict"] == "unreviewed"
    assert row["qc_note"] == ""
    assert row["qc_auto_verdict"] == "fail"
    assert row["qc_auto_reason"] == "joint_limit_exceeded"
    assert row["qc_auto_note"] == expected_note
    assert [slot["qc_state"] for slot in slots["slots"][:3]] == ["failed", "passed", "unreviewed"]
    assert slots["counts"]["failed"] == 1
    assert slots["slots"][0]["episode"]["qc_auto_note"] == expected_note
    assert (episode_path.read_bytes(), qc_path.read_bytes()) == originals


def test_collection_history_survives_description_change_with_task_identity(tmp_path):
    plan = tmp_path / "pour"
    plan.mkdir()
    header = "task_id,prompt_en,prompt_zh,total_epsiodes_count,scene_ids,scene_epsiodes_count\n"
    (plan / "tasks.csv").write_text(header + "TASK-1,old wording,旧描述,2,SC-1,2\n")
    config = console_config(collection={"task_set_dir": [str(plan)]})
    _normalize_collection_task_set(config)
    # This is a running process with the old prompt; the scene plan reloads new text.
    (plan / "tasks.csv").write_text(header + "TASK-1,new wording,新描述,2,SC-1,2\n")
    dataset_dir = tmp_path / "raw"
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    rows = [
        {"episode_index": 0, "tasks": ["historical wording"], "task_id": "TASK-1", "length": 5},
        {"episode_index": 1, "tasks": ["old wording"], "task_id": "TASK-OTHER", "length": 5},
    ]
    history_path = meta / "episodes.jsonl"
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = history_path.read_bytes()
    with serve_console(config) as console:
        console.runtime.episode_logger = _CollectionHistoryLogger(
            dataset_dir,
            [{"episode_index": 2, "task": "historical wording", "task_id": "TASK-1"}],
            expected_task="old wording",
            expected_collection_dataset="pour",
        )
        scene_plan = console.get("/api/scene_plan?set=pour").json
        assert scene_plan["tasks"][0]["prompt_zh"] == "新描述"
        assert scene_plan["tasks"][0]["runtime_prompt"] == "old wording"
        query = urlencode({"scope": "collect", "set": "pour", "task": "old wording", "limit": 1})
        response = console.get(f"/api/episodes?{query}")
    assert response.status == 200
    assert response.json["total"] == 1
    assert [row["episode_index"] for row in response.json["episodes"]] == [0]
    assert [row["episode_index"] for row in response.json["queue"]] == [2]
    assert history_path.read_bytes() == original


def _write_history(dataset_dir, count: int) -> None:
    path = dataset_dir / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["task"], "length": index + 1}) + "\n"
            for index in range(count)
        )
    )


class _CollectionHistoryLogger:
    def __init__(
        self,
        dataset_dir: Path,
        queue: list[dict[str, object]],
        *,
        expected_task: str | None = None,
        expected_collection_dataset: str | None = None,
    ) -> None:
        self._dataset_dir = dataset_dir
        self._queue = queue
        self._expected_task = expected_task
        self._expected_collection_dataset = expected_collection_dataset

    def status_snapshot(
        self, task: str, *, include_history: bool = True, collection_dataset: str | None = None
    ) -> dict[str, object]:
        assert include_history is False
        if self._expected_task is not None:
            assert task == self._expected_task
        if self._expected_collection_dataset is not None:
            assert collection_dataset == self._expected_collection_dataset
        return {"dataset_dir": str(self._dataset_dir), "queue": self._queue}


def test_status_excludes_rollout_history_and_history_endpoint_pages(tmp_path):
    config = console_config(
        rollout={
            "storage": {"enabled": True, "log_dir": str(tmp_path)},
        }
    )
    _write_history(tmp_path, 3)

    with serve_console(config) as console:
        status = console.get("/api/status").json
        assert "episodes" not in status["rollout"]
        assert status["rollout"]["completed_episodes"] == 3
        assert status["rollout"]["progress"] == 1.0

        first = console.get("/api/episodes?scope=rollout&limit=2").json
        assert first["ok"] is True
        assert first["scope"] == "rollout"
        assert [row["episode_index"] for row in first["episodes"]] == [0, 1]
        assert first["total"] == 3
        assert first["next_since"] == 2
        assert first["has_more"] is True
        assert first["cursor"]
        assert first["reset"] is False

        second_query = urlencode(
            {
                "scope": "rollout",
                "since": first["next_since"],
                "limit": 2,
                "cursor": first["cursor"],
            }
        )
        second = console.get(f"/api/episodes?{second_query}").json
        assert [row["episode_index"] for row in second["episodes"]] == [2]
        assert second["has_more"] is False
        assert second["reset"] is False
