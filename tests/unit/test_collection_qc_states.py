"""QC presentation stays independent of capture workflow and stale queue metadata."""

import json

import pytest

from core.app.console.collection_slots import (
    CollectionSlotState,
    build_collection_slots,
    collection_slot_status,
)
from core.config import ConfigDict
from core.utils.qc import (
    qc_state,
    replay_report_path,
    replay_report_signature,
    replay_validation,
)
from tools.datasets.collection import PlanCatalog

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("verdict", "auto_verdict", "expected"),
    [
        ("", "fail", "failed"),
        ("unreviewed", "fail", "failed"),
        ("pass", "fail", "passed"),
        ("fail", "pass", "failed"),
        ("unreviewed", "pass", "unreviewed"),
        ("unreviewed", "", "unreviewed"),
    ],
)
def test_machine_failure_marks_unreviewed_slots_in_both_apps(verdict, auto_verdict, expected):
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 1)]})), {}, "set"
    )
    episode = {
        "episode_index": 49,
        "slot_id": slots[0].slot_id,
        "status": "saved",
        "quality": "green",
        "qc_verdict": verdict,
        "qc_auto_verdict": auto_verdict,
        "qc_auto_reason": "joint_limit_exceeded",
        "qc_auto_note": "right_arm.joint4 exceeds its limit",
    }
    rows, _, counts = collection_slot_status(slots, [episode], [], CollectionSlotState([]))
    assert rows[0]["qc_state"] == expected
    assert counts[expected] == 1
    dataset_slot = PlanCatalog._slot(slots[0].slot_id, "DEFAULT", 0, 1, episode)
    assert dataset_slot["qc_state"] == expected
    assert dataset_slot["episode"]["qc_auto_verdict"] == auto_verdict
    assert qc_state(verdict, "green", qc_auto_verdict=auto_verdict) == expected


def test_four_qc_states_match_dataset_tools_and_ignore_stale_queue():
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 5)]})), {}, "set"
    )
    episodes = [
        {
            "episode_index": i,
            "slot_id": slots[i].slot_id,
            "status": "saved",
            "quality": quality,
            "qc_verdict": verdict,
        }
        for i, (quality, verdict) in enumerate(
            [
                ("green", ""),
                ("green", "pass"),
                ("green", "fail"),
                ("red", "pass"),
            ]
        )
    ]
    stale_queue = [{**episodes[2], "qc_verdict": ""}]
    rows, active, counts = collection_slot_status(
        slots, episodes, stale_queue, CollectionSlotState([slots[4].slot_id])
    )
    assert [row["qc_state"] for row in rows] == [
        "unreviewed",
        "passed",
        "failed",
        "passed",
        "pending",
    ]
    assert {key: counts[key] for key in ("unreviewed", "passed", "failed", "qc_pending")} == {
        "unreviewed": 1,
        "passed": 2,
        "failed": 1,
        "qc_pending": 1,
    }
    assert active["qc_state"] == "failed"


def test_sim_replay_gate_accepts_legacy_success_report_and_human_pass(tmp_path):
    report_dir = tmp_path / "meta" / "render_reports"
    report_dir.mkdir(parents=True)
    report = {"task_evaluation": {"success": True, "error": None}}
    report_path = report_dir / "episode_000000.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    episode = {
        "episode_index": 0,
        "sim_metadata": {"schema_version": 1},
        "qc_verdict": "pass",
        "quality": "green",
        "status": "saved",
    }
    waiting, waiting_reason = replay_validation(
        tmp_path / "missing", {"episode_index": 0, "sim_metadata": {"schema_version": 1}}
    )
    assert (waiting, waiting_reason) == (None, "replay_report_pending")
    verified, reason = replay_validation(tmp_path, episode)
    assert (verified, reason) == (True, "")
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 1)]})), {}, "set"
    )
    episode.update(
        slot_id=slots[0].slot_id,
        replay_required=True,
        replay_verified=True,
    )
    rows, _, counts = collection_slot_status(slots, [episode], [], CollectionSlotState([]))
    assert rows[0]["state"] == "complete"
    assert counts["complete"] == 1


@pytest.mark.parametrize("reference", ["default", "relative", "absolute"])
def test_replay_report_signature_tracks_the_actual_report(tmp_path, reference):
    episode = {"episode_index": 0, "sim_metadata": {"schema_version": 1}}
    if reference == "relative":
        episode["render_report"] = "custom/replay.json"
    elif reference == "absolute":
        episode["render_report"] = str(tmp_path / "external/replay.json")
    path = replay_report_path(tmp_path, episode)
    assert path is not None
    assert replay_report_signature(tmp_path, [episode]) == (0, 0, 0)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"task_evaluation": {"success": True, "error": None}}),
        encoding="utf-8",
    )
    signature = replay_report_signature(tmp_path, [episode])
    assert replay_validation(tmp_path, episode) == (True, "")
    path.write_text(
        json.dumps({"task_evaluation": {"success": False, "error": None}}),
        encoding="utf-8",
    )
    assert replay_report_signature(tmp_path, [episode]) != signature
    assert replay_validation(tmp_path, episode) == (False, "replay_summary_failed")


@pytest.mark.parametrize(
    ("success", "error"),
    [(False, None), (True, {"type": "failed"})],
)
def test_sim_replay_gate_rejects_failed_summary(tmp_path, success, error):
    report_dir = tmp_path / "meta" / "render_reports"
    report_dir.mkdir(parents=True)
    report = {
        "task_evaluation": {
            "success": success,
            "error": error,
        },
    }
    (report_dir / "episode_000000.json").write_text(json.dumps(report), encoding="utf-8")
    verified, reason = replay_validation(
        tmp_path, {"episode_index": 0, "sim_metadata": {"schema_version": 1}}
    )
    assert (verified, reason) == (False, "replay_summary_failed")


def test_sim_replay_gate_rejects_json_one_as_success(tmp_path):
    report_dir = tmp_path / "meta" / "render_reports"
    report_dir.mkdir(parents=True)
    report = {
        "task_evaluation": {
            "success": 1,
            "error": None,
        }
    }
    (report_dir / "episode_000000.json").write_text(json.dumps(report), encoding="utf-8")
    verified, reason = replay_validation(
        tmp_path, {"episode_index": 0, "sim_metadata": {"schema_version": 1}}
    )
    assert (verified, reason) == (False, "replay_summary_failed")


@pytest.mark.parametrize("verified", [None, False, True])
def test_human_pass_overrides_replay_without_erasing_machine_findings(verified):
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 1)]})), {}, "set"
    )
    episode = {
        "episode_index": 0,
        "slot_id": slots[0].slot_id,
        "status": "saved",
        "quality": "green",
        "qc_verdict": "pass",
        "replay_required": True,
        "replay_verified": verified,
        "replay_qc_reason": "replay_summary_failed" if verified is False else "",
        "qc_auto_verdict": "fail",
        "qc_auto_reason": "joint_limit_exceeded",
    }
    rows, _, counts = collection_slot_status(slots, [episode], [], CollectionSlotState([]))
    assert rows[0]["qc_state"] == "passed"
    assert counts["passed"] == 1
    dataset_slot = PlanCatalog._slot(slots[0].slot_id, "DEFAULT", 0, 1, episode)
    assert dataset_slot["qc_state"] == "passed"
    assert dataset_slot["episode"]["replay_verified"] is verified
    assert dataset_slot["episode"]["replay_qc_reason"] == episode["replay_qc_reason"]
    assert dataset_slot["episode"]["qc_auto_reason"] == "joint_limit_exceeded"
