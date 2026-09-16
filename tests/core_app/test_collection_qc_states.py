"""QC presentation stays independent of capture workflow and stale queue metadata."""

import pytest

from core.app.console.collection_slots import (
    CollectionSlotState,
    build_collection_slots,
    collection_slot_status,
)
from core.config import ConfigDict

pytestmark = pytest.mark.unit


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
        "failed",
        "pending",
    ]
    assert {key: counts[key] for key in ("unreviewed", "passed", "failed", "qc_pending")} == {
        "unreviewed": 1,
        "passed": 1,
        "failed": 2,
        "qc_pending": 1,
    }
    assert active["qc_state"] == "failed"


def test_new_take_does_not_inherit_previous_qc_verdict():
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 1)]})), {}, "set"
    )
    old = {
        "episode_index": 0,
        "slot_id": slots[0].slot_id,
        "status": "saved",
        "quality": "green",
        "qc_verdict": "fail",
    }
    new = {**old, "episode_index": 1, "qc_verdict": ""}
    rows, _, counts = collection_slot_status(slots, [old, new], [], CollectionSlotState([]))
    assert rows[0]["qc_state"] == "unreviewed"
    assert counts["failed"] == 0
    assert counts["unreviewed"] == 1


def test_explicit_unreviewed_keeps_automatic_issues():
    from core.app.console.collection_slots import episode_qc_state

    episode = {
        "status": "saved",
        "quality": "red",
        "qc_verdict": "unreviewed",
        "quality_issues": [{"detail": "camera skew"}],
    }
    assert episode_qc_state(episode) == "unreviewed"
    assert episode["quality_issues"] == [{"detail": "camera skew"}]
