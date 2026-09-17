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
