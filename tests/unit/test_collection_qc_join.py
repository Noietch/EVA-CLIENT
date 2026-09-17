"""QC records must not re-identify the recordings they are attached to."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.datasets.collection import PlanCatalog

pytestmark = pytest.mark.unit

BATCH = "bench_batch"


def _catalog(tmp_path: Path) -> PlanCatalog:
    plans = tmp_path / "task_sets"
    root = plans / BATCH
    root.mkdir(parents=True)
    (root / "info.yaml").write_text(
        "dataset_name: bench\nrobot_type: arx_x5\ncollection_dir: bench/raw\ntarget_episodes: 3\n",
        encoding="utf-8",
    )
    (root / "tasks.csv").write_text(
        "﻿task_id,prompt_en,total_epsiodes_count,scene_ids,scene_epsiodes_count\n"
        "TASK-1,pick up cup,3,SC-1,3\n",
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "objects.csv").write_text("﻿object_id,object_name\nOBJ-1,cup\n", encoding="utf-8")
    dataset = tmp_path / "collection" / "bench" / "raw" / "meta"
    dataset.mkdir(parents=True)
    episodes = [
        {"episode_index": index, "slot_id": f"TASK-1:SC-1:{index}", "length": 30}
        for index in range(3)
    ]
    (dataset / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    # A QC pass written against another enumeration: the slot column still names
    # the recording each verdict belongs to.
    qc = [
        {
            "episode_index": 2,
            "slot_id": "TASK-1:SC-1:0",
            "qc_verdict": "fail",
            "qc_note": "left camera",
        },
        {"episode_index": 0, "slot_id": "TASK-1:SC-1:2", "qc_verdict": "pass", "qc_note": ""},
        {"episode_index": 1, "slot_id": "TASK-1:SC-1:1", "qc_verdict": "unreviewed", "qc_note": ""},
    ]
    (dataset / "qc.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in qc), encoding="utf-8"
    )
    return PlanCatalog(plans, assets, tmp_path / "collection")


def test_qc_rows_keep_recording_identity_and_join_by_slot(tmp_path):
    catalog = _catalog(tmp_path)
    dataset_dir = tmp_path / "collection" / "bench" / "raw"
    rows = {row["episode_index"]: row for row in catalog._episode_rows(dataset_dir)}
    assert [rows[index]["slot_id"] for index in range(3)] == [
        "TASK-1:SC-1:0",
        "TASK-1:SC-1:1",
        "TASK-1:SC-1:2",
    ]
    assert rows[0]["qc_verdict"] == "fail"
    assert rows[0]["qc_note"] == "left camera"
    assert rows[1]["qc_verdict"] == "unreviewed"
    assert rows[2]["qc_verdict"] == "pass"


def test_shifted_qc_file_does_not_change_collection_counts(tmp_path):
    catalog = _catalog(tmp_path)
    summary = catalog._batch_summary(BATCH, catalog._plan_state(BATCH))
    assert summary["collected"] == 3
    assert summary["pending"] == 0
    assert summary["failed"] == 1
    assert summary["passed"] == 1
    assert summary["unreviewed"] == 1
