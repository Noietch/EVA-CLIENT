"""The console collect grid and the dataset service must number squares alike."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.app.console.collection_slots import build_collection_slots
from core.config import ConfigDict, load_collection_task_set
from core.utils.scene_plan import scene_plan_from_dir
from tools.datasets.collection import PlanCatalog

pytestmark = pytest.mark.unit

BATCH = "order_batch"


def _write_csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _task_set(tmp_path: Path) -> Path:
    root = tmp_path / "task_sets" / BATCH
    root.mkdir(parents=True)
    (root / "info.yaml").write_text(
        "dataset_name: order\nrobot_type: dual_yam\ncollection_dir: order/raw\n",
        encoding="utf-8",
    )
    (root / "layout.yaml").write_text(
        "unit: mm\nsampling_points:\n"
        "  - {position_id: P1, x: 0, y: 0}\n"
        "  - {position_id: P2, x: 250, y: 0}\n"
        "  - {position_id: P3, x: 0, y: 250}\n",
        encoding="utf-8",
    )
    # SC-1 holds the plate nearer the origin than the cup, so the scene's tasks
    # sort against their tasks.csv order.
    _write_csv(
        root / "scene.csv",
        ["scene_id", "placements"],
        [
            [
                "SC-1",
                json.dumps(
                    [
                        {"object_id": "cup", "position_ids": ["P2"]},
                        {"object_id": "plate", "position_ids": ["P1"]},
                    ]
                ),
            ],
            ["SC-2", json.dumps([{"object_id": "cup", "position_ids": ["P3"]}])],
        ],
    )
    _write_csv(
        root / "tasks.csv",
        [
            "task_id",
            "prompt_en",
            "operation_object_ids",
            "scene_ids",
            "scene_epsiodes_count",
            "total_epsiodes_count",
        ],
        [
            ["TASK-1", "put cup on the plate", "cup", "SC-1;SC-2", "1;1", "2"],
            ["TASK-2", "put plate aside", "plate", "SC-1", "1", "1"],
        ],
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    return root


def _catalog(tmp_path: Path) -> PlanCatalog:
    return PlanCatalog(tmp_path / "task_sets", tmp_path / "assets", tmp_path / "collection")


def test_both_panels_order_and_number_the_same_slots(tmp_path):
    root = _task_set(tmp_path)
    plan = scene_plan_from_dir(root)
    bindings: dict[str, str] = {}
    tasks = load_collection_task_set(root, task_prompts=bindings)
    config = ConfigDict(collection=ConfigDict(tasks=tasks, task_prompt_bindings={BATCH: bindings}))

    console = build_collection_slots(config, plan, BATCH)
    assert [slot.slot_id for slot in console] == [
        "TASK-2:SC-1:0",
        "TASK-1:SC-1:0",
        "TASK-1:SC-2:0",
    ]

    state = _catalog(tmp_path).state(BATCH)
    assert [item["slot_id"] for item in state["slot_order"]] == [slot.slot_id for slot in console]
    ordinals = {
        slot["slot_id"]: slot["ordinal"] for task in state["tasks"] for slot in task["slots"]
    }
    assert ordinals == {slot.slot_id: slot.ordinal for slot in console}
