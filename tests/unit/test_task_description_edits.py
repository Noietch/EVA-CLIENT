"""Editing display text must not re-identify collection slots or their history."""

import csv
import json

import pytest

from core.app.console.collection_slots import (
    CollectionSlotState,
    build_collection_slots,
    collection_slot_status,
)
from core.app.console.server import _load_scene_plan
from core.app.handlers.recording import load_episode_history
from core.config import ConfigDict, _normalize_collection_task_set

pytestmark = pytest.mark.unit


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def plan_files(tmp_path):
    root = tmp_path / "pour"
    root.mkdir()
    tasks = [
        dict(
            task_id="TASK-L",
            prompt_en="left pour and support",
            prompt_zh="左手倒，右手扶稳。",
            total_epsiodes_count=2,
            scene_ids="SC-L",
            scene_epsiodes_count=2,
        ),
        dict(
            task_id="TASK-R",
            prompt_en="right pour and support",
            prompt_zh="右手倒，左手扶稳。",
            total_epsiodes_count=2,
            scene_ids="SC-R",
            scene_epsiodes_count=2,
        ),
    ]
    _write_csv(root / "tasks.csv", tasks)
    _write_csv(
        root / "scene.csv",
        [
            {
                "scene_id": scene,
                "placements": json.dumps(
                    [
                        {"position_ids": [position], "object_id": "CUP"},
                    ]
                ),
            }
            for scene, position in [("SC-L", "P1"), ("SC-R", "P3")]
        ],
    )
    _write_csv(
        root / "objects.csv",
        [
            {"object_id": "CUP", "object_name_zh": "杯子", "photo_dir": "cup"},
        ],
    )
    photos = root / "object_photos" / "cup"
    photos.mkdir(parents=True)
    (photos / "cup.png").write_bytes(b"photo fixture")
    (root / "info.yaml").write_text("objects_file: objects.csv\n")
    return root, tasks


def _config(root):
    config = ConfigDict(collection=dict(task_set_dir=[str(root)], tasks={}))
    _normalize_collection_task_set(config)
    return config


@pytest.mark.parametrize(
    ("restart", "explicit_slot_id"),
    [(False, False), (True, True)],
)
def test_description_edit_preserves_slots_chinese_photos_and_history(
    plan_files,
    restart,
    explicit_slot_id,
):
    root, tasks = plan_files
    config = _config(root)
    old_plan = _load_scene_plan(config, "pour")
    old_slots = build_collection_slots(config, old_plan, "pour")
    episode = dict(
        episode_index=7,
        task="left pour and support",
        task_id="TASK-L",
        scene_id="SC-L",
        scene_round=0,
        quality="green",
        status="saved",
    )
    if explicit_slot_id:
        episode["slot_id"] = "TASK-L:SC-L:0"
    tasks[0].update(prompt_en="left pour", prompt_zh="左手倒。")
    tasks[1].update(prompt_en="right pour", prompt_zh="右手倒。")
    # Reordering rows must not reassign task indices by position.
    _write_csv(root / "tasks.csv", tasks[::-1])
    if restart:
        config = _config(root)
    plan = _load_scene_plan(config, "pour")
    slots = build_collection_slots(config, plan, "pour")
    assert [s.slot_id for s in slots] == [s.slot_id for s in old_slots]
    left = next(s for s in slots if s.slot_id == "TASK-L:SC-L:0")
    assert left.task_zh == "左手倒。"
    assert left.task == ("left pour" if restart else "left pour and support")
    assert config.collection.tasks["pour"][left.task_index][0] == left.task
    assert (
        plan["scenes"][0]["placements"][0]["photo_url"]
        == (old_plan["scenes"][0]["placements"][0]["photo_url"])
    )
    assert plan["scenes"][0]["placements"][0]["photo_url"]
    rows, _, counts = collection_slot_status(slots, [episode], [], CollectionSlotState([]))
    saved = next(row for row in rows if row["slot_id"] == "TASK-L:SC-L:0")
    assert saved["episode"]["episode_index"] == 7
    assert counts["complete"] == 1
    assert episode["task"] == "left pour and support"


def test_history_filters_by_identity_before_pagination(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    episodes = [
        dict(episode_index=0, tasks=["old wording"], task_id="TASK-L", length=5),
        dict(episode_index=1, tasks=["new wording"], task_id="TASK-R", length=5),
        dict(episode_index=2, tasks=["new wording"], length=5),
        dict(episode_index=3, tasks=["old wording"], task_id="TASK-L", length=5),
    ]
    path = meta / "episodes.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in episodes)
    path.write_text(original)
    first = load_episode_history(tmp_path, task="new wording", task_id="TASK-L", limit=1)
    assert first["total"] == 3
    second = load_episode_history(
        tmp_path,
        task="new wording",
        task_id="TASK-L",
        since=first["next_since"],
        cursor=first["cursor"],
        limit=5,
    )
    assert not second["reset"]
    ids = {row["episode_index"] for row in first["episodes"] + second["episodes"]}
    assert ids == {0, 2, 3}
    assert path.read_text() == original
