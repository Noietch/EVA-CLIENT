"""Editing display text must not re-identify collection slots or their history."""

import csv
import json
import shutil
import subprocess
from pathlib import Path

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
        dict(task_id="TASK-L", prompt_en="left pour and support", prompt_zh="左手倒，右手扶稳。",
             total_epsiodes_count=2, scene_ids="SC-L", scene_epsiodes_count=2),
        dict(task_id="TASK-R", prompt_en="right pour and support", prompt_zh="右手倒，左手扶稳。",
             total_epsiodes_count=2, scene_ids="SC-R", scene_epsiodes_count=2),
    ]
    _write_csv(root / "tasks.csv", tasks)
    _write_csv(root / "scene.csv", [
        {"scene_id": scene, "placements": json.dumps([
            {"position_ids": [position], "object_id": "CUP"},
        ])}
        for scene, position in [("SC-L", "P1"), ("SC-R", "P3")]
    ])
    _write_csv(root / "objects.csv", [
        {"object_id": "CUP", "object_name_zh": "杯子", "photo_dir": "cup"},
    ])
    photos = root / "object_photos" / "cup"
    photos.mkdir(parents=True)
    (photos / "cup.png").write_bytes(b"photo fixture")
    (root / "info.yaml").write_text("objects_file: objects.csv\n")
    return root, tasks


def _config(root):
    config = ConfigDict(collection=dict(task_set_dir=[str(root)], tasks={}))
    _normalize_collection_task_set(config)
    return config


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("explicit_slot_id", [False, True])
def test_description_edit_preserves_slots_chinese_photos_and_history(
    plan_files, restart, explicit_slot_id,
):
    root, tasks = plan_files
    config = _config(root)
    old_plan = _load_scene_plan(config, "pour")
    old_slots = build_collection_slots(config, old_plan, "pour")
    episode = dict(episode_index=7, task="left pour and support", task_id="TASK-L",
                   scene_id="SC-L", scene_round=0, quality="green", status="saved")
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
    assert plan["scenes"][0]["placements"][0]["photo_url"] == (
        old_plan["scenes"][0]["placements"][0]["photo_url"]
    )
    assert plan["scenes"][0]["placements"][0]["photo_url"]
    rows, _, counts = collection_slot_status(slots, [episode], [], CollectionSlotState([]))
    saved = next(row for row in rows if row["slot_id"] == "TASK-L:SC-L:0")
    assert saved["episode"]["episode_index"] == 7
    assert counts["complete"] == 1
    assert episode["task"] == "left pour and support"


def test_unmatched_scene_tasks_do_not_become_inline_tasks(plan_files):
    root, _ = plan_files
    config = ConfigDict(collection=dict(task_set_dir=[str(root)], tasks={"pour": [("unknown", 2)]}))
    assert build_collection_slots(config, _load_scene_plan(config, "pour"), "pour") == []


def test_bindings_do_not_leak_between_configs(plan_files):
    root, tasks = plan_files
    old_config = _config(root)
    tasks[0]["prompt_en"] = "left pour"
    _write_csv(root / "tasks.csv", tasks)
    new_config = _config(root)
    for config, expected in [(old_config, "left pour and support"), (new_config, "left pour")]:
        plan = _load_scene_plan(config, "pour")
        assert plan["tasks"][0]["runtime_prompt"] == expected


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
        tmp_path, task="new wording", task_id="TASK-L", since=first["next_since"],
        cursor=first["cursor"], limit=5,
    )
    assert not second["reset"]
    ids = {row["episode_index"] for row in first["episodes"] + second["episodes"]}
    assert ids == {0, 2, 3}
    assert path.read_text() == original


def test_frontend_retains_scene_and_history_after_description_edit():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend matching regression")
    root = Path(__file__).resolve().parents[2]
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const collect = fs.readFileSync('src/core/app/console/static/js/collect.js', 'utf8');
const run = fs.readFileSync('src/core/app/console/static/js/run.js', 'utf8');
function extract(source, name) {
  const start = source.indexOf('function ' + name + '(');
  const next = source.indexOf('\nfunction ', start + 1);
  assert(start >= 0 && next > start);
  return source.slice(start, next);
}
const state = {
  SCENE_PLAN: {
    tasks: [
      {task_id: 'R', prompt_en: 'old left', runtime_prompt: 'old right', scene_ids: ['SR']},
      {task_id: 'L', prompt_en: 'new left', runtime_prompt: 'old left', scene_ids: ['SL']},
    ],
    scenes: [{scene_id: 'SL', placements: [{photo_url: '/cup.png'}]}],
  },
  collectionSlots: {active: {task_id: 'L', slot_id: 'L:SL:0'}},
};
state.SCENE_PLAN.tasks[1].scene_epsiodes_count = [5];
const context = {S: state, collectTaskValue: () => 'old left'};
vm.createContext(context);
for (const name of ['scenePlanTasks', 'scenePlanTask', 'scenePlanScene',
    'scenePlanStartMetadata', 'syncScenePlanTask', 'itemsForPrompt', 'syncScenePlanToEpisode']) {
  vm.runInContext(extract(collect, name), context);
}
vm.runInContext(extract(run, 'collectTaskPlan'), context);
context.syncScenePlanTask('old left');
assert.equal(state.scenePlanTaskId, 'L');
assert.equal(context.scenePlanScene().placements[0].photo_url, '/cup.png');
assert.equal(context.scenePlanStartMetadata().slot_id, 'L:SL:0');
assert.equal(context.collectTaskPlan('old left').task_id, 'L');
const history = context.itemsForPrompt([
  {episode_index: 1, task_id: 'L', task: 'historical wording'},
  {episode_index: 2, task_id: 'R', task: 'old left'},
  {episode_index: 3, task: 'old left'},
], 'old left');
assert.deepEqual(Array.from(history, row => row.episode_index), [1, 3]);
context.syncScenePlanToEpisode({task_id: 'L', task: 'old right', scene_id: 'SL', scene_round: 2});
assert.equal(state.scenePlanTaskId, 'L');
assert.equal(state.scenePlanRoundIndex, 2);
"""
    subprocess.run([node, "-e", script], cwd=root, check=True, capture_output=True, text=True)
