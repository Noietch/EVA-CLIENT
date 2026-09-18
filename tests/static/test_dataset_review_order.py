"""Exercise the review grid's tile order against the plan's published slot order."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.static


def test_review_grid_follows_the_plan_slot_order():
    node = shutil.which("node")
    assert node is not None, "Install Node.js to run JavaScript syntax validation"
    source = (
        Path(__file__).resolve().parents[2] / "tools/datasets/static/entity-ui.js"
    ).read_text()
    functions = (
        source[source.index("function compareTaskIds(") : source.index("function buildSlotTile(")]
        + source[
            source.index("function qcSlotEntries()") : source.index("function renderTaskList()")
        ]
    )
    script = (
        r"""
import assert from "node:assert/strict";
const app = {batch: "B1", state: null};
"""
        + functions
        + r"""
app.state = {
  tasks: [
    {batch_id: "B1", task_id: "TASK-1",
      slots: [{slot_id: "TASK-1:SC-1:0"}, {slot_id: "TASK-1:SC-2:0"}]},
    {batch_id: "B1", task_id: "TASK-2", slots: [{slot_id: "TASK-2:SC-1:0"}]},
  ],
  slot_order: [
    {batch_id: "B1", slot_id: "TASK-2:SC-1:0"},
    {batch_id: "B1", slot_id: "TASK-1:SC-1:0"},
    {batch_id: "B1", slot_id: "TASK-1:SC-2:0"},
  ],
};
const ids = () => qcSlotEntries().map((entry) => entry.slot.slot_id);
assert.deepEqual(ids(), ["TASK-2:SC-1:0", "TASK-1:SC-1:0", "TASK-1:SC-2:0"]);

// A slot the plan no longer describes keeps a place instead of disappearing.
app.state.tasks.push({batch_id: "B1", task_id: "TASK-3", slots: [{slot_id: "TASK-3:SC-1:0"}]});
assert.deepEqual(ids(), [
  "TASK-2:SC-1:0", "TASK-1:SC-1:0", "TASK-1:SC-2:0", "TASK-3:SC-1:0",
]);

// Batches without a published order (unmatched recordings) keep grouped tiles.
app.state.slot_order = [];
assert.deepEqual(ids(), [
  "TASK-1:SC-1:0", "TASK-1:SC-2:0", "TASK-2:SC-1:0", "TASK-3:SC-1:0",
]);
"""
    )
    result = subprocess.run(
        [node, "--input-type=module"], input=script, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr

    # The grid only follows that order while both render paths go through it.
    for caller in ("renderTaskList", "navigateQcSlot"):
        body = source[source.index(f"function {caller}(") :][:400]
        assert "const entries = qcSlotEntries();" in body, caller
