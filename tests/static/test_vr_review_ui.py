"""Exercise spatial navigation and exactly-once QC on the production UI handlers."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.static


def test_automatic_repair_slot_stays_selected():
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    handler = source[
        source.index("async function activateCollectionSlot(") : source.index(
            "async function pollCollectionSlots("
        )
    ]
    assert "manual: manual || !!slot.repair" in handler


def test_collection_save_refresh_waits_for_pipeline_boundary():
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/core/app/console/static/js/collect.js").read_text()
    handlers = source[
        source.index("function collectionPipelineState(") : source.index(
            "function applyCollectionSlotsPayload("
        )
    ]
    script = (
        r"""
const assert = require("node:assert/strict");
const S = {collectTaskSelectionPending: false,
  collectionSlots: {active: {slot_id: "old"}}};
let collectionSlotClickTimer = null;
let collectionPipelineSnapshot = null;
let collectionSaveRefreshPending = false;
let collectionSaveRefreshInFlight = false;
let calls = 0;
const pollCollectionSlots = (force) => {
  assert.equal(force, true);
  calls++;
  return Promise.resolve({ok: false, active: {slot_id: "new"}});
};
"""
        + handlers
        + r"""
(async () => {
  assert.deepEqual(collectionPipelineState({collecting: true}),
    {collecting: true, saving: false});
  assert.deepEqual(collectionPipelineState({pipeline_state: "SAVING", queued_jobs: 1}),
    {collecting: false, saving: true});
  assert.equal(collectionSaveRefreshNeeded(null, {collecting: false, saving: false}), false);
  // Stopping a capture while its save is still running only arms the refresh.
  assert.equal(collectionSaveRefreshNeeded(
    {collecting: true, saving: false}, {collecting: false, saving: true}), true);
  // The queue drain is the second boundary for a normal asynchronous save.
  assert.equal(collectionSaveRefreshNeeded(
    {collecting: false, saving: true}, {collecting: false, saving: false}), true);
  // A running capture and an unchanged idle state cannot trigger it.
  assert.equal(collectionSaveRefreshNeeded(
    {collecting: true, saving: false}, {collecting: true, saving: false}), false);
  assert.equal(collectionSaveRefreshNeeded(
    {collecting: false, saving: false}, {collecting: false, saving: false}), false);

  const flush = () => new Promise((resolve) => setImmediate(resolve));
  collectionPipelineSnapshot = {collecting: false, saving: true};
  collectionSaveRefreshPending = true;
  requestCollectionSaveRefresh();
  assert.equal(calls, 0);
  collectionPipelineSnapshot = {collecting: false, saving: false};
  requestCollectionSaveRefresh();
  requestCollectionSaveRefresh();
  assert.equal(calls, 1);
  await flush();
  assert.equal(collectionSaveRefreshInFlight, false);

  collectionSaveRefreshPending = true;
  S.collectTaskSelectionPending = true;
  requestCollectionSaveRefresh();
  assert.equal(calls, 1);
  S.collectTaskSelectionPending = false;
  collectionSlotClickTimer = {};
  requestCollectionSaveRefresh();
  assert.equal(calls, 1);
  collectionSlotClickTimer = null;
  collectionSaveRefreshPending = true;
  requestCollectionSaveRefresh();
  assert.equal(calls, 2);
  await flush();
  // A failed/cancelled read is observational only; it cannot advance the slot.
  assert.equal(S.collectionSlots.active.slot_id, "old");
})().catch((error) => {console.error(error); process.exitCode = 1;});
"""
    )
    subprocess.run(["node", "-e", script], check=True, timeout=10)


def test_review_cursor_and_qc_events():
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    outcome = source[
        source.index("function collectQcState(item)") : source.index("function collectTone(item)")
    ]
    handlers = (
        outcome
        + source[
            source.index("function collectionSlotQuery(dataset)") : source.index(
                "function setCollectError("
            )
        ]
        + source[
            source.index("let collectionReviewSeen = null;") : source.index(
                "function renderCollectionSlotFilters()"
            )
        ]
    )
    script = (
        r"""
const assert = require("node:assert/strict");
const S = {ACTIVE_TAB: "collect", STATUS: {collect: {}},
  collectionSlots: {selectedSlotId: "", slots: [], dataset: "test", page: 1, pageCount: 3}};
let collectionSlotClickTimer = null, collectionSlotsPollPromise = null, target = null, marks = 0;
const tiles = Array.from({length: 6}, (_, i) => ({
  dataset: {slotId: String(i)},
  getBoundingClientRect: () => ({
    x: (i % 3) * 30, y: Math.floor(i / 3) * 30, width: 20, height: 20,
  }),
  scrollIntoView() {},
}));
S.collectionSlots.slots = tiles.map((tile, i) => ({
  dataset: "test", slot_id: String(i), episode: i === 5 ? null : {episode_index: i},
}));
const status = {};
const adoptCollectionSlot = () => {};
const host = {querySelectorAll: () => tiles};
const $ = (id) => id === "collect-queue-tiles" ? host : status;
const collectControlsConfig = () => ({mode: "vr"});
const collectEnabled = () => true;
const savedEpisodeId = (item) => item?.episode_index ?? null;
const selectedCollectEpisodeItem = () => target;
const selectCollectionQcTarget = (item) => {target = item;};
const renderCollect = () => {};
let activations = 0, activated = null, previews = 0, liveReturns = 0;
const selectCollectEpisode = (item) => {previews++; S.reviewKind = "collect";};
const reviewActiveInCurrentTab = () => S.reviewKind === "collect";
const returnReviewToLive = () => {liveReturns++; S.reviewKind = "";};
const setCollectError = (message) => {status.textContent = message;};
const activateCollectionSlot = async (slot, options) => {
  assert.equal(options.manual, true);
  activations++; activated = slot.slot_id; return true;
};
const pollCollectionSlots = async () => {
  const start = (S.collectionSlots.page - 1) * 6;
  S.collectionSlots.slots = tiles.map((tile, i) => {
    tile.dataset.slotId = String(start + i);
    return {dataset: "test", slot_id: String(start + i), episode: {episode_index: start + i}};
  });
};
const submitEpisodeQc = async (kind, verdict) => {
  assert.equal(kind, "collect");
  target.qc_verdict = verdict; marks++;
};
"""
        + handlers
        + r"""
(async () => {
  const send = (events) => handleCollectionReviewInput({connected: true, review_events: events});
  // Moving the cursor is the selection: the step arms the slot it lands on.
  const step = (id, action, slotId) => {
    const before = activations;
    send([{id, action}]);
    assert.equal(S.collectionSlots.selectedSlotId, slotId);
    assert.equal(S.collectionSlots.cursorPinned, true);
    assert.equal(activations, before + 1);
    assert.equal(activated, slotId);
  };
  send([]);
  step("1", "right", "0");
  send([{id: "1", action: "right"}, {id: "2", action: "right"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "1");
  assert.equal(activated, "1");
  step("3", "down", "4");
  const mark = [{id: "4", action: "mark_red"}];
  send(mark); send(mark);
  await new Promise(setImmediate);
  assert.equal(marks, 1);
  assert.equal(target.qc_verdict, "fail");
  // Marking leaves the operator's tile and capture slot alone.
  assert.equal(S.collectionSlots.selectedSlotId, "4");
  assert.equal(activated, "4");
  target.quality = "red";
  send([{id: "4b", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(target.qc_verdict, "unreviewed");
  send([{id: "4c", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(target.qc_verdict, "pass");
  assert.equal(collectQcState(target), "passed");
  send([{id: "4d", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(target.qc_verdict, "fail");
  assert.equal(collectQcState(target), "failed");
  assert.equal(marks, 4);
  step("5", "right", "5");
  send([{id: "6", action: "toggle_qc"}]);
  assert.equal(marks, 4);
  S.ACTIVE_TAB = "rl";
  send([{id: "7", action: "left"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  S.ACTIVE_TAB = "collect";
  resetCollectionReviewInput();
  send([{id: "8", action: "toggle_qc"}]);
  assert.equal(marks, 4);
  // A press waits out the double click and then only re-asserts the cursor.
  const pressed = activations;
  send([{id: "9", action: "select"}]);
  assert.equal(activations, pressed);
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, pressed + 1);
  assert.equal(activated, "5");
  send([{id: "9", action: "select"}]);
  assert.equal(activations, pressed + 1);
  S.STATUS.collect.collecting = true;
  send([{id: "10", action: "select"}]);
  assert.equal(activations, pressed + 1);
  S.STATUS.collect.collecting = false;
  send([{id: "11", action: "right"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 2);
  assert.equal(S.collectionSlots.selectedSlotId, "6");
  assert.equal(activated, "6");
  send([{id: "12", action: "left"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 1);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  assert.equal(activated, "5");
  send([{id: "13", action: "down"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 2);
  assert.equal(S.collectionSlots.selectedSlotId, "8");
  assert.equal(activated, "8");
  send([{id: "14", action: "up"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 1);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  assert.equal(activated, "5");
  send([{id: "15", action: "up"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "2");
  assert.equal(activated, "2");
  send([{id: "16", action: "up"}]);
  assert.equal(S.collectionSlots.page, 1);
  assert.equal(S.collectionSlots.selectedSlotId, "2");

  // A double press previews through the same helper as mouse double click,
  // and never switches the capture slot on its own.
  const previewed = activations;
  send([{id: "17", action: "select"}]);
  send([{id: "17", action: "select"}, {id: "18", action: "select"}]);
  assert.equal(previews, 1);
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, previewed);
  for (const [index, action] of ["up", "down", "left", "right"].entries()) {
    S.reviewKind = "collect";
    send([{id: `live-${index}`, action}]);
    assert.equal(S.reviewKind, "");
    assert.equal(S.collectionSlots.selectedSlotId, "2");
  }
  assert.equal(liveReturns, 4);
  // Navigation and disconnection cancel pending single-click selection.
  const cancelled = activations;
  send([{id: "19", action: "select"}]);
  send([{id: "20", action: "left"}]);
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, cancelled + 1);
  assert.equal(activated, "1");
  send([{id: "21", action: "select"}]);
  resetCollectionReviewInput();
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, cancelled + 1);
  send([]);
  send([{id: "22", action: "select"}]);
  S.STATUS.collect.collecting = true;
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, cancelled + 1);

  // The grid keeps one cursor: it follows the active slot until the operator
  // pins it, and an active slot outside the page frames nothing.
  S.STATUS.collect.collecting = false;
  S.collectionSlots.active = {slot_id: "3"};
  S.collectionSlots.cursorPinned = false;
  syncCollectionSlotCursor();
  assert.equal(S.collectionSlots.selectedSlotId, "3");
  assert.equal(S.collectionSlots.cursorPinned, false);
  S.collectionSlots.active = {slot_id: "99"};
  syncCollectionSlotCursor();
  assert.equal(S.collectionSlots.selectedSlotId, "");
  S.collectionSlots.active = {slot_id: "4"};
  S.collectionSlots.selectedSlotId = "2";
  S.collectionSlots.cursorPinned = true;
  syncCollectionSlotCursor();
  assert.equal(S.collectionSlots.selectedSlotId, "2");

  // A verdict pins the cursor, so a QC change never moves the framed tile.
  resetCollectionReviewInput();
  send([]);
  S.collectionSlots.selectedSlotId = "5";
  S.collectionSlots.cursorPinned = false;
  S.collectionSlots.active = {slot_id: "5"};
  target = {episode_index: 5, qc_verdict: "pass"};
  send([{id: "23", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.cursorPinned, true);
  S.collectionSlots.active = {slot_id: "6"};
  syncCollectionSlotCursor();
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  assert.equal(target.qc_verdict, "fail");

  // A poll keeps that single cursor: the armed slot when nothing is pinned,
  // the operator's tile while it is.
  const pollPayload = {
    dataset: "test", active: {slot_id: "4"}, counts: {}, page: 1, page_count: 1,
    slots: [{slot_id: "4"}, {slot_id: "5"}],
  };
  S.collectionSlots.cursorPinned = false;
  applyCollectionSlotsPayload(pollPayload);
  assert.equal(S.collectionSlots.selectedSlotId, "4");
  assert.equal(S.collectionSlots.cursorPinned, false);
  S.collectionSlots.selectedSlotId = "5";
  S.collectionSlots.cursorPinned = true;
  applyCollectionSlotsPayload(pollPayload);
  assert.equal(S.collectionSlots.selectedSlotId, "5");

})().catch((error) => {console.error(error); process.exitCode = 1;});
"""
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

    # One cursor also means one framed tile: no second "current data" outline.
    console = Path(__file__).resolve().parents[2] / "src/core/app/console/static"
    for asset in ("css/console.css", "js/collect.js"):
        assert "slot-current" not in (console / asset).read_text()


def test_machine_failure_color_and_editable_note():
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/core/app/console/static/js/collect.js").read_text()
    handlers = (
        source[source.index("function collectQcState(") : source.index("function collectTone(")]
        + source[
            source.index("function collectQualityReason(") : source.index(
                "function renderCollectionQcControls("
            )
        ]
    )
    script = (
        r"""
const assert = require("node:assert/strict");
const note = {value: "", defaultValue: ""};
const status = {};
const $ = (id) => id === "collect-qc-note" ? note : status;
const S = {collectionSlots: {dataset: "set"}};
let collectionQcTarget = null;
const reviewDatasetFor = () => "/dataset";
const collectTaskValue = () => "task";
const t = (key) => ({
  "collect.qc.replay_summary_failed": "REPLAY FAILED",
  "collect.qc.replay_report_invalid": "REPLAY REPORT INVALID",
  "collect.qc.replay_evaluation_missing": "REPLAY EVALUATION MISSING",
}[key] || key);
"""
        + handlers
        + r"""
const episode = {status: "saved", episode_index: 49, quality: "green",
  qc_verdict: "unreviewed", qc_note: "human note", qc_auto_verdict: "fail",
  qc_auto_reason: "joint_limit_exceeded", qc_auto_note: "joint4 exceeds its limit"};
assert.equal(collectQcState(episode), "failed");
assert.equal(collectResultLabel(episode), "QC FAILED");
selectCollectionQcTarget(episode);
assert.equal(note.value, "human note");
const automatic = {...episode, qc_note: ""};
selectCollectionQcTarget(automatic);
assert.equal(note.value, "joint4 exceeds its limit");
assert.equal(automatic.qc_note, "");
assert.equal(automatic.qc_verdict, "unreviewed");

// New replay findings fill the selected episode's untouched text box on refresh.
automatic.replay_required = true;
automatic.replay_verified = false;
automatic.replay_qc_reason = "replay_summary_failed";
syncCollectionQcNote(automatic);
assert.equal(note.value, "joint4 exceeds its limit · REPLAY FAILED");
note.value = "unsaved human edit";
syncCollectionQcNote({...automatic, replay_qc_reason: "updated machine finding"});
assert.equal(note.value, "unsaved human edit");
note.value = "";
syncCollectionQcNote(automatic);
assert.equal(note.value, "");
selectCollectionQcTarget(automatic);
assert.equal(note.value, "joint4 exceeds its limit · REPLAY FAILED");
automatic.replay_verified = true;
syncCollectionQcNote(automatic);
assert.equal(note.value, "joint4 exceeds its limit");

for (const [code, label] of [
  ["replay_summary_failed", "REPLAY FAILED"],
  ["replay_report_invalid", "REPLAY REPORT INVALID"],
  ["replay_evaluation_missing", "REPLAY EVALUATION MISSING"],
]) {
  selectCollectionQcTarget({status: "saved", episode_index: 52,
    replay_verified: false, replay_qc_reason: code});
  assert.equal(note.value, label);
}
selectCollectionQcTarget({status: "saved", episode_index: 53,
  qc_auto_note: "REPLAY FAILED", replay_verified: false,
  replay_qc_reason: "replay_summary_failed"});
assert.equal(note.value, "REPLAY FAILED");
selectCollectionQcTarget({status: "saved", episode_index: 54,
  qc_auto_reason: "joint_limit_exceeded"});
assert.equal(note.value, "joint_limit_exceeded");
selectCollectionQcTarget({status: "saved", episode_index: 55, qc_auto: true,
  qc_reason: "primary machine reason"});
assert.equal(note.value, "primary machine reason");
selectCollectionQcTarget({status: "saved", episode_index: 56,
  quality_issues: [{detail: "camera skew", count: 2}]});
assert.equal(note.value, "camera skew ×2");
episode.qc_verdict = "pass";
for (const verified of [undefined, null, false, true]) {
  const simEpisode = {...episode, replay_required: true, replay_verified: verified};
  assert.equal(collectQcState(simEpisode), "passed");
  assert.equal(collectOutcome(simEpisode), "usable");
  selectCollectionQcTarget(simEpisode);
  assert.equal(note.value, "human note");
}
selectCollectionQcTarget({status: "saved", episode_index: 50});
assert.equal(note.value, "");
selectCollectionQcTarget(null);
assert.equal(note.value, "");
"""
    )
    subprocess.run(["node", "-e", script], check=True, timeout=10)
    template = (root / "src/core/app/console/static/index.html").read_text()
    assert 'id="collect-qc-machine"' not in template


def test_larybench_display_grid_maps_world_points_to_21_cells():
    node = shutil.which("node")
    assert node is not None, "Install Node.js to run JavaScript validation"
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/scene_map.js"
    ).read_text()
    chunk = source[
        source.index("function scenePlanGridPositions(") : source.index(
            "function normalizedObjectText("
        )
    ]
    chunk += source[
        source.index("function renderSceneGridCells(") : source.index(
            "export function renderSceneGrid("
        )
    ]
    labels = [
        "P10",
        "P11",
        "P1",
        "P2",
        "P3",
        "P12",
        "P13",
        "P14",
        "P15",
        "P4",
        "P5",
        "P6",
        "P16",
        "P17",
        "P18",
        "P19",
        "P7",
        "P8",
        "P9",
        "P20",
        "P21",
    ]
    columns = [0.42, 0.28, 0.14, 0.0, -0.14, -0.28, -0.42]
    rows = [0.56, 0.42, 0.28]
    positions = [
        {"position_id": label, "x": rows[index // 7], "y": columns[index % 7]}
        for index, label in enumerate(labels)
    ]
    payload = json.dumps(
        {
            "positions": positions,
            "display_grid": {
                "column_axis": "y",
                "row_axis": "x",
                "columns": columns,
                "rows": rows,
                "position_ids": labels,
            },
        }
    )
    script = f"""
const plan = {payload};
const functions = new Function({json.dumps(chunk)} +
  "; return {{scenePlanGridPositions, scenePlanGridGeometry, renderSceneGridCells}};")();
const positions = functions.scenePlanGridPositions(plan, null);
const geometry = functions.scenePlanGridGeometry(plan, positions);
if (geometry.columns !== 7 || geometry.rows !== 3 || geometry.cells.length !== 21) {{
  throw new Error("grid");
}}
const p1 = geometry.cells.find((cell) => cell.label === "P1");
if (!p1 || p1.column !== 3 || p1.row !== 1 || !p1.positionIds.includes("P1")) {{
  throw new Error("P1");
}}
if (geometry.cells.reduce((count, cell) => count + cell.positionIds.length, 0) !== 21) {{
  throw new Error("points");
}}
globalThis.document = {{createElement: () => ({{
  dataset: {{}}, style: {{}}, append() {{}},
}})}};
const host = {{
  dataset: {{}}, style: {{setProperty() {{}}}}, renders: 0,
  replaceChildren(...cells) {{this.cells = cells; this.renders++;}},
}};
functions.renderSceneGridCells(host, plan, null);
functions.renderSceneGridCells(host, plan, null);
if (host.renders !== 1) throw new Error("unchanged grid rebuilt");
plan.positions[0].x += 0.001;
functions.renderSceneGridCells(host, plan, null);
if (host.renders !== 2 || !host.cells[0].title.includes("0.5610")) {{
  throw new Error("updated coordinates not rendered");
}}
"""
    result = subprocess.run(
        [node, "--input-type=module"], input=script, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_recording_slot_is_pending_and_titles_include_quality_reason():
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    handlers = (
        source[source.index("function collectQcState(") : source.index("function collectTone(")]
        + source[
            source.index("function collectQualityReason(") : source.index(
                "function savedEpisodeId("
            )
        ]
        + source[
            source.index("function savedEpisodeId(") : source.index(
                "function syncScenePlanToEpisode("
            )
        ]
        + source[
            source.index("function collectionRecordingSlotId(") : source.index(
                "// The cursor is the capture slot"
            )
        ]
        + source[source.index("function renderCollectTiles(") : source.index("function pipeBadge(")]
    )
    script = (
        r"""
const assert = require("node:assert/strict");
const S = {
  STATUS: {
    collect: {collecting: true},
    collection_slot_id: "slot-1",
  },
  collectionSlots: {
    dataset: "set-a", sessionDataset: "set-a", sessionSlotId: "slot-1",
    selectedSlotId: "", active: {slot_id: "slot-1"}, viewerActive: true,
  },
};
const t = (key) => ({
  "collect.slot": "SLOT", "collect.round": "ROUND", "collect.episodeLabel": "EPISODE",
  "collect.qc.replay_summary_failed": "REPLAY FAILED",
  "collect.qc.replay_report_invalid": "REPLAY REPORT INVALID",
}[key] || key);
function classList() {
  const values = new Set();
  return {
    add: (...names) => names.forEach((name) => values.add(name)),
    toggle: (name, enabled) => enabled ? values.add(name) : values.delete(name),
  };
}
function tile() {
  return {
    dataset: {}, classList: classList(), attributes: {},
    setAttribute(name, value) { this.attributes[name] = String(value); },
  };
}
const host = {children: [], contains: () => false, appendChild(item) { this.children.push(item); }};
Object.defineProperty(host, "innerHTML", {set() { this.children = []; }});
const $ = (id) => { assert.equal(id, "collect-queue-tiles"); return host; };
const document = {activeElement: null, createElement: () => tile()};
"""
        + handlers
        + r"""
const render = (slot) => { renderCollectTiles([slot]); return host.children[0]; };
const oldEpisode = {
  status: "saved", episode_index: 7, quality: "red", qc_verdict: "unreviewed",
  qc_note: "old note", qc_reason: "old reason",
};
let slot = {
  slot_id: "slot-1", ordinal: 0, scene_id: "scene", task: "task",
  round_index: 0, round_total: 1, state: "active", qc_state: "failed", episode: oldEpisode,
};

// Retake in progress uses yellow pending-QC styling and never repeats the old reason.
let recording = render(slot);
assert.match(recording.className, /slot-unreviewed/);
assert.doesNotMatch(recording.className, /slot-failed/);
assert.match(recording.title, /RECORDING/);
assert.doesNotMatch(recording.title, /old note|old reason/);
assert.equal(recording.attributes["aria-busy"], "true");

// After recording, the new failed QC result and its new reason are shown.
S.STATUS.collect.collecting = false;
slot = {...slot, qc_state: "failed", episode: {
  status: "saved", episode_index: 7, quality: "green", qc_verdict: "unreviewed",
  qc_auto_note: "new note", qc_auto_reason: "new reason",
}};
const newResult = render(slot);
assert.match(newResult.className, /slot-failed/);
assert.match(newResult.title, /new note/);
assert.match(newResult.title, /new reason/);

// Replay failures join the machine reason, while a successful replay clears it.
slot = {...slot, episode: {...slot.episode, replay_required: true,
  replay_verified: false, replay_qc_reason: "replay_summary_failed"}};
const replayFailure = render(slot);
assert.match(replayFailure.title, /new note/);
assert.match(replayFailure.title, /new reason/);
assert.match(replayFailure.title, /REPLAY FAILED/);
assert.equal((replayFailure.title.match(/REPLAY FAILED/g) || []).length, 1);
slot = {...slot, episode: {...slot.episode, replay_verified: true}};
const replayPassed = render(slot);
assert.doesNotMatch(replayPassed.title, /REPLAY FAILED/);

// A replay-only failure still has a readable reason when no auto-QC finding exists.
slot = {...slot, episode: {status: "saved", episode_index: 7, quality: "green",
  qc_verdict: "unreviewed", replay_required: true, replay_verified: false,
  replay_qc_reason: "replay_report_invalid"}};
const replayOnly = render(slot);
assert.equal((replayOnly.title.match(/REPLAY REPORT INVALID/g) || []).length, 1);

// Cancelling the retake leaves the previous failed state and reason visible.
slot = {...slot, qc_state: "failed", episode: oldEpisode};
const cancelled = render(slot);
assert.match(cancelled.className, /slot-failed/);
assert.match(cancelled.title, /old note/);
assert.match(cancelled.title, /old reason/);

// A same-named slot from another dataset must not inherit the recording color.
S.STATUS.collect.collecting = true;
S.collectionSlots.dataset = "set-b";
S.collectionSlots.sessionDataset = "set-a";
const otherDataset = render(slot);
assert.match(otherDataset.className, /slot-failed/);
assert.doesNotMatch(otherDataset.className, /slot-unreviewed/);
assert.doesNotMatch(otherDataset.title, /RECORDING/);
"""
    )
    subprocess.run(
        ["node", "-e", script],
        check=True,
        timeout=10,
    )


def test_dataset_qc_optimistic_human_pass_overrides_replay_and_rolls_back_on_error():
    source = (Path(__file__).resolve().parents[2] / "tools/datasets/static/review.js").read_text()
    handler = source[
        source.index("async function saveQc(") : source.index("function reviewSeries(")
    ]
    script = (
        r"""
const assert = require("node:assert/strict");
const controls = {"qc-reason": {value: ""}, "qc-note": {value: ""}};
const $ = (id) => controls[id];
const showToast = () => {};
const translate = (key) => key;
const REVIEW_REASON = "needs_manual_review";
const renderTaskList = () => {};
const navigateQcSlot = () => {};
const writeJson = async () => {};
let saved = true;
const runWrite = async (control, operation) => {await operation(); return saved;};
"""
        + handler
        + r"""
(async () => {
  for (const verified of [null, false, true]) {
    const slot = {qc_state: "failed", episode: {
      episode_index: 0, qc_verdict: "unreviewed", qc_auto_verdict: "fail",
      replay_required: true, replay_verified: verified, replay_qc_reason: "original replay finding",
      qc_auto_reason: "joint_limit_exceeded",
    }};
    await saveQc({batch_id: "batch"}, slot, "pass", null);
    assert.equal(slot.qc_state, "passed");
    assert.equal(slot.episode.qc_verdict, "pass");
    assert.equal(slot.episode.replay_verified, verified);
    assert.equal(slot.episode.replay_qc_reason, "original replay finding");
    assert.equal(slot.episode.qc_auto_verdict, "fail");
    assert.equal(slot.episode.qc_auto_reason, "joint_limit_exceeded");
    await saveQc({batch_id: "batch"}, slot, "unreviewed", null);
    assert.equal(slot.qc_state, "failed");
  }
  saved = false;
  const slot = {qc_state: "failed", episode: {
    episode_index: 0, qc_verdict: "unreviewed", qc_auto_verdict: "fail",
    replay_required: true, replay_verified: null,
  }};
  await saveQc({batch_id: "batch"}, slot, "pass", null);
  assert.equal(slot.qc_state, "failed");
  assert.equal(slot.episode.qc_verdict, "unreviewed");
})().catch((error) => {console.error(error); process.exitCode = 1;});
"""
    )
    subprocess.run(["node", "-e", script], check=True, timeout=10)


def test_dataset_refresh_and_stale_review_responses_preserve_fresh_cache():
    root = Path(__file__).resolve().parents[2]
    source = (root / "tools/datasets/static/review.js").read_text()
    editor = (root / "tools/datasets/static/editor.js").read_text()
    handlers = (
        source[
            source.index("async function openSlot(") : source.index("function renderReviewLoading(")
        ]
        + editor[editor.index("async function loadState(") : editor.index("function applyState(")]
    )
    script = (
        r"""
const assert = require("node:assert/strict");
const app = {selected: {}, original: {}, draft: {}, reviewToken: 0, reviewCache: new Map(),
  qcLoadingToken: 0, stateCache: new Map(), stateRequests: new Map()};
const $ = (id) => id === "qc-left-controls" ? null : {replaceChildren() {}};
const stopPlayback = () => {};
const recordKey = (task) => task.task_id;
const clone = (value) => ({...value});
const renderTaskList = () => {};
const renderReviewLoading = () => {};
const renderReview = () => {};
const loadThree = async () => {};
const showGlobalError = () => {};
const setQcLoading = () => {};
const readStoredState = () => null;
const storeState = () => {};
const applyState = () => {};
const moveTabThumb = () => {};
const requestAnimationFrame = () => 0;
let api;
"""
        + handlers
        + r"""
(async () => {
  const task = {batch_id: "batch", task_id: "task"};
  const slot = {slot_id: "slot"};
  let resolveOld;
  api = () => new Promise((resolve) => {resolveOld = resolve;});
  const oldRequest = openSlot(task, slot);
  const fresh = {series: "fresh"};
  api = async () => fresh;
  await openSlot(task, slot);
  resolveOld({series: "old"});
  await oldRequest;
  assert.equal(app.review, fresh);
  assert.equal(app.reviewCache.get("batch::slot").payload, fresh);
  api = async () => ({tasks: []});
  await loadState(false, true);
  assert.equal(app.reviewCache.size, 0);
})().catch((error) => {console.error(error); process.exitCode = 1;});
"""
    )
    subprocess.run(["node", "-e", script], check=True, timeout=10)
