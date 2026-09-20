"""Exercise spatial navigation and exactly-once QC on the production UI handlers."""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.static


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
let collectionSlotClickTimer = null, collectionSlotsPolling = false, target = null, marks = 0;
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
