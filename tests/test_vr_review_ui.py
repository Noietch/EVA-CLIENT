"""Exercise spatial navigation and exactly-once QC on the production UI handlers."""
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.static


def test_review_cursor_and_qc_events():
    source = (Path(__file__).resolve().parents[1] /
              "src/core/app/console/static/js/collect.js").read_text()
    outcome = source[source.index("function collectOutcome(item)"):
                     source.index("function collectTone(item)")]
    handlers = outcome + source[source.index("let collectionReviewSeen = null;"):
                      source.index("function renderCollectionSlotFilters()")]
    script = r'''
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
'''+handlers+r'''
(async () => {
  const send = (events) => handleCollectionReviewInput({connected: true, review_events: events});
  send([]);
  send([{id: "1", action: "right"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "0");
  send([{id: "1", action: "right"}, {id: "2", action: "right"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "1");
  send([{id: "3", action: "down"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "4");
  const mark = [{id: "4", action: "mark_red"}];
  send(mark); send(mark);
  await new Promise(setImmediate);
  assert.equal(marks, 1);
  assert.equal(target.qc_verdict, "fail");
  send([{id: "4b", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(target.qc_verdict, "pass");
  send([{id: "4c", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(target.qc_verdict, "fail");
  target.quality = "red";
  send([{id: "4d", action: "toggle_qc"}]);
  await new Promise(setImmediate);
  assert.equal(target.qc_verdict, "pass");
  assert.equal(collectOutcome(target), "usable");
  assert.equal(marks, 4);
  send([{id: "5", action: "right"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  send([{id: "6", action: "toggle_qc"}]);
  assert.equal(marks, 4);
  S.ACTIVE_TAB = "rl";
  send([{id: "7", action: "left"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  S.ACTIVE_TAB = "collect";
  resetCollectionReviewInput();
  send([{id: "8", action: "toggle_qc"}]);
  assert.equal(marks, 4);
  send([{id: "9", action: "select"}]);
  assert.equal(activations, 0);
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, 1);
  assert.equal(activated, "5");
  send([{id: "9", action: "select"}]);
  assert.equal(activations, 1);
  S.STATUS.collect.collecting = true;
  send([{id: "10", action: "select"}]);
  assert.equal(activations, 1);
  S.STATUS.collect.collecting = false;
  send([{id: "11", action: "right"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 2);
  assert.equal(S.collectionSlots.selectedSlotId, "6");
  send([{id: "12", action: "left"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 1);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  send([{id: "13", action: "down"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 2);
  assert.equal(S.collectionSlots.selectedSlotId, "8");
  send([{id: "14", action: "up"}]);
  await new Promise(setImmediate);
  assert.equal(S.collectionSlots.page, 1);
  assert.equal(S.collectionSlots.selectedSlotId, "5");
  send([{id: "15", action: "up"}]);
  assert.equal(S.collectionSlots.selectedSlotId, "2");
  send([{id: "16", action: "up"}]);
  assert.equal(S.collectionSlots.page, 1);
  assert.equal(S.collectionSlots.selectedSlotId, "2");

  // A double click previews through the same helper as mouse double click,
  // and never changes the active capture slot.
  send([{id: "17", action: "select"}]);
  send([{id: "17", action: "select"}, {id: "18", action: "select"}]);
  assert.equal(previews, 1);
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, 1);
  for (const [index, action] of ["up", "down", "left", "right"].entries()) {
    S.reviewKind = "collect";
    send([{id: `live-${index}`, action}]);
    assert.equal(S.reviewKind, "");
    assert.equal(S.collectionSlots.selectedSlotId, "2");
  }
  assert.equal(liveReturns, 4);
  // Navigation and disconnection cancel pending single-click selection.
  send([{id: "19", action: "select"}]);
  send([{id: "20", action: "left"}]);
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, 1);
  send([{id: "21", action: "select"}]);
  resetCollectionReviewInput();
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, 1);
  send([]);
  send([{id: "22", action: "select"}]);
  S.STATUS.collect.collecting = true;
  await new Promise((resolve) => setTimeout(resolve, 330));
  assert.equal(activations, 1);

})().catch((error) => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
