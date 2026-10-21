"""QC feedback must not wait for replay/status polling or enqueue repeated clicks."""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.static


def test_collection_qc_feedback_and_failure_recovery():
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    handlers = source[
        source.index("async function submitEpisodeQc(") : source.index(
            "async function submitEpisodeNote("
        )
    ] + source[
        source.index("function renderCollectionQcControls(") : source.index(
            "function selectedCollectEpisodeItem("
        )
    ]
    script = r'''
const assert = require("node:assert/strict");
const episodeQcPending = new Set();
const episode = {episode_index: 7, qc_verdict: "fail", qc_auto: true,
  qc_note: "machine finding", qc_reason: "joint_limit_exceeded"};
const slot = {episode, qc_state: "failed"};
const collectionQcTarget = {dataset: "set", dataset_dir: "/dataset", task: "task", item: episode};
const S = {collectTaskSelectionPending: false, collectionSlots: {dataset: "set", slots: [slot]},
  episodeHistory: {collect: {collectionSet: "set", episodes: [episode]}}};
const collectionSlotClickTimer = null;
const elements = new Map();
const $ = (id) => {
  if (!elements.has(id)) elements.set(id, {textContent: "", value: "human note", disabled: false});
  return elements.get(id);
};
const selectedCollectEpisodeItem = () => episode;
const savedEpisodeId = (item) => item?.episode_index ?? null;
const reviewNoteFor = () => $("collect-qc-note");
const episodeQcEndpoint = () => "/api/collect_qc_mark";
const collectQcState = (item) => (
  {pass: "passed", fail: "failed", unreviewed: "unreviewed"}[item.qc_verdict]);
const collectResultLabel = collectQcState;
const syncCollectionQcNote = () => {};
const clientTrace = () => {};
const invalidateEpisodeHistory = () => {};
const renderCollect = () => renderCollectionQcControls(true, false);
const pollEpisodeHistory = () => {throw new Error("QC must not wait for history");};
const apiGet = () => {throw new Error("QC must not wait for status");};
let posts = 0, refreshes = 0, resolvePost, rejectPost;
const apiPost = (path, body) => {
  posts++;
  assert.equal(path, "/api/collect_qc_mark");
  assert.equal(body.episode, "7");
  return new Promise((resolve, reject) => {resolvePost = resolve; rejectPost = reject;});
};
const pollCollectionSlots = (force, options) => {
  refreshes++;
  assert.equal(force, true);
  assert.equal(options.preserveView, true);
  return new Promise(() => {});
};
''' + handlers + r'''
(async () => {
  const first = submitEpisodeQc("collect", "pass");
  assert.equal($("collect-qc-status").textContent, "saving…");
  for (const id of ["b-collect-qc-pass", "b-goto-qc",
    "b-collect-qc-unreviewed", "b-collect-note-save"])
    assert.equal($(id).disabled, true);
  assert.equal(await submitEpisodeQc("collect", "fail"), false);
  assert.equal(posts, 1);
  assert.equal(episode.qc_verdict, "fail");
  resolvePost({ok: true});
  let timeout;
  const result = await Promise.race([first, new Promise((resolve, reject) => {
    timeout = setTimeout(() => reject(new Error("QC waited for background refresh")), 500);
  })]);
  clearTimeout(timeout);
  assert.equal(result, true);
  assert.equal(refreshes, 1);
  assert.equal(slot.qc_state, "passed");
  assert.equal(episode.qc_auto, false);
  assert.equal(episode.qc_auto_note, "machine finding");
  assert.equal(episode.qc_note, "human note");
  assert.equal($("b-goto-qc").disabled, false);

  const failure = submitEpisodeQc("collect", "fail");
  resolvePost({ok: false, error: "disk unavailable"});
  assert.equal(await failure, false);
  assert.equal(episode.qc_verdict, "pass");
  assert.match($("collect-qc-status").textContent, /disk unavailable/);
  assert.equal($("b-goto-qc").disabled, false);
  const networkFailure = submitEpisodeQc("collect", "fail");
  rejectPost(new Error("connection lost"));
  assert.equal(await networkFailure, false);
  assert.equal(episodeQcPending.size, 0);
  const retry = submitEpisodeQc("collect", "fail");
  resolvePost({ok: true});
  assert.equal(await retry, true);
  assert.equal(slot.qc_state, "failed");
})().catch((error) => {console.error(error); process.exitCode = 1;});
'''
    subprocess.run(["node", "-e", script], check=True, timeout=10)


def test_qc_slot_refresh_preserves_tiles_even_if_request_fails():
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    handler = source[
        source.index("async function pollCollectionSlots(") : source.index(
            "async function selectCollectionDataset("
        )
    ]
    script = r'''
const assert = require("node:assert/strict");
const tiles = [{slot_id: "selected", episode: {episode_index: 7}}];
const state = {dataset: "set", loaded: true, slots: tiles, active: null};
const S = {collectionSlots: state};
let collectionSlotsController = null, rejectGet;
const renderCollect = () => {};
const setCollectError = () => {};
const t = (key, values) => values.error;
const collectionSlotQuery = () => "/slots";
const apiGet = () => new Promise((resolve, reject) => {rejectGet = reject;});
''' + handler + r'''
(async () => {
  const refresh = pollCollectionSlots(true, {preserveView: true});
  assert.equal(state.slots, tiles);
  assert.equal(state.loading, false);
  rejectGet(new Error("unavailable"));
  await refresh;
  assert.equal(state.slots, tiles);
  assert.equal(state.error, "unavailable");
  const navigation = pollCollectionSlots(true);
  assert.deepEqual(state.slots, []);
  assert.equal(state.loading, true);
  rejectGet(new Error("unavailable"));
  await navigation;
})().catch((error) => {console.error(error); process.exitCode = 1;});
'''
    subprocess.run(["node", "-e", script], check=True, timeout=10)
