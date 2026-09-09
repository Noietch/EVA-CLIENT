// collect.js: data-collection tab (collect) + QC/annotation review &
// stage-video playback control (review).
import { $, LIVE, S, apiGet, apiPost, clientTrace } from "./core.js";
import { updateScrub } from "./charts.js";
import {
  applyCollectTaskSelection, collectTaskIndexValue, collectSetValue, collectTaskValue,
  setPanel, applyStatus, uiMode,
} from "./run.js";
import {
  exitReplayMode, loadReviewPlayback, refreshCameraStreams, replayStop,
} from "./replay.js";
import { setActiveTab } from "./main.js";

// ===== collect =====

const qualityTransfer = {
  phase: "idle",
  exporting: false,
  exportJobId: "",
  exportState: "idle",
  episodesCompleted: 0,
  episodesTotal: 0,
  uploading: false,
  acceptedDir: "",
  uploadJobId: "",
  uploadPlanId: "",
  uploadState: "idle",
  datasetFormat: "",
  filesCompleted: 0,
  filesTotal: 0,
  bytesCompleted: 0,
  bytesTotal: 0,
  localFiles: 0,
  localBytes: 0,
  filesSkipped: 0,
  bytesSkipped: 0,
  newFiles: 0,
  changedFiles: 0,
  filesToDelete: 0,
  filesDeleted: 0,
};

// Saved episode history is review data, not heartbeat data. Fetch only the
// visible collection/RL scope and retain the last complete snapshot locally.
const EPISODE_HISTORY_POLL_MS = 5000;
const EPISODE_HISTORY_PAGE_SIZE = 128;
const COLLECTION_SLOT_POLL_MS = 2000;
let episodeHistoryPolling = false;
let collectionSlotsPolling = false;
let collectItemsRenderKey = "";
let rolloutItemsRenderKey = "";
let collectionSlotClickTimer = null;
let collectionQcTarget = null;

function scenePlanTasks() {
  return (S.SCENE_PLAN && Array.isArray(S.SCENE_PLAN.tasks)) ? S.SCENE_PLAN.tasks : [];
}

function scenePlanTask() {
  return scenePlanTasks().find((task) => task.task_id === S.scenePlanTaskId) || null;
}

function scenePlanScene() {
  const task = scenePlanTask();
  const scenes = (S.SCENE_PLAN && Array.isArray(S.SCENE_PLAN.scenes))
    ? S.SCENE_PLAN.scenes : [];
  if (!task || !task.scene_ids.length) return null;
  const preferredIndex = task.scene_ids.indexOf(S.scenePlanSceneId);
  const index = preferredIndex >= 0
    ? preferredIndex
    : Math.max(0, Math.min(S.scenePlanSceneIndex, task.scene_ids.length - 1));
  S.scenePlanSceneIndex = index;
  S.scenePlanSceneId = task.scene_ids[index];
  const scene = scenes.find((item) => item.scene_id === task.scene_ids[index]);
  return scene
    ? { ...scene, index, target: Number(task.scene_epsiodes_count[index] || 0) }
    : null;
}

function scenePlanStartMetadata() {
  const scene = scenePlanScene();
  const task = scenePlanTask();
  const slot = S.collectionSlots && S.collectionSlots.active;
  if (!task || !scene || !slot || task.prompt_en !== collectTaskValue()) return {};
  return {
    scene_id: scene.scene_id,
    scene_round: S.scenePlanRoundIndex,
    slot_id: slot.slot_id,
    task_id: slot.task_id,
  };
}

function syncScenePlanTask(prompt) {
  if (S.scenePlanTaskPromptKey === prompt) return;
  const linked = scenePlanTasks().find((task) => task.prompt_en === prompt);
  S.scenePlanTaskId = linked ? linked.task_id : "";
  const preferredIndex = linked && S.scenePlanSceneId
    ? linked.scene_ids.indexOf(S.scenePlanSceneId) : -1;
  S.scenePlanSceneIndex = preferredIndex >= 0 ? preferredIndex : 0;
  S.scenePlanSceneId = linked ? linked.scene_ids[S.scenePlanSceneIndex] || "" : "";
  S.scenePlanRoundIndex = 0;
  S.scenePlanTaskPromptKey = prompt;
}

function adoptCollectionSlot(slot) {
  if (!slot) return;
  const task = scenePlanTasks().find((entry) => entry.task_id === slot.task_id);
  S.scenePlanTaskId = slot.task_id;
  S.scenePlanTaskPromptKey = slot.task;
  S.scenePlanSceneId = slot.scene_id;
  S.scenePlanSceneIndex = task ? Math.max(0, task.scene_ids.indexOf(slot.scene_id)) : 0;
  S.scenePlanRoundIndex = Number(slot.round_index) || 0;
  S.scenePlanSelectionManual = false;
}

function collectionSlotQuery(dataset) {
  const state = S.collectionSlots;
  const params = new URLSearchParams({ dataset, page: String(state.page || 1) });
  if (state.sceneFilter) params.set("scene", state.sceneFilter);
  if (state.taskFilter) params.set("task", state.taskFilter);
  if (state.showAll) params.set("all", "1");
  return `/api/collection_slots?${params.toString()}`;
}

function applyCollectionSlotsPayload(payload) {
  const state = S.collectionSlots;
  if (payload.scene_plan) S.SCENE_PLAN = payload.scene_plan;
  state.loaded = true;
  state.dataset = String(payload.dataset || state.dataset || "");
  state.datasetDir = String(payload.dataset_dir || "");
  state.active = payload.active || null;
  state.counts = payload.counts || {};
  state.scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  state.tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
  state.slots = Array.isArray(payload.slots) ? payload.slots : [];
  state.page = Number(payload.page) || 1;
  state.pageCount = Number(payload.page_count) || 1;
  state.filteredTotal = Number(payload.filtered_total) || 0;
  state.viewerActive = !!payload.viewer_active;
  if (state.active) adoptCollectionSlot(state.active);
  if (!state.slots.some((slot) => slot.slot_id === state.selectedSlotId)) {
    state.selectedSlotId = "";
  }
}

function setCollectError(message = "") {
  const node = $("collect-err");
  if (node) node.textContent = message;
}

async function activateCollectionSlot(slot, { manual = false } = {}) {
  if (!slot || S.collectTaskSelectionPending) return false;
  if (S.STATUS.collect && S.STATUS.collect.collecting) return false;
  const state = S.collectionSlots;
  if (manual) state.followActivePage = false;
  const previousActive = state.active;
  const previousSelectedSlotId = state.selectedSlotId;
  const previousStatus = {
    task: S.STATUS.selected_collect_task,
    dataset: S.STATUS.selected_collect_set,
    taskIndex: S.STATUS.selected_collect_task_index,
    slotId: S.STATUS.collection_slot_id,
    taskId: S.STATUS.collection_task_id,
  };
  setCollectError();
  S.collectTaskSelectionPending = true;
  if (slot.task) state.active = { ...slot, state: "active" };
  state.selectedSlotId = "";
  adoptCollectionSlot(state.active);
  renderCollect();
  try {
    const response = await apiPost("/api/select_collection_slot", {
      dataset: slot.dataset,
      slot_id: slot.slot_id,
      manual,
    }, { concurrent: true, timeoutMs: 5000 });
    if (!response.ok || !response.active) {
      setCollectError(response.error || "Unable to select this slot");
      return false;
    }
    const taskApplied = applyCollectTaskSelection(
      response.active.task,
      response.active.dataset,
      Number(response.active.task_index),
      response.active.scene_id
    );
    if (!taskApplied) {
      setCollectError("The selected slot is not present in this dataset");
      return false;
    }
    state.active = response.active;
    state.counts = response.counts || state.counts;
    S.STATUS.collection_slot_id = response.active.slot_id;
    S.STATUS.collection_task_id = response.active.task_id;
    adoptCollectionSlot(response.active);
    return true;
  } catch (error) {
    setCollectError(error.message || "Unable to select this slot");
    return false;
  } finally {
    if (S.STATUS.collection_slot_id !== slot.slot_id) {
      state.active = previousActive;
      state.selectedSlotId = previousSelectedSlotId;
      S.STATUS.selected_collect_task = previousStatus.task;
      S.STATUS.selected_collect_set = previousStatus.dataset;
      S.STATUS.selected_collect_task_index = previousStatus.taskIndex;
      S.STATUS.collection_slot_id = previousStatus.slotId;
      S.STATUS.collection_task_id = previousStatus.taskId;
      if (previousActive) adoptCollectionSlot(previousActive);
    }
    S.collectTaskSelectionPending = false;
    renderCollect();
  }
}

async function pollCollectionSlots(force = false) {
  const state = S.collectionSlots;
  const dataset = state.dataset || collectSetValue();
  if (!dataset || collectionSlotsPolling) return;
  const now = performance.now();
  if (!force && now - Number(state.lastAttemptAt || 0) < COLLECTION_SLOT_POLL_MS) return;
  state.lastAttemptAt = now;
  collectionSlotsPolling = true;
  const previousActiveId = String((state.active && state.active.slot_id) || "");
  try {
    let payload = await apiGet(collectionSlotQuery(dataset));
    if (!payload || payload.ok === false) return;
    const activeChanged = String((payload.active && payload.active.slot_id) || "") !== previousActiveId;
    if (state.followActivePage && activeChanged && payload.active_page &&
        Number(payload.active_page) !== Number(payload.page)) {
      state.page = Number(payload.active_page);
      payload = await apiGet(collectionSlotQuery(dataset));
      if (!payload || payload.ok === false) return;
    }
    // A dataset switch can happen while the previous request is in flight.
    if (dataset !== (state.dataset || collectSetValue()) ||
        collectionSlotClickTimer !== null || S.collectTaskSelectionPending) return;
    applyCollectionSlotsPayload(payload);
    renderCollect();
    const active = state.active;
    const collecting = !!(S.STATUS.collect && S.STATUS.collect.collecting);
    if (active && !collecting && !S.collectTaskSelectionPending &&
        S.STATUS.collection_slot_id !== active.slot_id) {
      queueMicrotask(() => activateCollectionSlot(active));
    }
  } catch {
    // Keep the last complete plan while the low-frequency endpoint recovers.
  } finally {
    collectionSlotsPolling = false;
  }
}

async function selectCollectionDataset(dataset) {
  const value = String(dataset || "").trim();
  if (!value || (S.STATUS.collect && S.STATUS.collect.collecting)) return false;
  const state = S.collectionSlots;
  state.dataset = value;
  state.loaded = false;
  state.active = null;
  state.slots = [];
  state.scenes = [];
  state.tasks = [];
  S.SCENE_PLAN = { tasks: [], scenes: [], positions: [], objects: [] };
  state.page = 1;
  state.sceneFilter = "";
  state.taskFilter = "";
  state.showAll = true;
  state.selectedSlotId = "";
  state.followActivePage = true;
  const select = $("collect-set-list");
  if (select) select.value = value;
  await pollCollectionSlots(true);
  return !!state.active;
}

function changeCollectionSlotFilter(kind, value) {
  const state = S.collectionSlots;
  if (kind === "scene") state.sceneFilter = String(value || "");
  if (kind === "task") state.taskFilter = String(value || "");
  state.showAll = !state.sceneFilter && !state.taskFilter;
  state.page = 1;
  state.selectedSlotId = "";
  state.followActivePage = false;
  pollCollectionSlots(true);
  renderCollect();
}

function toggleCollectionSlotAll() {
  const state = S.collectionSlots;
  state.showAll = !state.showAll;
  state.sceneFilter = "";
  state.taskFilter = "";
  state.selectedSlotId = "";
  state.followActivePage = false;
  state.page = state.showAll && state.active
    ? Math.floor(Number(state.active.ordinal) / 50) + 1 : 1;
  pollCollectionSlots(true);
  renderCollect();
}

function changeCollectionSlotPage(delta) {
  const state = S.collectionSlots;
  const page = Math.max(1, Math.min(state.pageCount, Number(state.page) + delta));
  if (page === state.page) return;
  state.page = page;
  state.selectedSlotId = "";
  state.followActivePage = false;
  pollCollectionSlots(true);
  renderCollect();
}

function scenePlanGridPositions(scene) {
  const configured = S.SCENE_PLAN && Array.isArray(S.SCENE_PLAN.positions)
    ? S.SCENE_PLAN.positions : [];
  const positions = configured.map((position) => ({
    positionId: String(position.position_id || ""),
    x: position.x === null || position.x === "" ? NaN : Number(position.x),
    y: position.y === null || position.y === "" ? NaN : Number(position.y),
  })).filter((position) => position.positionId);
  const known = new Set(positions.map((position) => position.positionId));
  const groups = scene && Array.isArray(scene.placement_groups)
    ? scene.placement_groups : [];
  groups.flatMap((group) => group.position_ids || []).forEach((value) => {
    const positionId = String(value || "");
    if (!positionId || known.has(positionId)) return;
    known.add(positionId);
    positions.push({ positionId, x: NaN, y: NaN });
  });
  return positions;
}

function scenePlanGridGeometry(positions) {
  const count = Math.max(1, positions.length);
  const fallbackColumns = Math.ceil(Math.sqrt(count));
  const fallback = {
    columns: fallbackColumns,
    rows: Math.ceil(count / fallbackColumns),
    cells: positions.map((position, index) => ({
      ...position,
      column: index % fallbackColumns + 1,
      row: Math.floor(index / fallbackColumns) + 1,
    })),
  };
  if (!positions.length || positions.some((position) => (
    !Number.isFinite(position.x) || !Number.isFinite(position.y)
  ))) return fallback;

  const xValues = [...new Set(positions.map((position) => position.x))].sort((a, b) => a - b);
  const yValues = [...new Set(positions.map((position) => position.y))].sort((a, b) => a - b);
  if (xValues.length * yValues.length > positions.length * 2) return fallback;
  return {
    columns: xValues.length,
    rows: yValues.length,
    cells: positions.map((position) => ({
      ...position,
      column: xValues.indexOf(position.x) + 1,
      row: yValues.indexOf(position.y) + 1,
    })),
  };
}

function renderSceneGridCells(host, scene) {
  const geometry = scenePlanGridGeometry(scenePlanGridPositions(scene));
  const bounds = S.SCENE_PLAN && S.SCENE_PLAN.bounds || {};
  const physicalAspect = Number(bounds.width) / Number(bounds.height);
  const gridAspect = geometry.columns / geometry.rows;
  const aspect = Number.isFinite(physicalAspect) && physicalAspect > 0
    ? physicalAspect : Math.max(0.5, Math.min(2, gridAspect));
  const layoutKey = geometry.cells.map((cell) => (
    `${cell.positionId}:${cell.column}:${cell.row}`
  )).join("|");
  host.style.setProperty("--scene-grid-columns", String(geometry.columns));
  host.style.setProperty("--scene-grid-rows", String(geometry.rows));
  host.style.setProperty("--scene-grid-aspect", String(aspect));
  if (host.dataset.layoutKey === layoutKey) return;
  host.dataset.layoutKey = layoutKey;
  host.replaceChildren(...geometry.cells.map((position) => {
    const cell = document.createElement("div");
    const label = document.createElement("span");
    const object = document.createElement("b");
    const photo = document.createElement("img");
    cell.className = "collect-scene-cell empty";
    cell.dataset.positionId = position.positionId;
    cell.style.gridColumn = String(position.column);
    cell.style.gridRow = String(position.row);
    cell.title = position.positionId;
    label.textContent = position.positionId;
    photo.className = "collect-scene-object-photo";
    photo.alt = "";
    photo.hidden = true;
    photo.onerror = () => { photo.hidden = true; };
    cell.append(label, photo, object);
    return cell;
  }));
}

function renderCurrentSceneGrid(scene) {
  const host = $("collect-current-scene-grid");
  if (!host) return;
  renderSceneGridCells(host, scene);
  const byPosition = new Map();
  (scene && scene.placements || []).forEach((placement) => {
    const positionId = String(placement.position_id || "");
    if (!positionId) return;
    const values = byPosition.get(positionId) || [];
    values.push(placement);
    byPosition.set(positionId, values);
  });
  host.querySelectorAll(".collect-scene-cell").forEach((cell) => {
    const placements = byPosition.get(cell.dataset.positionId) || [];
    const names = [...new Set(placements.map((placement) => String(placement.name || "")))]
      .filter(Boolean);
    cell.classList.toggle("empty", placements.length === 0);
    cell.classList.toggle("fixed", placements.length > 0);
    const object = cell.querySelector("b");
    object.textContent = names.join(" / ");
    object.title = names.join(" / ");
    const photo = cell.querySelector("img");
    const photoUrl = placements.find((placement) => placement.photo_url)?.photo_url || "";
    if (photoUrl) {
      if (photo.getAttribute("src") !== photoUrl) {
        photo.hidden = false;
        photo.src = photoUrl;
      }
    } else {
      photo.hidden = true;
      photo.removeAttribute("src");
    }
    cell.title = names.length
      ? `${cell.dataset.positionId} · ${names.join(" / ")}` : cell.dataset.positionId;
  });
}

function itemsForPrompt(items, prompt) {
  return (items || []).filter(
    (item) => String((item && (item.task || item.prompt)) || "") === prompt
  );
}

function collectHistorySelection() {
  return { collectionSet: collectSetValue(), task: collectTaskValue() };
}

function historyFor(scope, status) {
  const cache = S.episodeHistory && S.episodeHistory[scope];
  const statusDir = String((status && status.dataset_dir) || "");
  const selection = scope === "collect" ? collectHistorySelection() : null;
  const cacheMatchesScope = scope === "collect"
    ? cache && cache.collectionSet === selection.collectionSet && cache.task === selection.task
    : cache && (!statusDir || cache.datasetDir === statusDir);
  if (cache && cache.loaded && cacheMatchesScope) {
    const statusQueueMatches = Array.isArray(status && status.queue) &&
      (!statusDir || cache.datasetDir === statusDir);
    const queue = statusQueueMatches ? status.queue : (cache.queue || []);
    return {
      episodes: cache.episodes || [],
      queue: scope === "collect" ? itemsForPrompt(queue, selection.task) : queue,
      summary: cache.summary || episodeItemsSummary(cache.episodes),
    };
  }
  const episodes = scope === "collect"
    ? itemsForPrompt(status && status.episodes, selection.task)
    : (Array.isArray(status && status.episodes) ? status.episodes : []);
  return {
    episodes,
    queue: scope === "collect"
      ? itemsForPrompt(status && status.queue, selection.task)
      : (Array.isArray(status && status.queue) ? status.queue : []),
    summary: episodeItemsSummary(episodes),
  };
}

function episodeItemsSummary(items) {
  const summary = { usable: 0, rejected: 0, pending: 0, signature: "" };
  const signatures = [];
  (items || []).forEach((item) => {
    const outcome = collectOutcome(item);
    if (outcome === "usable") summary.usable += 1;
    else if (outcome === "rejected") summary.rejected += 1;
    else summary.pending += 1;
    signatures.push([
      item && item.episode_index,
      item && item.status,
      item && item.quality,
      item && item.qc_verdict,
      item && item.length,
      item && item.error,
      item && item.quality_issue_count,
      JSON.stringify(((item && item.quality_issues) || []).map((issue) => [
        issue && issue.code,
        issue && issue.count,
      ])),
    ].map((value) => String(value == null ? "" : value)).join(":"));
  });
  summary.signature = signatures.join("|");
  return summary;
}

function appendEpisodeItemsSummary(summary, items) {
  const tail = episodeItemsSummary(items);
  return {
    usable: summary.usable + tail.usable,
    rejected: summary.rejected + tail.rejected,
    pending: summary.pending + tail.pending,
    signature: [summary.signature, tail.signature].filter(Boolean).join("|"),
  };
}

function episodeItemsSignature(items) {
  return episodeItemsSummary(items).signature;
}

function invalidateEpisodeHistory(scope) {
  const cache = S.episodeHistory && S.episodeHistory[scope];
  if (cache) cache.lastAttemptAt = 0;
}

async function fetchEpisodeHistory(scope, since, selection = null, cursor = "") {
  const params = new URLSearchParams({
    scope,
    since: String(Math.max(0, Number(since) || 0)),
    limit: String(EPISODE_HISTORY_PAGE_SIZE),
  });
  if (cursor) params.set("cursor", cursor);
  if (scope === "collect") {
    params.set("set", selection.collectionSet);
    params.set("task", selection.task);
  }
  return apiGet(`/api/episodes?${params.toString()}`, { timeoutMs: 5000 });
}

async function pollEpisodeHistory(force = false) {
  const scope = S.ACTIVE_TAB === "collect" ? "collect"
    : S.ACTIVE_TAB === "rl" ? "rollout" : null;
  if (!scope || document.hidden || episodeHistoryPolling) return;
  const cache = S.episodeHistory && S.episodeHistory[scope];
  if (!cache) return;
  const selection = scope === "collect" ? collectHistorySelection() : null;
  if (scope === "collect" && (!selection.collectionSet || !selection.task)) return;
  const selectionChanged = scope === "collect" && (
    cache.collectionSet !== selection.collectionSet || cache.task !== selection.task
  );
  const now = performance.now();
  if (!force && !selectionChanged &&
      now - Number(cache.lastAttemptAt || 0) < EPISODE_HISTORY_POLL_MS) return;
  cache.lastAttemptAt = now;
  episodeHistoryPolling = true;
  try {
    const statusDir = String((S.STATUS && S.STATUS[scope] && S.STATUS[scope].dataset_dir) || "");
    const datasetChanged = scope !== "collect" && cache.loaded && !!statusDir &&
      !!cache.datasetDir && statusDir !== cache.datasetDir;
    const canContinue = cache.loaded && !datasetChanged && !selectionChanged;
    let nextSince = canContinue ? cache.episodes.length : 0;
    let nextCursor = canContinue ? String(cache.cursor || "") : "";
    const addedEpisodes = [];
    let resetHistory = !canContinue;
    let hasMore = true;
    let payload = null;
    while (hasMore) {
      const requestedSince = nextSince;
      payload = await fetchEpisodeHistory(scope, nextSince, selection, nextCursor);
      if (!payload || payload.ok === false || !Array.isArray(payload.episodes)) return;
      const activeScope = S.ACTIVE_TAB === "collect" ? "collect"
        : S.ACTIVE_TAB === "rl" ? "rollout" : null;
      if (activeScope !== scope || document.hidden) return;
      const currentStatusDir = String(
        (S.STATUS && S.STATUS[scope] && S.STATUS[scope].dataset_dir) || ""
      );
      if (scope !== "collect" && currentStatusDir !== statusDir) return;
      if (scope === "collect") {
        const current = collectHistorySelection();
        if (current.collectionSet !== selection.collectionSet || current.task !== selection.task) {
          return;
        }
        if (payload.set !== selection.collectionSet || payload.task !== selection.task) return;
      }
      if (payload.reset) {
        resetHistory = true;
        addedEpisodes.length = 0;
      }
      addedEpisodes.push(...payload.episodes);
      const loadedLength = resetHistory
        ? addedEpisodes.length : cache.episodes.length + addedEpisodes.length;
      nextSince = Math.max(0, Number(payload.next_since) || loadedLength);
      nextCursor = String(payload.cursor || "");
      hasMore = !!payload.has_more;
      if (hasMore && nextSince <= requestedSince) {
        throw new Error("episode history cursor stalled");
      }
    }
    cache.loaded = true;
    if (resetHistory) {
      cache.episodes = addedEpisodes;
      cache.summary = episodeItemsSummary(addedEpisodes);
    } else if (addedEpisodes.length) {
      cache.episodes.push(...addedEpisodes);
      cache.summary = appendEpisodeItemsSummary(cache.summary, addedEpisodes);
    }
    cache.queue = Array.isArray(payload.queue) ? payload.queue : [];
    cache.datasetDir = String(payload.dataset_dir || statusDir || "");
    if (scope === "collect") {
      cache.collectionSet = selection.collectionSet;
      cache.task = selection.task;
    }
    cache.version = String(payload.version || "");
    cache.cursor = nextCursor;
    if (scope === "collect") {
      renderCollect();
      pollCollectionSlots(force);
    }
    else renderRolloutSave();
  } catch {
    // Keep the last complete snapshot while the low-frequency endpoint recovers.
  } finally {
    episodeHistoryPolling = false;
  }
}

const DATASET_FORMAT_LABELS = {
  lerobot_v21: "LeRobot v2.1",
  lerobot_v3: "LeRobot v3",
  hdf5: "HDF5",
  mcap: "MCAP",
};

function formatTransferBytes(value) {
    const bytes = Math.max(0, Number(value) || 0);
    if (bytes < 1024) return `${Math.round(bytes)} B`;
    const units = ["KiB", "MiB", "GiB", "TiB"];
    let scaled = bytes / 1024;
    let unit = units[0];
    for (let index = 1; index < units.length && scaled >= 1024; index += 1) {
      scaled /= 1024;
      unit = units[index];
    }
    return `${scaled.toFixed(scaled >= 10 ? 1 : 2)} ${unit}`;
  }

function qualityTransferFormatLabel(value) {
  return DATASET_FORMAT_LABELS[value] || String(value || "dataset");
}

function changeCollectionExportFormat() {
  const select = $("collect-export-format");
  if (!select || qualityTransfer.exporting || qualityTransfer.uploading) return;
  if (qualityTransfer.datasetFormat === select.value && !qualityTransfer.acceptedDir) return;

  qualityTransfer.phase = "export";
  qualityTransfer.exportState = "idle";
  qualityTransfer.uploadState = "idle";
  qualityTransfer.acceptedDir = "";
  qualityTransfer.exportJobId = "";
  qualityTransfer.uploadJobId = "";
  qualityTransfer.uploadPlanId = "";
  qualityTransfer.episodesCompleted = 0;
  qualityTransfer.episodesTotal = 0;
  qualityTransfer.filesCompleted = 0;
  qualityTransfer.filesTotal = 0;
  qualityTransfer.bytesCompleted = 0;
  qualityTransfer.bytesTotal = 0;
  qualityTransfer.localFiles = 0;
  qualityTransfer.localBytes = 0;
  qualityTransfer.filesSkipped = 0;
  qualityTransfer.bytesSkipped = 0;
  qualityTransfer.newFiles = 0;
  qualityTransfer.changedFiles = 0;
  qualityTransfer.filesToDelete = 0;
  qualityTransfer.filesDeleted = 0;
  qualityTransfer.datasetFormat = select.value;
  const status = $("collect-quality-status");
  if (status) {
    status.textContent = `${qualityTransferFormatLabel(select.value)} selected · export required before upload`;
  }
  renderCollect();
}

const exportFormatSelect = $("collect-export-format");
if (exportFormatSelect) exportFormatSelect.onchange = changeCollectionExportFormat;

const keyboardControlState = {
  active: new Map(),
  pressed: new Set(),
  holdProgress: {},
  animationFrame: null,
  installed: false,
};

function collectControlsConfig() {
  return (S.CFG && S.CFG.collection && S.CFG.collection.controls) || {
    mode: "keyboard", bindings: {}, groups: [],
  };
}

function controlBinding(action) {
  return (collectControlsConfig().bindings || {})[action] || null;
}

function createSvgElement(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

let controlHintSvgId = 0;

function buildControlHint(host, binding) {
  host.replaceChildren();
  if (!binding) {
    host.hidden = true;
    delete host.dataset.binding;
    return;
  }
  const mode = collectControlsConfig().mode === "vr" ? "gamepad" : "keyboard";
  const key = String(binding.key || "?").toUpperCase();
  const signature = `${mode}:${key}:${binding.gesture || "tap"}`;
  host.hidden = false;
  host.dataset.binding = signature;
  host.dataset.key = mode === "gamepad" ? key : "";
  host.dataset.gesture = String(binding.gesture || "tap");
  host.className = `control-hint ${mode}`;
  host.setAttribute("aria-label", `${key} ${String(binding.gesture || "tap")}`);

  const keycap = document.createElement("span");
  keycap.className = "control-keycap";
  const wide = key.length > 1;
  const circular = mode === "gamepad" && !wide;
  const svg = createSvgElement("svg", {
    viewBox: wide ? "0 0 52 32" : "0 0 32 32",
    "aria-hidden": "true",
  });
  controlHintSvgId += 1;
  const clipId = `control-key-clip-${controlHintSvgId}`;
  const defs = createSvgElement("defs");
  const clip = createSvgElement("clipPath", { id: clipId });
  const shapeName = circular ? "circle" : "rect";
  const shape = circular
    ? { cx: 16, cy: 16, r: 13 }
    : { x: 2, y: 2, width: wide ? 48 : 28, height: 28, rx: 5 };
  clip.appendChild(createSvgElement(shapeName, shape));
  defs.appendChild(clip);
  svg.appendChild(defs);
  svg.appendChild(createSvgElement(shapeName, {
    class: "control-key-track", ...shape, pathLength: 100,
  }));
  svg.appendChild(createSvgElement("rect", {
    class: "control-key-fill", x: 2, y: 2,
    width: wide ? 48 : 28, height: 28,
    "clip-path": `url(#${clipId})`,
  }));
  svg.appendChild(createSvgElement(shapeName, {
    class: "control-key-progress", ...shape, pathLength: 100,
  }));
  const label = document.createElement("span");
  label.className = "control-key-label";
  label.dataset.key = key;
  label.dataset.label = key;
  label.textContent = key;
  keycap.classList.toggle("wide", wide);
  keycap.append(svg, label);

  const gesture = document.createElement("span");
  gesture.className = "control-gesture";
  gesture.textContent = String(binding.gesture || "tap").toUpperCase();
  host.append(keycap, gesture);
}

function controlFeedback() {
  if (collectControlsConfig().mode !== "vr") {
    return {
      pressed: keyboardControlState.pressed,
      holdProgress: keyboardControlState.holdProgress,
    };
  }
  const teleop = S.TELEOP_FEEDBACK || (S.STATUS && S.STATUS.teleop) || {};
  return {
    pressed: new Set(teleop.pressed_controls || []),
    holdProgress: teleop.hold_progress || {},
  };
}

function updateControlHint(host, binding, feedback) {
  if (!host) return;
  if (!binding) {
    if (!host.hidden) buildControlHint(host, null);
    return;
  }
  const mode = collectControlsConfig().mode === "vr" ? "gamepad" : "keyboard";
  const signature = `${mode}:${String(binding.key || "?").toUpperCase()}:${binding.gesture || "tap"}`;
  if (host.dataset.binding !== signature) buildControlHint(host, binding);
  const control = String(binding.control || "");
  const pressed = feedback.pressed.has(control);
  const progress = binding.gesture === "hold"
    ? Math.max(0, Math.min(1, Number(feedback.holdProgress[control] || 0)))
    : 0;
  host.classList.toggle("pressed", pressed);
  host.classList.toggle("holding", pressed && progress > 0);
  host.classList.toggle("complete", pressed && progress >= 1);
  host.style.setProperty("--control-progress", String(progress * 100));
  host.style.setProperty("--control-fill-width", `${progress * 100}%`);
  const label = host.querySelector(".control-key-label");
  if (label) {
    const key = label.dataset.key || String(binding.key || "?").toUpperCase();
    const display = pressed && binding.gesture === "hold"
      ? `${Math.round(progress * 100)}%`
      : key;
    label.textContent = display;
    label.dataset.label = display;
  }
  const gesture = host.querySelector(".control-gesture");
  if (gesture) {
    gesture.textContent = pressed && binding.gesture === "hold"
      ? `${Math.round(progress * 100)}%`
      : String(binding.gesture || "tap").toUpperCase();
  }
}

function buildControlGroups(groups) {
  const host = $("collect-control-groups");
  if (!host) return;
  const signature = JSON.stringify(groups);
  if (host.dataset.groups === signature) return;
  host.dataset.groups = signature;
  host.replaceChildren();
  groups.forEach((group) => {
    const row = document.createElement("div");
    row.className = "collect-control-state";
    row.dataset.group = String(group.id || "");
    const name = document.createElement("span");
    name.className = "collect-state-name";
    name.textContent = String(group.label || group.id || "ARM").toUpperCase();
    const value = document.createElement("span");
    value.className = "collect-state-value";
    value.textContent = "UNAVAILABLE";
    const hint = document.createElement("span");
    hint.className = "control-hint";
    row.append(name, value, hint);
    host.appendChild(row);
  });
}

function renderCollectControls() {
  const config = collectControlsConfig();
  const isVr = config.mode === "vr";
  const groups = isVr ? (config.groups || []) : [];
  const feedback = controlFeedback();
  buildControlGroups(groups);
  updateControlHint($("collect-hint-motion"), controlBinding("motion"), feedback);
  updateControlHint($("collect-hint-record-toggle"), controlBinding("record_toggle"), feedback);
  updateControlHint($("collect-hint-record-cancel"), controlBinding("record_cancel"), feedback);
  updateControlHint($("collect-hint-home"), controlBinding("home"), feedback);

  const teleop = (S.STATUS && S.STATUS.teleop) || {};
  const connected = !!teleop.connected;
  const armEnabled = !!S.collectArmEnabled;
  const authorized = new Set(teleop.authorized_groups || []);
  const engaged = new Set(teleop.engaged_groups || []);
  $("collect-control-groups").querySelectorAll(".collect-control-state").forEach((row) => {
    const group = groups.find((item) => String(item.id) === row.dataset.group);
    if (!group) return;
    let state = "disabled";
    if (!connected) state = "unavailable";
    else if (armEnabled && engaged.has(group.id)) state = "active";
    else if (armEnabled && authorized.has(group.id)) state = "ready";
    const value = row.querySelector(".collect-state-value");
    if (value) value.textContent = state.toUpperCase();
    row.dataset.state = state;
    const armBinding = controlBinding(group.binding) || {};
    updateControlHint(row.querySelector(".control-hint"), {
      ...armBinding,
      control: group.control,
    }, feedback);
  });
}

function keyboardTarget(action) {
  if (action === "motion") return $("collect-arm-enable");
  if (action === "record_toggle") return $("b-collect-toggle");
  if (action === "record_cancel") return $("b-collect-cancel");
  if (action === "home") return $("b-collect-home");
  return null;
}

function triggerKeyboardAction(action) {
  const target = keyboardTarget(action);
  if (!target || target.disabled) return;
  target.click();
}

function keyboardHint(action) {
  if (action === "motion") return $("collect-hint-motion");
  if (action === "record_toggle") return $("collect-hint-record-toggle");
  if (action === "record_cancel") return $("collect-hint-record-cancel");
  if (action === "home") return $("collect-hint-home");
  return null;
}

function updateKeyboardHoldVisual(entry, progress) {
  const host = keyboardHint(entry.action);
  if (!host) return;
  const percent = Math.round(progress * 100);
  host.classList.add("pressed");
  host.classList.toggle("holding", progress > 0);
  host.classList.toggle("complete", progress >= 1);
  host.style.setProperty("--control-progress", String(progress * 100));
  host.style.setProperty("--control-fill-width", `${progress * 100}%`);
  const label = host.querySelector(".control-key-label");
  if (label) {
    const display = `${percent}%`;
    if (label.textContent !== display) label.textContent = display;
    label.dataset.label = display;
  }
  const gesture = host.querySelector(".control-gesture");
  if (gesture && gesture.textContent !== `${percent}%`) gesture.textContent = `${percent}%`;
}

function keyboardShortcutAllowed(event) {
  if (S.ACTIVE_TAB !== "collect" || collectControlsConfig().mode !== "keyboard") return false;
  if (event.altKey || event.ctrlKey || event.metaKey) return false;
  const target = event.target;
  return !(target instanceof HTMLElement && (
    target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)
  ));
}

function resetKeyboardControl(control) {
  keyboardControlState.pressed.delete(control);
  delete keyboardControlState.holdProgress[control];
  renderCollectControls();
}

function animateKeyboardHold(entry) {
  const binding = entry.binding;
  const holdMs = Math.max(1, Number(binding.hold_ms || 1000));
  const progress = Math.min(1, (performance.now() - entry.startedAt) / holdMs);
  keyboardControlState.holdProgress[binding.control] = progress;
  updateKeyboardHoldVisual(entry, progress);
  if (progress >= 1) {
    if (!entry.triggered) {
      entry.triggered = true;
      triggerKeyboardAction(entry.action);
    }
    keyboardControlState.animationFrame = null;
    return;
  }
  keyboardControlState.animationFrame = requestAnimationFrame(() => animateKeyboardHold(entry));
}

function installCollectKeyboardControls() {
  if (keyboardControlState.installed) return;
  keyboardControlState.installed = true;
  window.addEventListener("keydown", (event) => {
    if (!keyboardShortcutAllowed(event) || event.repeat) return;
    const bindings = Object.entries(collectControlsConfig().bindings || {});
    const match = bindings.find(([, binding]) => binding.control === event.code);
    if (!match) return;
    const [action, binding] = match;
    const target = keyboardTarget(action);
    if (!target || target.disabled) return;
    event.preventDefault();
    const entry = { action, binding, startedAt: performance.now(), triggered: false };
    keyboardControlState.active.set(binding.control, entry);
    keyboardControlState.pressed.add(binding.control);
    keyboardControlState.holdProgress[binding.control] = 0;
    renderCollectControls();
    if (binding.gesture === "hold") animateKeyboardHold(entry);
  });
  window.addEventListener("keyup", (event) => {
    const entry = keyboardControlState.active.get(event.code);
    if (!entry) return;
    event.preventDefault();
    keyboardControlState.active.delete(event.code);
    if (entry.binding.gesture !== "hold" && !entry.triggered) {
      triggerKeyboardAction(entry.action);
    }
    if (keyboardControlState.animationFrame !== null) {
      cancelAnimationFrame(keyboardControlState.animationFrame);
      keyboardControlState.animationFrame = null;
    }
    resetKeyboardControl(event.code);
  });
  window.addEventListener("blur", () => {
    keyboardControlState.active.clear();
    keyboardControlState.pressed.clear();
    keyboardControlState.holdProgress = {};
    if (keyboardControlState.animationFrame !== null) {
      cancelAnimationFrame(keyboardControlState.animationFrame);
      keyboardControlState.animationFrame = null;
    }
    renderCollectControls();
  });
}

async function startCollectFromTab() {
    if (collectionSlotClickTimer !== null) {
      S.collectToggleBusy = null;
      return;
    }
    if (S.reviewKind === "collect") {
      returnReviewToLive();
    }
    const task = collectTaskValue();
    const collectionSet = collectSetValue();
    const taskIndex = collectTaskIndexValue();
    const slot = S.collectionSlots && S.collectionSlots.active;
    if (!task || !slot || S.collectTaskSelectionPending) {
      S.collectToggleBusy = null;
      renderCollect();
      return;
    }
    const confirmed = S.STATUS.collection_slot_id === slot.slot_id ||
      await activateCollectionSlot(slot);
    if (!confirmed) {
      S.collectToggleBusy = null;
      renderCollect();
      return;
    }
    S.STATUS.selected_collect_task = task;
    S.STATUS.selected_collect_set = collectionSet;
    S.STATUS.selected_collect_task_index = taskIndex;
    let started;
    try {
      started = await apiPost("/api/collect_start", scenePlanStartMetadata());
    } catch {
      started = null;
    }
    if (!started || !started.ok) {
      S.collectToggleBusy = null;
      renderCollect();
    }
  }

function fmtEta(sec) {
    if (sec == null) return "—";
    const s = Math.max(0, Math.round(Number(sec)));
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  }

function collectConfigured() {
    return !!(S.CFG && S.CFG.collection && S.CFG.collection.enabled);
  }

function collectEnabled() {
    return !!(collectConfigured() && S.STATUS.collect);
  }

function collectOutcome(item) {
    if (item.status === "failed") return "rejected";
    if (item.qc_verdict === "pass") return "usable";
    if (item.qc_verdict === "fail") return "rejected";
    if (item.quality === "red") return "rejected";
    if (savedEpisodeId(item) != null && item.quality === "green") return "usable";
    return "pending";
  }

function collectTone(item) {
    if (item.status === "queued") return "cq-queued";
    if (item.status === "saving") return "cq-busy";
    const outcome = collectOutcome(item);
    if (outcome === "usable") return "cq-ok";
    if (outcome === "rejected") return "cq-fail";
    return "cq-queued";
  }

function threeDigitCount(value) {
    return String(Math.max(0, Number(value) || 0)).padStart(3, "0");
  }

function collectIssueText(item) {
    const issues = item.quality_issues || [];
    if (item.error) return item.error;
    if (item.qc_verdict) return `qc ${item.qc_verdict}`;
    if (!issues.length) return item.status || "ok";
    return issues.map((issue) => (
      `${issue.code || "issue"}${Number(issue.count || 1) > 1 ? ` ×${issue.count}` : ""}`
    )).join(", ");
  }

function savedEpisodeId(item) {
    if (!item || item.status !== "saved") return null;
    const episode = Number(item.episode_index);
    return Number.isFinite(episode) ? episode : null;
  }

function syncScenePlanToEpisode(item) {
    const prompt = String((item && (item.task || item.prompt)) || "");
    const task = scenePlanTasks().find((entry) => (
      entry.task_id === String((item && item.task_id) || "") || entry.prompt_en === prompt
    ));
    if (!task) return;
    S.scenePlanTaskId = task.task_id;
    const sceneIndex = task.scene_ids.indexOf(String((item && item.scene_id) || ""));
    if (sceneIndex >= 0) {
      S.scenePlanSceneIndex = sceneIndex;
      S.scenePlanSceneId = task.scene_ids[sceneIndex];
    }
    const roundIndex = Number(item && item.scene_round);
    if (Number.isInteger(roundIndex) && roundIndex >= 0) {
      S.scenePlanRoundIndex = roundIndex;
    }
    S.scenePlanTaskPromptKey = prompt;
    S.scenePlanSelectionManual = true;
  }

function selectCollectEpisode(item) {
    const episode = savedEpisodeId(item);
    if (episode == null) return;
    selectCollectionQcTarget(item);
    syncScenePlanToEpisode(item);
    S.collectReplayEpisode = episode;
    reviewEpisode("collect", item);
    renderCollect();
  }

function selectRolloutSaveEpisode(item) {
    const episode = savedEpisodeId(item);
    if (episode == null) return;
    S.rolloutSaveEpisode = episode;
    reviewEpisode("rollout", item);
    renderRolloutSave();
  }

function selectCollectEpisodePointer(event, item) {
    event.preventDefault();
    selectCollectEpisode(item);
  }

function selectCollectionQcTarget(item) {
    collectionQcTarget = savedEpisodeId(item) == null ? null : {
      item: { ...item },
      dataset: S.collectionSlots.dataset,
      dataset_dir: reviewDatasetFor("collect"),
      task: String(item.task || item.prompt || collectTaskValue()),
    };
    if ($("collect-qc-note")) $("collect-qc-note").value = item ? item.qc_note || "" : "";
    if ($("collect-qc-status")) $("collect-qc-status").textContent = "";
  }

function selectedCollectEpisodeItem() {
    if (!collectionQcTarget || collectionQcTarget.dataset !== S.collectionSlots.dataset) return null;
    const episode = collectionQcTarget.item.episode_index;
    const slotEpisode = (S.collectionSlots.slots || []).map((slot) => slot.episode).find(
      (item) => savedEpisodeId(item) === episode
    );
    if (slotEpisode) return slotEpisode;
    const collect = S.STATUS.collect || {};
    const history = historyFor("collect", collect);
    const items = history.episodes.concat(history.queue);
    return items.find((item) => savedEpisodeId(item) === episode) || collectionQcTarget.item;
  }

function renderCollectionSlotFilters() {
    const state = S.collectionSlots;
    const scene = $("collect-slot-scene-filter");
    const task = $("collect-slot-task-filter");
    const sceneKey = JSON.stringify(state.scenes || []);
    const taskKey = JSON.stringify(state.tasks || []);
    if (scene.dataset.options !== sceneKey) {
      scene.innerHTML = '<option value="">ALL SCENES</option>';
      (state.scenes || []).forEach((entry) => {
        const option = document.createElement("option");
        option.value = entry.id;
        option.textContent = entry.label;
        scene.appendChild(option);
      });
      scene.dataset.options = sceneKey;
    }
    if (task.dataset.options !== taskKey) {
      task.innerHTML = '<option value="">ALL TASKS</option>';
      (state.tasks || []).forEach((entry) => {
        const option = document.createElement("option");
        option.value = entry.id;
        option.textContent = entry.label;
        task.appendChild(option);
      });
      task.dataset.options = taskKey;
    }
    scene.value = state.sceneFilter || "";
    task.value = state.taskFilter || "";
    scene.disabled = !state.loaded;
    task.disabled = !state.loaded;
  }

function renderCollectTiles(items) {
    const host = $("collect-queue-tiles");
    host.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("span");
      empty.className = "collect-empty";
      empty.textContent = S.collectionSlots.viewerActive ? "NO MATCHING SLOTS" : "FILTER OFF";
      host.appendChild(empty);
      return;
    }
    items.forEach((slot) => {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.title = `${slot.scene_label} · ${slot.task_zh || slot.task} · ` +
        `round ${Number(slot.round_index) + 1}/${slot.round_total}`;
      const episode = savedEpisodeId(slot.episode);
      tile.title = `SLOT ${Number(slot.ordinal) + 1} · ${tile.title}`;
      if (episode != null) {
        tile.title += ` · EPISODE ${episode} · ${String(slot.episode.quality || "unknown").toUpperCase()}`;
        if (slot.episode.qc_verdict) tile.title += ` · QC ${slot.episode.qc_verdict.toUpperCase()}`;
      }
      if (slot.state === "saving") tile.title += " · CONVERTING";
      tile.textContent = String(Number(slot.ordinal) + 1);
      tile.setAttribute("aria-label", tile.title);
      const saved = savedEpisodeId(slot.episode) != null;
      const current = S.collectionSlots.active &&
        S.collectionSlots.active.slot_id === slot.slot_id;
      const rejected = saved && (
        String(slot.episode.quality).toLowerCase() === "red" ||
        String(slot.episode.qc_verdict).toLowerCase() === "fail"
      );
      const visibleState = slot.state === "saving" ? "saving"
        : rejected ? "rejected"
        : saved && String(slot.episode.quality).toLowerCase() === "green" ? "complete"
        : current ? "active" : (slot.state === "active" ? "pending" : slot.state);
      tile.className = `collect-tile slot-${visibleState}` +
        `${current ? " slot-current" : ""}`;
      if (slot.slot_id === S.collectionSlots.selectedSlotId) tile.classList.add("selected");
      const locked = slot.state === "saving" || S.collectTaskSelectionPending ||
        !!(S.STATUS.collect && S.STATUS.collect.collecting);
      tile.disabled = locked || (current && !saved);
      tile.classList.toggle("actionable", !tile.disabled);
      tile.onclick = (event) => {
        clearTimeout(collectionSlotClickTimer);
        if (event.detail > 1) return;
        // Delay selection so a double click can preview without changing the capture target.
        collectionSlotClickTimer = setTimeout(() => {
          collectionSlotClickTimer = null;
          if (slot.dataset !== S.collectionSlots.dataset ||
              S.collectTaskSelectionPending || (S.STATUS.collect && S.STATUS.collect.collecting)) return;
          selectCollectionQcTarget(slot.episode);
          activateCollectionSlot(slot, { manual: true });
        }, 300);
      };
      tile.ondblclick = (event) => {
        event.preventDefault();
        clearTimeout(collectionSlotClickTimer);
        collectionSlotClickTimer = null;
        if (!saved || S.collectTaskSelectionPending ||
            (S.STATUS.collect && S.STATUS.collect.collecting)) return;
        S.collectionSlots.selectedSlotId = slot.slot_id;
        selectCollectEpisode(slot.episode);
      };
      host.appendChild(tile);
    });
  }

function pipeBadge(el, text) {
    if (!el) return;
    const state = String(text || "IDLE").toUpperCase();
    el.textContent = state;
    el.dataset.state = state;
  }

function renderRolloutSaveTiles(items) {
    const host = $("rollout-save-queue-tiles");
    if (!host) return;
    host.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("span");
      empty.className = "collect-empty";
      empty.textContent = "no saved rollouts";
      host.appendChild(empty);
      return;
    }
    items.forEach((item) => {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = `collect-tile ${collectTone(item)}`;
      const episode = savedEpisodeId(item);
      if (episode != null) {
        tile.classList.add("replayable");
        if (episode === S.rolloutSaveEpisode) tile.classList.add("selected");
        tile.onclick = () => selectRolloutSaveEpisode(item);
      }
      host.appendChild(tile);
    });
  }

function renderRolloutSaveList(items) {
    const host = $("rollout-save-queue-list");
    if (!host) return;
    host.style.display = S.rolloutSaveQueueExpanded ? "block" : "none";
    $("rollout-save-queue-toggle").textContent = S.rolloutSaveQueueExpanded ? "COLLAPSE" : "EXPAND";
    host.innerHTML = "";
    if (!items.length) return;
    items.slice().reverse().forEach((item) => {
      const row = document.createElement("div");
      const episode = savedEpisodeId(item);
      row.className = `collect-row ${episode != null ? "replayable" : ""}${episode === S.rolloutSaveEpisode ? " selected" : ""}`;
      const ep = document.createElement("span");
      const frames = document.createElement("span");
      const issue = document.createElement("span");
      ep.textContent = `#${String(item.episode_index).padStart(3, "0")}`;
      frames.textContent = `${item.length || 0}f`;
      issue.className = "issue";
      issue.textContent = collectIssueText(item);
      row.appendChild(ep);
      row.appendChild(frames);
      row.appendChild(issue);
      if (episode != null) row.onclick = () => selectRolloutSaveEpisode(item);
      host.appendChild(row);
    });
  }

function renderRolloutSave() {
    const panel = $("rollout-save-panel");
    if (!panel) return;
    const rollout = S.STATUS.rollout || {};
    const history = historyFor("rollout", rollout);
    const episodes = history.episodes;
    const queue = history.queue;
    const totalItems = episodes.length + queue.length;
    const hideRolloutSave = ["sim", "step"].includes(uiMode(S.STATUS.cli_mode)) &&
      totalItems === 0 && S.reviewKind !== "rollout";
    panel.style.display = hideRolloutSave ? "none" : "";
    if (hideRolloutSave) return;
    const enabled = !!rollout.enabled;
    const saveReady = !!rollout.save_ready;
    const saveBlocked = !!rollout.save_blocked_by_intervention;
    const running = S.STATUS.session_status === "running";
    const progress = Math.max(0, Math.min(1, Number(rollout.progress || 0)));
    const savedComplete = enabled && !saveReady && queue.length === 0 && episodes.length > 0;

    pipeBadge($("rollout-save-pipeline"), enabled ? (rollout.pipeline_state || "IDLE") : "DISABLED");
    $("rollout-save-dir").style.display = savedComplete ? "block" : "none";
    $("rollout-save-dir").textContent = savedComplete ? `saved to ${rollout.dataset_dir || "—"}` : "";
    $("rollout-save-count").textContent = `${episodes.length}/${totalItems}`;
    $("rollout-save-progress-fill").style.width = `${progress * 100}%`;
    $("rollout-save-eta").textContent = fmtEta(rollout.eta_sec);
    const acceptedInterventions = Number(rollout.accepted_intervention_segments || 0);
    const activeInterventionFrames = Number(rollout.active_intervention_frames || 0);
    $("rollout-save-err").textContent = enabled
      ? (saveBlocked
          ? `continue or abandon intervention · active ${activeInterventionFrames}f · accepted ${acceptedInterventions}`
          : (saveReady ? `ready after ${rollout.reason || "stop"}` : ""))
      : "rollout saving is disabled";
    const hasFrames = Number(rollout.current_episode_frames || 0) > 0;
    $("b-rollout-save").disabled = !enabled || (!saveReady && !running) || !hasFrames || saveBlocked;
    $("b-rollout-qc-pass").disabled = !enabled || S.rolloutSaveEpisode == null;
    $("b-rollout-qc-fail").disabled = !enabled || S.rolloutSaveEpisode == null;

    const renderKey = [
      history.summary.signature,
      episodeItemsSignature(queue),
      S.rolloutSaveEpisode == null ? "" : S.rolloutSaveEpisode,
      S.rolloutSaveQueueExpanded ? "expanded" : "collapsed",
    ].join("\n");
    if (renderKey !== rolloutItemsRenderKey) {
      rolloutItemsRenderKey = renderKey;
      const items = episodes.concat(queue);
      renderRolloutSaveTiles(items);
      renderRolloutSaveList(items);
    }

    if (S.rolloutSaveEpisode == null) {
      $("rollout-review-title").textContent = "no rollout selected";
    }
  }

function renderCollectionTransfer(enabled, usableCount, rejectedCount) {
  const exportButton = $("b-collect-quality-export");
  const uploadButton = $("b-collect-quality-upload");
  const exportFormat = $("collect-export-format");
  const selectedFormat = exportFormat ? exportFormat.value : "";
  const selectedExportReady = !!qualityTransfer.acceptedDir &&
    qualityTransfer.datasetFormat === selectedFormat;
  const upload = (S.CFG && S.CFG.collection && S.CFG.collection.upload) || {};
  if (exportButton) {
    exportButton.disabled = !enabled || usableCount + rejectedCount === 0 ||
      qualityTransfer.exporting || qualityTransfer.uploading;
  }
  if (uploadButton) {
    const uploadPlanReady = qualityTransfer.uploadState === "ready" &&
      !!qualityTransfer.uploadPlanId;
    uploadButton.disabled = !upload.configured || !selectedExportReady ||
      qualityTransfer.exporting || qualityTransfer.uploading;
    const backendLabel = (upload.backends || []).map((value) => String(value).toUpperCase());
    const targetLabel = backendLabel.length ? backendLabel.join(" + ") : "TARGET";
    const operationCount = qualityTransfer.filesTotal + qualityTransfer.filesToDelete;
    uploadButton.textContent = uploadPlanReady
      ? `CONFIRM ${targetLabel} · ${operationCount} CHANGES`
      : `SCAN ${targetLabel}`;
  }
  if (exportFormat) {
    exportFormat.disabled = qualityTransfer.exporting || qualityTransfer.uploading;
  }
  const transferProgressBar = $("collect-quality-progress-bar");
  const transferProgressFill = $("collect-quality-progress-fill");
  const transferProgressLabel = $("collect-quality-progress-label");
  const transferProgressDetail = $("collect-quality-progress-detail");
  const showingExport = qualityTransfer.phase !== "upload";
  const formatLabel = qualityTransferFormatLabel(
    qualityTransfer.datasetFormat || selectedFormat
  );
  const transferFraction = showingExport
    ? (qualityTransfer.episodesTotal > 0
        ? qualityTransfer.episodesCompleted / qualityTransfer.episodesTotal
        : (qualityTransfer.exportState === "completed" ? 1 : 0))
    : (qualityTransfer.bytesTotal > 0
        ? qualityTransfer.bytesCompleted / qualityTransfer.bytesTotal
        : (qualityTransfer.filesTotal > 0
            ? qualityTransfer.filesCompleted / qualityTransfer.filesTotal
            : (qualityTransfer.uploadState === "completed" ? 1 : 0)));
  const transferPercent = Math.round(Math.max(0, Math.min(1, transferFraction)) * 100);
  if (transferProgressBar) {
    transferProgressBar.setAttribute("aria-valuenow", String(transferPercent));
    transferProgressBar.setAttribute(
      "aria-label", `${formatLabel} ${showingExport ? "export" : "upload"} progress`
    );
  }
  if (transferProgressFill) transferProgressFill.style.width = `${transferPercent}%`;
  if (transferProgressLabel) transferProgressLabel.textContent = `${transferPercent}%`;
  if (transferProgressDetail) {
    transferProgressDetail.textContent = showingExport
      ? `${qualityTransfer.episodesCompleted}/${qualityTransfer.episodesTotal} episodes · ${formatLabel}`
      : (qualityTransfer.uploadState === "ready"
          ? (`${qualityTransfer.filesTotal} upload · ${qualityTransfer.filesToDelete} remove · ` +
            `${qualityTransfer.filesSkipped} same · ` +
            `${qualityTransfer.localFiles} local · ${formatLabel}`)
      : (`${qualityTransfer.filesCompleted}/${qualityTransfer.filesTotal} files · ` +
        `${formatTransferBytes(qualityTransfer.bytesCompleted)}/` +
        `${formatTransferBytes(qualityTransfer.bytesTotal)} · ${formatLabel}`));
  }
}

function renderCollectionReplayStatus(selectedEpisodeSaved) {
  const replayStatus = $("collect-replay-status");
  const history = historyFor("collect", S.STATUS.collect || {});
  const item = history.episodes.find((entry) => savedEpisodeId(entry) === S.collectReplayEpisode);
  const label = `episode ${S.collectReplayEpisode} · ${String((item && item.quality) || "unknown").toUpperCase()}`;
  if (S.reviewKind === "collect" && LIVE.replayOwner === "collect") {
    replayStatus.textContent = LIVE.replayError
      ? `${label} · error · ${LIVE.replayError}`
      : (LIVE.replayLoading
          ? `${label} · loading`
          : `${label} · review`);
    replayStatus.style.display = S.ACTIVE_TAB === "collect" ? "" : "none";
  } else {
    replayStatus.textContent = "";
    replayStatus.style.display = "none";
  }
}

function renderCollect() {
    if (!$("collect-control-col")) return;
    const collect = S.STATUS.collect || {};
    const history = historyFor("collect", collect);
    const slotPlan = S.collectionSlots;
    const enabled = collectEnabled();
    const collecting = !!collect.collecting;
    // The toggle's action depends on the polled `collecting` flag, which lags the
    // click by up to a poll interval + round trip. While that catches up, keep the
    // button held so a second click can't re-fire start/stop on stale state.
    if (S.collectToggleBusy !== null && collecting === S.collectToggleBusy) {
      S.collectToggleBusy = null;
    }
    const toggleBusy = S.collectToggleBusy !== null;
    const activeSlot = slotPlan.active;
    const prompt = activeSlot ? activeSlot.task : collectTaskValue();
    const collectionSet = slotPlan.dataset || collectSetValue();
    const hasPrompt = !!activeSlot;
    const queueFull = collect.pipeline_state === "QUEUE_FULL";
    const episodes = history.episodes;
    const queue = history.queue;
    const counts = slotPlan.counts || {};
    const totalSlots = Number(counts.total) || 0;
    const usableCount = Number(counts.complete) || 0;
    const rejectedCount = (Number(counts.rejected) || 0) + (Number(counts.deferred) || 0);
    const pendingCount = Math.max(0, totalSlots - usableCount - rejectedCount);
    const progress = totalSlots > 0 ? usableCount / totalSlots : 0;
    const requirementComplete = totalSlots > 0 && usableCount >= totalSlots;

    if (activeSlot) adoptCollectionSlot(activeSlot);
    const collectFps = S.CFG && S.CFG.collection ? S.CFG.collection.fps : null;
    $("collect-fps").textContent = collectFps ? `${collectFps} FPS` : "";
    $("collect-count").textContent = `${usableCount}/${totalSlots}`;
    $("collect-usable-count").textContent = threeDigitCount(usableCount);
    $("collect-rejected-count").textContent = threeDigitCount(rejectedCount);
    $("collect-pending-count").textContent = threeDigitCount(pendingCount);
    $("collect-requirement-count").textContent = `${usableCount} / ${totalSlots || "--"}`;
    $("collect-requirement-status").textContent = requirementComplete
      ? "COMPLETE" : `${Math.max(0, totalSlots - usableCount)} REMAINING`;
    $("collect-requirement").classList.toggle("complete", requirementComplete);
    $("collect-requirement").classList.toggle("unset", totalSlots === 0);
    $("collect-progress-label").textContent = `${Math.round(progress * 100)}%`;
    $("collect-progress-fill").style.width = `${progress * 100}%`;
    $("collect-eta").textContent = fmtEta(collect.eta_sec);

    $("collect-current-position").textContent = activeSlot
      ? `${Number(activeSlot.ordinal) + 1} / ${totalSlots}` : `${totalSlots} / ${totalSlots}`;
    renderCurrentSceneGrid(scenePlanScene());
    $("collect-current-task").textContent = activeSlot
      ? (activeSlot.task_zh || activeSlot.task) : "--";
    $("collect-current-task-en").textContent = activeSlot && activeSlot.task_zh
      ? activeSlot.task : "";
    $("collect-current-round").textContent = activeSlot
      ? `ROUND ${Number(activeSlot.round_index) + 1} / ${activeSlot.round_total}` : "ROUND -- / --";
    renderCollectionSlotFilters();
    const allButton = $("collect-queue-toggle");
    allButton.setAttribute("aria-pressed", slotPlan.showAll ? "true" : "false");
    const page = $("collect-slot-page");
    page.hidden = !slotPlan.viewerActive || slotPlan.pageCount <= 1;
    $("collect-slot-page-label").textContent = `${slotPlan.page} / ${slotPlan.pageCount}`;
    $("b-collect-slot-prev").disabled = slotPlan.page <= 1;
    $("b-collect-slot-next").disabled = slotPlan.page >= slotPlan.pageCount;

    const armSwitch = $("collect-arm-enable");
    if (armSwitch) {
      armSwitch.checked = S.collectArmEnabled;
      armSwitch.disabled = !enabled || (!hasPrompt && !S.collectArmEnabled);
      const gate = armSwitch.closest(".collect-arm-gate");
      if (gate) {
        gate.classList.toggle("on", S.collectArmEnabled);
        gate.classList.toggle("disabled", armSwitch.disabled);
      }
    }
    const armLabel = $("collect-arm-label");
    if (armLabel) armLabel.textContent = S.collectArmEnabled ? "ENABLED" : "LOCKED";

    const toggle = $("b-collect-toggle");
    toggle.disabled = toggleBusy || S.collectTaskSelectionPending ||
      (collecting ? false : (!enabled || !hasPrompt || queueFull || !S.collectArmEnabled));
    toggle.classList.toggle("recording", collecting);
    toggle.classList.toggle("primary", !collecting);
    toggle.querySelector(".rec-label").textContent = collecting ? "END / SAVE" : "START RECORD";
    $("b-collect-cancel").disabled = !collecting;
    const home = $("b-collect-home");
    if (home) {
      home.disabled = S.collectHomeBusy || S.collectArmEnabled || collecting || !!S.STATUS.setup_stage;
    }
    const selectedEpisode = selectedCollectEpisodeItem();
    const selectedEpisodeSaved = savedEpisodeId(selectedEpisode) != null;
    const qcPending = S.collectTaskSelectionPending || collectionSlotClickTimer !== null;
    $("collect-qc-target").textContent = selectedEpisodeSaved
      ? `EPISODE ${selectedEpisode.episode_index} · ${String(selectedEpisode.qc_verdict || selectedEpisode.quality || "unknown").toUpperCase()}` : "--";
    $("b-collect-qc-pass").disabled = !enabled || !selectedEpisodeSaved || qcPending;
    $("b-goto-qc").disabled = !enabled || !selectedEpisodeSaved || qcPending;
    $("b-collect-note-save").disabled = !selectedEpisodeSaved || qcPending;
    renderCollectionTransfer(enabled, usableCount, rejectedCount);

    const recordState = collecting || (hasPrompt && !S.collectArmEnabled)
      ? "active"
      : (hasPrompt ? "done" : "pending");
    setPanel("collect-panel-task", enabled && hasPrompt ? "done" : "active");
    setPanel("collect-panel-record", recordState);
    setPanel("collect-panel-queue", slotPlan.viewerActive ? "active" : (slotPlan.loaded ? "done" : "pending"));
    setPanel("collect-panel-replay", selectedEpisodeSaved || episodes.length ? "active" : "pending");

    const renderKey = [
      collectionSet,
      activeSlot ? activeSlot.slot_id : "",
      slotPlan.page,
      slotPlan.sceneFilter,
      slotPlan.taskFilter,
      slotPlan.showAll,
      slotPlan.selectedSlotId,
      S.collectTaskSelectionPending,
      JSON.stringify(slotPlan.slots || []),
    ].join("\n");
    if (renderKey !== collectItemsRenderKey) {
      collectItemsRenderKey = renderKey;
      renderCollectTiles(slotPlan.slots || []);
    }
    renderCollectControls();
    if (collectionSet && !collectionSlotsPolling) {
      queueMicrotask(() => pollCollectionSlots(!slotPlan.loaded));
    }

    renderCollectionReplayStatus(selectedEpisodeSaved);
  }

async function exportCollectionQuality() {
    if (qualityTransfer.exporting || qualityTransfer.uploading) return;
    const status = $("collect-quality-status");
    const datasetFormat = $("collect-export-format").value;
    const formatLabel = qualityTransferFormatLabel(datasetFormat);
    qualityTransfer.phase = "export";
    qualityTransfer.exporting = true;
    qualityTransfer.exportJobId = "";
    qualityTransfer.exportState = "queued";
    qualityTransfer.episodesCompleted = 0;
    qualityTransfer.episodesTotal = 0;
    qualityTransfer.acceptedDir = "";
    qualityTransfer.uploadPlanId = "";
    qualityTransfer.datasetFormat = datasetFormat;
    if (status) status.textContent = `exporting ${formatLabel}…`;
    renderCollect();
    try {
      const result = await apiPost("/api/collect_quality_export", {
        task: collectTaskValue(),
        dataset: S.collectionSlots.dataset || collectSetValue(),
        dataset_format: datasetFormat,
      }, { concurrent: true });
      if (!result.ok) {
        qualityTransfer.exportState = "failed";
        if (status) status.textContent = `✗ ${formatLabel} export · ${result.error || "request failed"}`;
        return;
      }
      qualityTransfer.exportJobId = result.job_id || "";
      while (qualityTransfer.exportJobId === result.job_id) {
        const job = await apiGet(
          `/api/collect_quality_export?job_id=${encodeURIComponent(result.job_id)}`
        );
        if (!job.ok) throw new Error(job.error || "export status unavailable");
        qualityTransfer.exportState = job.state || "running";
        qualityTransfer.episodesCompleted = Number(job.episodes_completed || 0);
        qualityTransfer.episodesTotal = Number(job.episodes_total || 0);
        renderCollect();
        if (job.state === "completed") {
          qualityTransfer.acceptedDir = job.accepted_dir || "";
          qualityTransfer.datasetFormat = job.dataset_format || datasetFormat;
          if (status) {
            const completedFormat = qualityTransferFormatLabel(qualityTransfer.datasetFormat);
            status.textContent = `${completedFormat} export complete · ` +
              `${job.accepted_episodes || 0} accepted · ${job.rejected_episodes || 0} rejected`;
          }
          return;
        }
        if (job.state === "failed") {
          if (status) status.textContent = `✗ ${formatLabel} export · ${job.error || "job failed"}`;
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
    } catch (error) {
      qualityTransfer.exportState = "failed";
      const message = error instanceof Error ? error.message : String(error);
      if (status) status.textContent = `✗ ${formatLabel} export · ${message}`;
    } finally {
      qualityTransfer.exporting = false;
      renderCollect();
    }
  }

async function uploadCollectionQuality() {
    if (qualityTransfer.exporting || qualityTransfer.uploading) return;
    const status = $("collect-quality-status");
    const selectedFormat = $("collect-export-format").value;
    const datasetFormat = qualityTransfer.datasetFormat;
    if (!qualityTransfer.acceptedDir || datasetFormat !== selectedFormat) {
      if (status) {
        status.textContent = `${qualityTransferFormatLabel(selectedFormat)} export required before upload`;
      }
      renderCollect();
      return;
    }
    const formatLabel = qualityTransferFormatLabel(datasetFormat);
    const confirmed = qualityTransfer.uploadState === "ready" &&
      !!qualityTransfer.uploadPlanId;
    const planId = confirmed ? qualityTransfer.uploadPlanId : "";
    qualityTransfer.phase = "upload";
    qualityTransfer.uploading = true;
    qualityTransfer.uploadJobId = "";
    qualityTransfer.uploadState = "queued";
    qualityTransfer.filesCompleted = 0;
    qualityTransfer.bytesCompleted = 0;
    if (!confirmed) {
      qualityTransfer.uploadPlanId = "";
      qualityTransfer.filesTotal = 0;
      qualityTransfer.bytesTotal = 0;
      qualityTransfer.localFiles = 0;
      qualityTransfer.localBytes = 0;
      qualityTransfer.filesSkipped = 0;
      qualityTransfer.bytesSkipped = 0;
      qualityTransfer.newFiles = 0;
      qualityTransfer.changedFiles = 0;
      qualityTransfer.filesToDelete = 0;
      qualityTransfer.filesDeleted = 0;
    }
    if (status) {
      status.textContent = confirmed
        ? `uploading ${formatLabel} accepted export…`
        : `scanning ${formatLabel} accepted export…`;
    }
    renderCollect();
    try {
      const result = await apiPost("/api/collect_quality_upload", {
        dataset: S.collectionSlots.dataset || collectSetValue(),
        task: collectTaskValue(),
        dataset_format: datasetFormat,
        confirmed,
        plan_id: planId,
      }, { concurrent: true });
      if (!result.ok) {
        qualityTransfer.uploadState = "failed";
        if (status) status.textContent = `✗ ${formatLabel} upload · ${result.error || "request failed"}`;
        return;
      }
      qualityTransfer.uploadJobId = result.job_id || "";
      while (qualityTransfer.uploadJobId === result.job_id) {
        const job = await apiGet(
          `/api/collect_quality_upload?job_id=${encodeURIComponent(result.job_id)}`
        );
        if (!job.ok) throw new Error(job.error || "upload status unavailable");
        qualityTransfer.uploadState = job.state || "running";
        qualityTransfer.filesCompleted = Number(job.files_completed || 0);
        qualityTransfer.filesTotal = Number(job.files_total || 0);
        qualityTransfer.bytesCompleted = Number(job.bytes_completed || 0);
        qualityTransfer.bytesTotal = Number(job.bytes_total || 0);
        qualityTransfer.localFiles = Number(job.local_files || 0);
        qualityTransfer.localBytes = Number(job.local_bytes || 0);
        qualityTransfer.filesSkipped = Number(job.files_skipped || 0);
        qualityTransfer.bytesSkipped = Number(job.bytes_skipped || 0);
        qualityTransfer.newFiles = Number(job.new_files || 0);
        qualityTransfer.changedFiles = Number(job.changed_files || 0);
        qualityTransfer.filesToDelete = Number(job.files_to_delete || 0);
        qualityTransfer.filesDeleted = Number(job.files_deleted || 0);
        renderCollect();
        if (job.state === "ready") {
          const operationCount = qualityTransfer.filesTotal + qualityTransfer.filesToDelete;
          qualityTransfer.uploadPlanId = operationCount > 0
            ? job.job_id || result.job_id || ""
            : "";
          if (status) {
            status.textContent = operationCount > 0
              ? `${formatLabel} scan complete · ${qualityTransfer.newFiles} new · ` +
                `${qualityTransfer.changedFiles} changed · ` +
                `${qualityTransfer.filesToDelete} remove · ` +
                `${qualityTransfer.filesSkipped} same`
              : `${formatLabel} scan complete · all ${qualityTransfer.filesSkipped} files same`;
          }
          return;
        }
        if (job.state === "completed") {
          if (status) {
            const remoteDir = job.remote_dir || "remote target";
            status.textContent = job.skipped
              ? `${formatLabel} accepted upload skipped · all files same · ${remoteDir}`
              : `${formatLabel} accepted upload complete · ` +
                `${job.files_completed || 0} uploaded · ` +
                `${job.files_deleted || 0} removed · ` +
                `${job.files_skipped || 0} same · ${remoteDir}`;
          }
          return;
        }
        if (job.state === "failed") {
          qualityTransfer.uploadPlanId = "";
          if (status) status.textContent = `✗ ${formatLabel} upload · ${job.error || "job failed"}`;
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
    } catch (error) {
      qualityTransfer.uploadState = "failed";
      const message = error instanceof Error ? error.message : String(error);
      if (status) status.textContent = `✗ ${formatLabel} upload · ${message}`;
    } finally {
      qualityTransfer.uploading = false;
      renderCollect();
    }
  }

function dotClass(kind) { return "dot " + kind; }

// ===== review =====

let reviewDatasetDir = "";

let reviewTask = "";

let reviewCollectionSet = "";

let reviewEpisodeId = null;

let reviewRequestId = 0;

function reviewDatasetFor(kind) {
    if (kind === "rollout") return (S.STATUS.rollout || {}).dataset_dir || "";
    return (S.collectionSlots && S.collectionSlots.datasetDir) ||
      (S.STATUS.collect || {}).dataset_dir ||
      (S.CFG && S.CFG.collection ? (S.CFG.collection.dataset_dir || "") : "");
  }

function reviewTitleFor(kind) {
    if (kind === "rollout") return $("rollout-review-title");
    return $("collect-replay-status");
  }

function reviewErrorFor(kind) {
    if (kind === "rollout") return $("rollout-save-err");
    return $("collect-err");
  }

function reviewNoteFor(kind) {
    if (kind === "rollout") return $("rollout-qc-note");
    return $("collect-qc-note");
  }

function episodeQcEndpoint(kind) {
    return kind === "collect" ? "/api/collect_qc_mark" : "/api/qc_mark";
  }

function collectReviewMatchesSelection() {
    return reviewCollectionSet === (S.collectionSlots.dataset || collectSetValue());
  }

function reviewActiveInCurrentTab() {
    return (S.reviewKind === "collect" && S.ACTIVE_TAB === "collect") ||
      (S.reviewKind === "rollout" && S.ACTIVE_TAB === "debug");
  }

function clearReviewPlayback() {
    reviewRequestId += 1;
    S.reviewKind = "";
    reviewDatasetDir = "";
    reviewTask = "";
    reviewCollectionSet = "";
    reviewEpisodeId = null;
  }

function showReviewError(kind, message) {
    LIVE.replayOwner = kind;
    LIVE.replayMode = true;
    LIVE.replayLoading = false;
    LIVE.replayError = message;
    replayStop();
    const title = reviewTitleFor(kind);
    const error = reviewErrorFor(kind);
    if (title) title.textContent = `episode ${reviewEpisodeId} · error`;
    if (error) error.textContent = message;
    updateScrub();
  }

function returnReviewToLive() {
    S.collectReplayEpisode = null;
    S.rolloutSaveEpisode = null;
    S.rlQcEpisode = null;
    clearReviewPlayback();
    exitReplayMode();
    refreshCameraStreams();
    renderCollect();
  }

function reviewEpisode(kind, item) {
    return kind === "rollout" ? reviewRolloutEpisode(item) : reviewCollectEpisode(item);
  }

async function reviewCollectEpisode(item) {
    const episode = savedEpisodeId(item);
    if (episode == null) return;
    const requestId = ++reviewRequestId;
    clientTrace("review.collect.select", {
      episode,
      request_id: requestId,
      dataset_dir: reviewDatasetFor("collect"),
    });
    exitReplayMode();
    S.collectReplayEpisode = episode;
    reviewTask = String(item.task || item.prompt || collectTaskValue());
    reviewCollectionSet = S.collectionSlots.dataset || collectSetValue();
    S.reviewKind = "collect";
    reviewDatasetDir = reviewDatasetFor("collect");
    reviewEpisodeId = episode;
    LIVE.replayOwner = "collect";
    LIVE.replayMode = true;
    LIVE.replayLoading = true;
    LIVE.replayError = "";
    updateScrub();
    const title = reviewTitleFor("collect");
    const qualityLabel = String(item.quality || "unknown").toUpperCase();
    if (title) title.textContent = `episode ${episode} · ${qualityLabel} · loading`;
    const err = reviewErrorFor("collect");
    if (err) err.textContent = "";
    const r = await apiPost("/api/review_episode",
      {
        dataset_dir: reviewDatasetDir,
        episode: String(reviewEpisodeId),
      },
      { timeoutMs: 0 },
    );
    if (requestId !== reviewRequestId) return;
    if (!r.ok) {
      clientTrace("review.collect.error", { episode, request_id: requestId, error: r.error || "review failed" });
      showReviewError("collect", r.error || "review failed");
      return;
    }
    const started = await loadReviewPlayback({ ...r, dataset_dir: reviewDatasetDir }, "collect");
    if (requestId !== reviewRequestId) return;
    if (!started) {
      clientTrace("review.collect.error", {
        episode, request_id: requestId, error: LIVE.replayError || "review playback failed",
      });
      showReviewError("collect", LIVE.replayError || "review playback failed");
      return;
    }
    clientTrace("review.collect.ready", { episode, request_id: requestId, frames: LIVE.n });
    if (title) title.textContent = `episode ${episode} · ${qualityLabel} · review`;
  }

async function reviewRolloutEpisode(item) {
    const episode = savedEpisodeId(item);
    if (episode == null) return;
    const requestId = ++reviewRequestId;
    exitReplayMode();
    S.rolloutSaveEpisode = episode;
    S.reviewKind = "rollout";
    reviewTask = "";
    reviewCollectionSet = "";
    reviewDatasetDir = reviewDatasetFor("rollout");
    reviewEpisodeId = episode;
    LIVE.replayOwner = "rollout";
    LIVE.replayMode = true;
    LIVE.replayLoading = true;
    LIVE.replayError = "";
    updateScrub();
    const title = reviewTitleFor("rollout");
    const err = reviewErrorFor("rollout");
    if (title) title.textContent = `episode ${episode} · loading`;
    if (err) err.textContent = "";
    const r = await apiPost("/api/review_episode",
      {
        dataset_dir: reviewDatasetDir,
        episode: String(reviewEpisodeId),
      },
      { timeoutMs: 0 },
    );
    if (requestId !== reviewRequestId) return;
    if (!r.ok) {
      showReviewError("rollout", r.error || "review failed");
      return;
    }
    const started = await loadReviewPlayback({ ...r, dataset_dir: reviewDatasetDir }, "rollout");
    if (requestId !== reviewRequestId) return;
    if (!started) {
      showReviewError("rollout", LIVE.replayError || "review playback failed");
      return;
    }
    if (title) title.textContent = `episode ${episode} · review`;
  }

async function submitEpisodeQc(kind, verdict) {
    if (kind === "collect" && (collectionSlotClickTimer !== null || S.collectTaskSelectionPending)) return false;
    const target = kind === "collect" && selectedCollectEpisodeItem() ? collectionQcTarget : null;
    const episode = kind === "rollout" ? S.rolloutSaveEpisode : target && target.item.episode_index;
    if (episode == null) return false;
    const datasetDir = target ? target.dataset_dir : reviewDatasetFor(kind);
    clientTrace("review.qc.begin", { kind, episode, verdict, dataset_dir: datasetDir });
    const r = await apiPost(episodeQcEndpoint(kind), {
      dataset_dir: datasetDir,
      task: target ? target.task : reviewTask,
      dataset: target ? target.dataset : "",
      episode: String(episode),
      verdict,
      note: reviewNoteFor(kind).value || "",
    });
    const title = kind === "collect" ? $("collect-qc-target") : reviewTitleFor(kind);
    const status = kind === "rollout" ? $("rollout-save-err") : $("collect-qc-status");
    clientTrace("review.qc.end", {
      kind, episode, verdict, ok: !!r.ok, error: r.error || "",
    });
    if (!r.ok) {
      if (title) title.textContent = `episode ${episode} · QC failed`;
      if (status) status.textContent = `✗ ${r.error || "QC failed"}`;
      return false;
    }
    if (target) {
      const patch = (item) => {
        if (item && item.episode_index === episode) item.qc_verdict = verdict;
      };
      patch(target.item);
      if (target.dataset === S.collectionSlots.dataset) {
        (S.collectionSlots.slots || []).forEach((slot) => patch(slot.episode));
        const cache = S.episodeHistory && S.episodeHistory.collect;
        if (cache && cache.collectionSet === target.dataset) (cache.episodes || []).forEach(patch);
        renderCollect();
      }
    }
    if (title) title.textContent = `episode ${episode} · ${verdict}`;
    if (status) status.textContent = `episode ${episode} marked ${verdict}`;
    invalidateEpisodeHistory(kind);
    await pollEpisodeHistory(true);
    applyStatus(await apiGet("/api/status"));
    if (kind === "collect") await pollCollectionSlots(true);
    return true;
  }

async function submitEpisodeNote(kind) {
    if (kind === "collect" && (collectionSlotClickTimer !== null || S.collectTaskSelectionPending)) return;
    const target = kind === "collect" && selectedCollectEpisodeItem() ? collectionQcTarget : null;
    const episode = kind === "rollout" ? S.rolloutSaveEpisode : target && target.item.episode_index;
    const status = kind === "rollout" ? $("rollout-save-err") : $("collect-qc-status");
    if (episode == null) {
      if (status) status.textContent = "✗ select an episode first";
      return;
    }
    if (status) status.textContent = "saving…";
    const r = await apiPost(episodeQcEndpoint(kind), {
      dataset_dir: target ? target.dataset_dir : reviewDatasetFor(kind),
      task: target ? target.task : reviewTask,
      dataset: target ? target.dataset : "",
      episode: String(episode),
      verdict: "",
      note: reviewNoteFor(kind).value || "",
    });
    if (status) status.textContent = r.ok ? `episode ${episode} note saved` : `✗ ${r.error || "save failed"}`;
  }

async function submitQc(verdict) {
    if (S.qcEpisode == null) { $("replay-qc-status").textContent = "✗ load an episode first"; return; }
    const dir = ($("replay-dataset-input").value || "").trim();
    const ep = S.qcEpisode;
    const r = await apiPost("/api/qc_mark", {
      dataset_dir: dir, episode: ep, verdict, note: $("replay-qc-note").value || "",
    });
    const label = verdict === "pass" ? "PASS" : "FAIL";
    $("replay-qc-status").textContent = r.ok ? `episode ${ep} marked ${label}` : `✗ ${r.error || "mark failed"}`;
  }

async function saveAnnotation() {
    if (S.qcEpisode == null) { $("replay-anno-status").textContent = "✗ load an episode first"; return; }
    const dir = ($("replay-dataset-input").value || "").trim();
    const ep = S.qcEpisode;
    $("replay-anno-status").textContent = "saving…";
    const r = await apiPost(
      "/api/annotate",
      { dataset_dir: dir, episode: ep, annotation: $("replay-anno-text").value || "" },
      { timeoutMs: 0 },
    );
    $("replay-anno-status").textContent = r.ok ? `episode ${ep} annotation saved` : `✗ ${r.error || "save failed"}`;
  }

async function loadAnnotation(dir, ep) {
    const r = await apiPost("/api/episode_annotation", { dataset_dir: dir, episode: ep });
    $("replay-anno-text").value = (r && r.annotation) || "";
    $("replay-anno-status").textContent = r && r.annotation ? "loaded existing annotation" : "no annotation yet";
  }

function openBatchQc(dir, episode) {
    const datasetDir = (dir || "").trim();
    if (!datasetDir) return;
    const episodeId = Math.max(0, Math.trunc(Number(episode) || 0));
    S.pendingQcLoad = { dir: datasetDir, episode: episodeId };
    S.qcMode = true;
    apiPost("/api/tab_switch", { tab: "replay" });
    // setActiveTab → renderReplayConfig picks up pendingQcLoad and inspect+loads.
    setActiveTab("replay");
    $("replay-qc-note").value = "";
    $("replay-qc-status").textContent = `loading QC episode ${episodeId}…`;
  }

export {
  collectConfigured, collectEnabled, dotClass, renderCollect,
  renderRolloutSave, returnReviewToLive, savedEpisodeId, startCollectFromTab,
  clearReviewPlayback, loadAnnotation, reviewActiveInCurrentTab, reviewEpisode,
  exportCollectionQuality, saveAnnotation, submitEpisodeNote, submitEpisodeQc, submitQc,
  installCollectKeyboardControls, renderCollectControls, uploadCollectionQuality,
  changeCollectionExportFormat, invalidateEpisodeHistory, pollEpisodeHistory,
  pollCollectionSlots, selectCollectionDataset,
  changeCollectionSlotFilter, changeCollectionSlotPage, toggleCollectionSlotAll,
};
