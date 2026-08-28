// collect.js: data-collection tab (collect) + QC/annotation review &
// stage-video playback control (review).
import { $, LIVE, S, apiGet, apiPost, clientTrace } from "./core.js";
import { updateScrub } from "./charts.js";
import {
  advanceCollectTask, collectTaskIndexValue, collectSetValue, collectTaskSelectionKey,
  collectTaskTarget, collectTaskValue, setPanel, applyStatus, uiMode,
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
  uploadState: "idle",
  datasetFormat: "",
  filesCompleted: 0,
  filesTotal: 0,
  bytesCompleted: 0,
  bytesTotal: 0,
};

// Saved episode history is review data, not heartbeat data. Fetch only the
// visible collection/RL scope and retain the last complete snapshot locally.
const EPISODE_HISTORY_POLL_MS = 5000;
const EPISODE_HISTORY_PAGE_SIZE = 128;
let episodeHistoryPolling = false;
let collectItemsRenderKey = "";
let rolloutItemsRenderKey = "";
const collectAutoAdvanceState = {
  selectionKey: "",
  historyReady: false,
  usableCollected: null,
  completionPending: false,
  scheduledKey: "",
};

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
    if (scope === "collect") renderCollect();
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
  qualityTransfer.episodesCompleted = 0;
  qualityTransfer.episodesTotal = 0;
  qualityTransfer.filesCompleted = 0;
  qualityTransfer.filesTotal = 0;
  qualityTransfer.bytesCompleted = 0;
  qualityTransfer.bytesTotal = 0;
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
  const teleop = (S.STATUS && S.STATUS.teleop) || {};
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
  const authorized = new Set(teleop.authorized_groups || []);
  const engaged = new Set(teleop.engaged_groups || []);
  $("collect-control-groups").querySelectorAll(".collect-control-state").forEach((row) => {
    const group = groups.find((item) => String(item.id) === row.dataset.group);
    if (!group) return;
    let state = "locked";
    if (!connected) state = "unavailable";
    else if (engaged.has(group.id)) state = "active";
    else if (authorized.has(group.id)) state = "ready";
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
  renderCollectControls();
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
    if (S.reviewKind === "collect") {
      returnReviewToLive();
    }
    const task = collectTaskValue();
    const collectionSet = collectSetValue();
    const taskIndex = collectTaskIndexValue();
    const needsSelection = task && (
      task !== S.STATUS.selected_collect_task ||
      collectionSet !== S.STATUS.selected_collect_set ||
      taskIndex !== S.STATUS.selected_collect_task_index
    );
    if (needsSelection) {
      S.STATUS.selected_collect_task = task;
      S.STATUS.selected_collect_set = collectionSet;
      S.STATUS.selected_collect_task_index = taskIndex;
      await apiPost("/api/select_collect_task", {
        task,
        dataset: collectionSet,
        task_index: taskIndex,
      });
    }
    await apiPost("/api/operator_action", { intent: "start" });
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

function selectCollectEpisode(item) {
    const episode = savedEpisodeId(item);
    if (episode == null) return;
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

function selectedCollectEpisodeItem() {
    const episode = S.collectReplayEpisode;
    if (episode == null) return null;
    if (!collectReviewMatchesSelection()) return null;
    const collect = S.STATUS.collect || {};
    const history = historyFor("collect", collect);
    const items = history.episodes.concat(history.queue);
    return items.find((item) => savedEpisodeId(item) === episode) || null;
  }

function renderCollectTiles(items) {
    const host = $("collect-queue-tiles");
    host.innerHTML = "";
    const hint = $("collect-tiles-hint");
    const anyReplayable = items.some((item) => savedEpisodeId(item) != null);
    if (hint) hint.style.display = anyReplayable ? "block" : "none";
    if (!items.length) {
      const empty = document.createElement("span");
      empty.className = "collect-empty";
      empty.textContent = "no episodes";
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
        tile.title = `episode ${item.episode_index}`;
        if (collectReviewMatchesSelection() && episode === S.collectReplayEpisode) {
          tile.classList.add("selected");
        }
        tile.onpointerdown = (event) => selectCollectEpisodePointer(event, item);
        tile.onclick = () => selectCollectEpisode(item);
      } else {
        tile.title = `episode ${item.episode_index} · ${item.status}`;
      }
      const episodeIndex = Number(item.episode_index);
      tile.textContent = Number.isFinite(episodeIndex)
        ? String(episodeIndex).padStart(3, "0")
        : "---";
      tile.setAttribute("aria-label", tile.title);
      host.appendChild(tile);
    });
  }

function renderCollectList(items) {
    const host = $("collect-queue-list");
    host.style.display = S.collectQueueExpanded ? "block" : "none";
    $("collect-queue-toggle").textContent = S.collectQueueExpanded ? "COLLAPSE" : "EXPAND";
    host.innerHTML = "";
    if (!items.length) return;
    items.slice().reverse().forEach((item) => {
      const row = document.createElement("div");
      const episode = savedEpisodeId(item);
      const selected = collectReviewMatchesSelection() && episode === S.collectReplayEpisode;
      row.className = `collect-row ${episode != null ? "replayable" : ""}${selected ? " selected" : ""}`;
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
      if (episode != null) {
        row.title = "select episode";
        row.onpointerdown = (event) => selectCollectEpisodePointer(event, item);
        row.onclick = () => selectCollectEpisode(item);
      }
      host.appendChild(row);
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

function collectHistoryReadyFor(collectionSet, prompt) {
    const cache = S.episodeHistory && S.episodeHistory.collect;
    return !!(
      cache && cache.loaded &&
      cache.collectionSet === collectionSet && cache.task === prompt
    );
  }

function maybeAutoAdvanceCollectTask(prompt, usableCollected, required, collecting) {
    const selectionKey = collectTaskSelectionKey();
    const historyReady = collectHistoryReadyFor(collectSetValue(), prompt);
    if (collectAutoAdvanceState.selectionKey !== selectionKey) {
      collectAutoAdvanceState.selectionKey = selectionKey;
      collectAutoAdvanceState.historyReady = false;
      collectAutoAdvanceState.usableCollected = null;
      collectAutoAdvanceState.completionPending = false;
      collectAutoAdvanceState.scheduledKey = "";
    }
    if (!historyReady) {
      collectAutoAdvanceState.historyReady = false;
      collectAutoAdvanceState.usableCollected = null;
      return;
    }
    if (!collectAutoAdvanceState.historyReady) {
      collectAutoAdvanceState.historyReady = true;
      collectAutoAdvanceState.usableCollected = usableCollected;
      return;
    }
    if (collectAutoAdvanceState.usableCollected === null) {
      collectAutoAdvanceState.usableCollected = usableCollected;
      return;
    }
    const crossedTarget = Number.isInteger(required) && required > 0 &&
      collectAutoAdvanceState.usableCollected < required && usableCollected >= required;
    collectAutoAdvanceState.usableCollected = usableCollected;
    if (crossedTarget) collectAutoAdvanceState.completionPending = true;
    if (usableCollected < required) collectAutoAdvanceState.completionPending = false;
    if (!collectAutoAdvanceState.completionPending || collecting) return;
    if (collectAutoAdvanceState.scheduledKey === selectionKey) return;
    collectAutoAdvanceState.scheduledKey = selectionKey;
    queueMicrotask(() => {
      const stillCollecting = !!(S.STATUS.collect && S.STATUS.collect.collecting);
      if (collectTaskSelectionKey() !== selectionKey || stillCollecting) return;
      collectAutoAdvanceState.completionPending = false;
      collectAutoAdvanceState.scheduledKey = "";
      advanceCollectTask();
    });
  }

function renderCollect() {
    if (!$("collect-control-col")) return;
    const collect = S.STATUS.collect || {};
    const history = historyFor("collect", collect);
    const enabled = collectEnabled();
    const collecting = !!collect.collecting;
    // The toggle's action depends on the polled `collecting` flag, which lags the
    // click by up to a poll interval + round trip. While that catches up, keep the
    // button held so a second click can't re-fire start/stop on stale state.
    if (S.collectToggleBusy !== null && collecting === S.collectToggleBusy) {
      S.collectToggleBusy = null;
    }
    const toggleBusy = S.collectToggleBusy !== null;
    const prompt = collectTaskValue();
    const collectionSet = collectSetValue();
    const hasPrompt = !!prompt;
    const queueFull = collect.pipeline_state === "QUEUE_FULL";
    // historyFor is the collection view's set+prompt scope boundary.
    const episodes = history.episodes;
    const queue = history.queue;
    const totalItems = episodes.length + queue.length;
    const required = collectTaskTarget(prompt);
    const hasRequirement = Number.isInteger(required) && required > 0;
    const unlimited = required === -1;
    const usableCollected = history.summary.usable;
    const requirementComplete = hasRequirement && usableCollected >= required;
    maybeAutoAdvanceCollectTask(prompt, usableCollected, required, collecting);
    const queueSummary = episodeItemsSummary(queue);
    const usableCount = history.summary.usable + queueSummary.usable;
    const rejectedCount = history.summary.rejected + queueSummary.rejected;
    const pendingCount = history.summary.pending + queueSummary.pending;
    const progress = totalItems === 0
      ? 0 : Math.max(0, Math.min(1, (usableCount + rejectedCount) / totalItems));

    const collectFps = S.CFG && S.CFG.collection ? S.CFG.collection.fps : null;
    $("collect-fps").textContent = collectFps ? `${collectFps} FPS` : "";
    $("collect-count").textContent = `${episodes.length}/${totalItems}`;
    $("collect-usable-count").textContent = threeDigitCount(usableCount);
    $("collect-rejected-count").textContent = threeDigitCount(rejectedCount);
    $("collect-pending-count").textContent = threeDigitCount(pendingCount);
    $("collect-requirement-count").textContent = `${usableCollected} / ${unlimited ? "∞" : (hasRequirement ? required : "--")}`;
    $("collect-requirement-status").textContent = unlimited
      ? "NO LIMIT"
      : hasRequirement
      ? (requirementComplete ? "COMPLETE" : `${required - usableCollected} REMAINING`)
      : "TARGET NOT SET";
    $("collect-requirement").classList.toggle("complete", requirementComplete);
    $("collect-requirement").classList.toggle("unset", !hasRequirement && !unlimited);
    $("collect-progress-label").textContent = `${Math.round(progress * 100)}%`;
    $("collect-progress-fill").style.width = `${progress * 100}%`;
    $("collect-eta").textContent = fmtEta(collect.eta_sec);

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
    toggle.disabled = toggleBusy ||
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
    $("b-collect-qc-pass").disabled = !enabled || !selectedEpisodeSaved;
    $("b-goto-qc").disabled = !enabled || !selectedEpisodeSaved;
    $("b-collect-note-save").disabled = !selectedEpisodeSaved;
    const exportButton = $("b-collect-quality-export");
    const uploadButton = $("b-collect-quality-upload");
    const exportFormat = $("collect-export-format");
    const selectedFormat = exportFormat ? exportFormat.value : "";
    const selectedExportReady = !!qualityTransfer.acceptedDir &&
      qualityTransfer.datasetFormat === selectedFormat;
    const upload = (S.CFG && S.CFG.collection && S.CFG.collection.upload) || {};
    if (exportButton) {
      exportButton.disabled = !enabled || !episodes.length ||
        qualityTransfer.exporting || qualityTransfer.uploading;
    }
    if (uploadButton) {
      uploadButton.disabled = !upload.configured || !selectedExportReady ||
        qualityTransfer.exporting || qualityTransfer.uploading;
      const backendLabel = (upload.backends || []).map((value) => String(value).toUpperCase());
      uploadButton.textContent = backendLabel.length
        ? `UPLOAD ${backendLabel.join(" + ")}`
        : "UPLOAD ACCEPTED";
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
        : (`${qualityTransfer.filesCompleted}/${qualityTransfer.filesTotal} files · ` +
          `${formatTransferBytes(qualityTransfer.bytesCompleted)}/` +
          `${formatTransferBytes(qualityTransfer.bytesTotal)} · ${formatLabel}`);
    }

    const recordState = collecting || (hasPrompt && !S.collectArmEnabled)
      ? "active"
      : (hasPrompt ? "done" : "pending");
    setPanel("collect-panel-task", enabled && hasPrompt ? "done" : "active");
    setPanel("collect-panel-record", recordState);
    const queueEnabled = enabled && (S.collectQueueEnabled || episodes.length > 0 || queue.length > 0);
    setPanel("collect-panel-queue", queue.length ? "active" : (queueEnabled ? "done" : "pending"));
    setPanel("collect-panel-replay", selectedEpisodeSaved ? "active" : "pending");

    const renderKey = [
      collectionSet,
      prompt,
      history.summary.signature,
      queueSummary.signature,
      selectedEpisodeSaved ? S.collectReplayEpisode : "",
      S.collectQueueExpanded ? "expanded" : "collapsed",
    ].join("\n");
    if (renderKey !== collectItemsRenderKey) {
      collectItemsRenderKey = renderKey;
      const items = episodes.concat(queue);
      renderCollectTiles(items);
      renderCollectList(items);
    }
    renderCollectControls();

    const replayStatus = $("collect-replay-status");
    if (S.reviewKind === "collect" && LIVE.replayOwner === "collect") {
      replayStatus.textContent = LIVE.replayError
        ? `episode ${S.collectReplayEpisode} · error · ${LIVE.replayError}`
        : (LIVE.replayLoading
            ? `episode ${S.collectReplayEpisode} · loading`
            : `episode ${S.collectReplayEpisode} · review`);
      replayStatus.style.display = S.ACTIVE_TAB === "collect" ? "" : "none";
    } else if (selectedEpisodeSaved) {
      replayStatus.textContent = `episode ${S.collectReplayEpisode} selected`;
      replayStatus.style.display = S.ACTIVE_TAB === "collect" ? "" : "none";
    } else {
      replayStatus.textContent = "";
      replayStatus.style.display = "none";
    }
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
    qualityTransfer.datasetFormat = datasetFormat;
    if (status) status.textContent = `exporting ${formatLabel}…`;
    renderCollect();
    try {
      const result = await apiPost("/api/collect_quality_export", {
        task: collectTaskValue(),
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
    qualityTransfer.phase = "upload";
    qualityTransfer.uploading = true;
    qualityTransfer.uploadJobId = "";
    qualityTransfer.uploadState = "queued";
    qualityTransfer.filesCompleted = 0;
    qualityTransfer.filesTotal = 0;
    qualityTransfer.bytesCompleted = 0;
    qualityTransfer.bytesTotal = 0;
    if (status) status.textContent = `uploading ${formatLabel} accepted export…`;
    renderCollect();
    try {
      const result = await apiPost("/api/collect_quality_upload", {
        task: collectTaskValue(),
        dataset_format: datasetFormat,
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
        renderCollect();
        if (job.state === "completed") {
          if (status) {
            const remoteDir = job.remote_dir || "remote target";
            status.textContent = `${formatLabel} accepted upload complete · ` +
              `${job.files_total || 0} files · ${remoteDir}`;
          }
          return;
        }
        if (job.state === "failed") {
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
    return (S.STATUS.collect || {}).dataset_dir ||
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
    return reviewTask === collectTaskValue() && reviewCollectionSet === collectSetValue();
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
    reviewTask = collectTaskValue();
    reviewCollectionSet = collectSetValue();
    S.reviewKind = "collect";
    reviewDatasetDir = reviewDatasetFor("collect");
    reviewEpisodeId = episode;
    LIVE.replayOwner = "collect";
    LIVE.replayMode = true;
    LIVE.replayLoading = true;
    LIVE.replayError = "";
    updateScrub();
    const title = reviewTitleFor("collect");
    if (title) title.textContent = `episode ${episode} · loading`;
    const err = reviewErrorFor("collect");
    if (err) err.textContent = "";
    const r = await apiPost("/api/review_episode", {
      dataset_dir: reviewDatasetDir,
      episode: String(reviewEpisodeId),
    });
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
    if (title) title.textContent = `episode ${episode} · review`;
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
    const r = await apiPost("/api/review_episode", {
      dataset_dir: reviewDatasetDir,
      episode: String(reviewEpisodeId),
    });
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
    const episode = kind === "rollout" ? S.rolloutSaveEpisode : S.collectReplayEpisode;
    if (episode == null) return;
    S.reviewKind = kind;
    reviewDatasetDir = reviewDatasetFor(kind);
    reviewEpisodeId = episode;
    clientTrace("review.qc.begin", { kind, episode, verdict, dataset_dir: reviewDatasetDir });
    const r = await apiPost(episodeQcEndpoint(kind), {
      dataset_dir: reviewDatasetDir,
      task: reviewTask,
      dataset: kind === "collect" ? reviewCollectionSet : "",
      episode: String(reviewEpisodeId),
      verdict,
      note: reviewNoteFor(kind).value || "",
    });
    const title = reviewTitleFor(kind);
    const status = kind === "rollout" ? $("rollout-save-err") : $("collect-qc-status");
    clientTrace("review.qc.end", {
      kind, episode, verdict, ok: !!r.ok, error: r.error || "",
    });
    if (!r.ok) {
      if (title) title.textContent = `episode ${episode} · QC failed`;
      if (status) status.textContent = `✗ ${r.error || "QC failed"}`;
      return;
    }
    if (title) title.textContent = `episode ${episode} · ${verdict}`;
    if (status) status.textContent = `episode ${episode} marked ${verdict}`;
    invalidateEpisodeHistory(kind);
    pollEpisodeHistory(true);
    applyStatus(await apiGet("/api/status"));
  }

async function submitEpisodeNote(kind) {
    const episode = kind === "rollout" ? S.rolloutSaveEpisode : S.collectReplayEpisode;
    const status = kind === "rollout" ? $("rollout-save-err") : $("collect-qc-status");
    if (episode == null) {
      if (status) status.textContent = "✗ select an episode first";
      return;
    }
    S.reviewKind = kind;
    reviewDatasetDir = reviewDatasetFor(kind);
    reviewEpisodeId = episode;
    if (status) status.textContent = "saving…";
    const r = await apiPost(episodeQcEndpoint(kind), {
      dataset_dir: reviewDatasetDir,
      task: reviewTask,
      dataset: kind === "collect" ? reviewCollectionSet : "",
      episode: String(reviewEpisodeId),
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
    const r = await apiPost("/api/annotate", {
      dataset_dir: dir, episode: ep, annotation: $("replay-anno-text").value || "",
    });
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
};
