// Local execution sessions. Reviewing metadata never issues motion commands.
import { $, LIVE, S, apiGet, apiPost, releaseCameraImage, setCommandMetadata } from "./core.js";
import { loadMountedReplaySeries, replayPlay } from "./replay.js";

let selectedSession = "";
let requestVersion = 0;
let lastRefresh = 0;
let refreshing = false;
let listKey = "";
let recordPending = null;
let sessionItems = [];
let openingReplay = false;
let lastDeleted = null;
let historyVersion = 0;
let lastRolloutId;

function busy() {
  const status = S.STATUS || {};
  return !!status.local_history?.rollout_id || status.session_status === "running" ||
    ["starting", "running", "stopping", "resetting"].includes(status.web_phase);
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function timeLabel(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function showError(message) {
  $("debug-history-status").textContent = message || "";
}

const icons = {
  trash: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/></svg>',

  play: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m9 5 10 7-10 7Z"/></svg>',
};

function icon(name) {
  const node = element("span", null, "debug-history-icon");
  node.innerHTML = icons[name];
  return node;
}

function runButton(sessionId, run) {
  const row = element("div", null, "debug-run-row");
  const button = element("button", null, "debug-session-item debug-run-item");
  button.type = "button";
  button.dataset.sessionId = sessionId;
  button.dataset.unrecorded = run.playable ? "0" : "1";
  button.disabled = busy() || !run.playable;
  button.title = `${run.prompt} · ${run.model} · ${run.status} · ${run.steps || 0} actions`;
  button.appendChild(icon("play"));
  const label = element("span", null, "debug-history-label");
  label.appendChild(element("strong", run.prompt || "Untitled run"));
  label.appendChild(element("small", `${run.model} · ${run.playable ? timeLabel(run.started_at) : recordLabel(run)}`));
  button.appendChild(label);
  button.onclick = () => { selectedSession = sessionId; highlightSession(); void openLocalReplay(sessionId, run); };
  row.appendChild(button);
  const remove = element("button", null, "debug-run-delete");
  remove.type = "button";
  remove.dataset.deleteRun = "1";
  remove.title = run.status === "running" ? "Stop this run before deleting it" :
    ["saving", "recording"].includes(run.record_status) ? "Wait for this recording to finish" : "Delete run";
  remove.setAttribute("aria-label", `Delete run: ${run.prompt || "Untitled run"}`);
  remove.dataset.unavailable = run.status === "running" || ["saving", "recording"].includes(run.record_status) ? "1" : "0";
  remove.disabled = remove.dataset.unavailable === "1";
  remove.appendChild(icon("trash"));
  remove.onclick = async (event) => {
    event.stopPropagation();
    if (remove.disabled) return;
    remove.dataset.deletePending = "1";
    remove.disabled = true;
    try {
      const response = await apiPost("/api/delete_local_run", { session_id: sessionId, rollout_id: run.id });
      if (!response.ok) throw new Error(response.error || "Unable to delete run");
      sessionItems.forEach((session) => {
        if (session.id === sessionId) session.runs = session.runs.filter((item) => item.id !== run.id);
      });
      ++historyVersion;
      listKey = "";
      lastDeleted = { session_id: sessionId, rollout_id: run.id };
      renderSessions();
      const status = $("debug-history-status");
      const undo = element("button", "Undo", "debug-history-undo");
      undo.type = "button";
      const deleted = lastDeleted;
      undo.onclick = async () => {
        try {
          const restored = await apiPost("/api/restore_local_run", deleted);
          if (!restored.ok) throw new Error(restored.error || "Unable to restore run");
          lastDeleted = null;
          ++historyVersion;
          listKey = "";
          lastRefresh = 0;
          await refreshLocalHistory(true);
        } catch (error) { showError(error.message); }
      };
      status.replaceChildren(element("span", "Run deleted · "), undo);
      lastRefresh = 0;
    } catch (error) {
      remove.dataset.deletePending = "";
      remove.disabled = false;
      showError(error.message);
    }
  };
  row.appendChild(remove);
  return row;
}

function renderSessions() {
  const rows = sessionItems.flatMap((session) => (session.runs || []).map((run) => ({ sessionId: session.id, run })));
  rows.sort((a, b) => String(b.run.started_at).localeCompare(String(a.run.started_at)));
  $("debug-session-list").replaceChildren(...rows.map(({ sessionId, run }) => runButton(sessionId, run)));
}

function highlightSession() {
  $("debug-session-list").querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.sessionId === selectedSession);
  });
}

function recordLabel(run) {
  if (!run.record) return "Not recorded";
  if (run.playable) return "Recorded";
  return ({ recording: "Recording", saving: "Saving…", empty: "No recorded frames",
    error: "Recording failed", interrupted: "Recording interrupted" })[run.record_status] || "Not recorded";
}

async function openLocalReplay(sessionId, run) {
  if (busy() || openingReplay) return;
  openingReplay = true;
  const version = ++requestVersion;
  showError("Opening Replay…");
  try {
    // Resolve the recording from its local IDs before reusing the normal Replay
    // workspace. The sim replay mode plays saved data without robot dispatch.
    const info = await apiPost("/api/local_replay", { session_id: sessionId, rollout_id: run.id });
    if (version !== requestVersion || S.ACTIVE_TAB !== "debug") return;
    if (!info.ok) throw new Error(info.error || "Recording unavailable");
    S.tabSwitchPending = true;
    document.querySelectorAll("img.cam").forEach(releaseCameraImage);
    const switched = await apiPost("/api/tab_switch", { tab: "replay" });
    if (!switched.ok) throw new Error(switched.error || "Unable to open Replay");
    await apiPost("/api/select_mode", { mode: "sim" });
    S.STATUS.cli_mode = "sim";
    const { setActiveTab } = await import("./main.js");
    closeLocalHistoryView();
    setActiveTab("replay");
    $("replay-dataset-input").value = info.dataset_dir;
    $("replay-episode-input").value = "0";
    const mounted = await apiPost("/api/load_replay_dataset", { dataset_dir: info.dataset_dir, episode: 0 });
    if (!mounted.ok) throw new Error(mounted.error || "Recording could not be loaded");
    if (await loadMountedReplaySeries(mounted)) replayPlay();
    else throw new Error(LIVE.replayError || "Recording could not be played");
  } catch (cause) {
    showError(cause.message);
    const replayError = $("replay-err");
    if (S.ACTIVE_TAB === "replay" && replayError) replayError.textContent = cause.message;
  } finally {
    openingReplay = false;
    S.tabSwitchPending = false;
    if (S.ACTIVE_TAB === "replay") showError("");
  }
}

function closeLocalHistoryView() {
  ++requestVersion;
  selectedSession = "";
  if ($("debug-session-list")) highlightSession();
}

async function refreshLocalHistory(force = false) {
  if (refreshing || S.ACTIVE_TAB !== "debug" || document.hidden ||
      (!force && Date.now() - lastRefresh < 5000)) return;
  refreshing = true;
  const version = historyVersion;
  lastRefresh = Date.now();
  try {
    const data = await apiGet("/api/local_sessions");
    if (version !== historyVersion || S.ACTIVE_TAB !== "debug") return;
    const key = JSON.stringify(data.sessions || []);
    if (key !== listKey) {
      listKey = key;
      sessionItems = data.sessions || [];
      renderSessions();
    }
    if (!lastDeleted) showError(sessionItems.some((item) => item.runs?.length) ? "" : "No executions yet");
  } catch {
    showError("Local history unavailable");
  } finally { refreshing = false; }
}

function applyLocalHistoryStatus(status) {
  if (!$("debug-history-sidebar") || !S.CFG) return;
  const rolloutId = status.local_history?.rollout_id ?? null;
  if (rolloutId !== lastRolloutId) {
    lastRolloutId = rolloutId;
    ++historyVersion;
    lastRefresh = 0;
  }
  const available = (status.local_history || S.CFG.local_history || {}).enabled !== false;
  $("debug-history-sidebar").hidden = !available;
  const button = $("debug-record-toggle");
  button.hidden = !available;
  const enabled = !!status.local_history?.record;
  if (recordPending && (enabled === recordPending.enabled || Date.now() - recordPending.started > 10000)) recordPending = null;
  button.setAttribute("aria-pressed", String(recordPending ? recordPending.enabled : enabled));
  button.title = (recordPending ? recordPending.enabled : enabled) ? "Record rollout: on" : "Record rollout: off";
  button.disabled = busy() || !!recordPending;
  $("debug-session-list").querySelectorAll("button").forEach((item) => {
    item.disabled = item.dataset.deleteRun === "1"
      ? item.dataset.unavailable === "1" || item.dataset.deletePending === "1"
      : busy() || item.dataset.unrecorded === "1";
  });
  if (available) void refreshLocalHistory();
}

function initLocalHistory() {
  if (!$("debug-record-toggle")) return;
  const record = $("debug-record-toggle");
  setCommandMetadata(record, "web:debug_record:{enabled}", true);
  record.onclick = async () => {
    if (busy() || recordPending) return;
    const enabled = !S.STATUS.local_history?.record;
    recordPending = { enabled, started: Date.now() };
    applyLocalHistoryStatus(S.STATUS);
    try {
      const response = await apiPost("/api/debug_record", { enabled });
      if (!response.ok) throw new Error(response.error || "Unable to change recording");
    } catch (error) {
      recordPending = null;
      showError(error.message);
      applyLocalHistoryStatus(S.STATUS);
    }
  };
  applyLocalHistoryStatus(S.STATUS || {});
}

export { applyLocalHistoryStatus, closeLocalHistoryView, initLocalHistory };
