// rl.js: RL rollout/HIL controls, saved-episode replay, and Critic telemetry.
import { $, LIVE, S, apiGet, apiPost, clientTrace, setCommandMetadata } from "./core.js";
import { drawLiveCharts, updateScrub } from "./charts.js";
import { exitReplayMode, loadReviewPlayback } from "./replay.js";
import { renderRlGripper, updateGuide } from "./run.js";

let selectedEpisode = null;
let seriesPolling = false;
let lastQueuedReplayFrame = -1;
let replayRequestId = 0;
let rlReplayRequestPending = false;
let rlCriticInFlight = false;
let rlCriticPendingFrame = null;
let rlCriticGeneration = 0;
let rlSetupTimer = null;
let rlSetupRequestPending = false;
let rlLiveActionSince = 0;
let rlSelectionChain = Promise.resolve();
let rlSelectionGeneration = 0;
let rlPendingTask = null;
let rlPendingPolicy = null;
let rlSavedRenderKey = "";
let rlSaveSetupPending = false;

function queueRlSelection(path, body) {
  const generation = ++rlSelectionGeneration;
  rlSelectionChain = rlSelectionChain
    .catch(() => {})
    .then(() => apiPost(path, body))
    .then((response) => {
      if (!response || response.ok === false) {
        throw new Error((response && response.error) || `${path} failed`);
      }
    })
    .catch((error) => {
      if (generation !== rlSelectionGeneration) return;
      S.STATUS.last_error = String(error);
      renderRlStatus(S.STATUS);
    });
  return rlSelectionChain;
}

function requestRlSetup() {
  if (rlSetupRequestPending) return;
  rlSetupRequestPending = true;
  apiPost("/api/rl/setup").then(
    () => { rlSetupRequestPending = false; },
    () => { rlSetupRequestPending = false; },
  );
}

function scheduleRlSetup() {
  if (rlSetupTimer !== null) clearTimeout(rlSetupTimer);
  rlSetupTimer = setTimeout(() => {
    rlSetupTimer = null;
    const status = S.STATUS || {};
    if (!S.rlTask || S.rlPolicy === "" || status.is_setup_done || status.setup_stage || status.session_status === "running") return;
    if (status.last_error && !status.is_setup_done) return;
    requestRlSetup();
  }, 0);
}

function addPlaceholder(select, label) {
  const option = document.createElement("option");
  option.value = "";
  option.disabled = true;
  option.textContent = label;
  select.appendChild(option);
}

function fillSelect(select, items, placeholder, selected) {
  select.innerHTML = "";
  addPlaceholder(select, placeholder);
  items.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(item.value != null ? item.value : (item.slot != null ? item.slot : index));
    option.textContent = `${String(index + 1).padStart(2, "0")} ${item.name || item}`;
    select.appendChild(option);
  });
  select.value = selected == null ? "" : String(selected);
  select.classList.toggle("has-value", !!select.value);
}

function syncTaskChip(select) {
  const chip = $("rl-task-chip");
  const option = select.selectedOptions && select.selectedOptions[0];
  let value = chip.querySelector(".v");
  if (!value) {
    value = document.createElement("span");
    value.className = "v";
    chip.appendChild(value);
  }
  value.textContent = option && option.value ? option.textContent.replace(/^\d+\s+/, "") : "";
  chip.classList.toggle("set", !!select.value);
}

function syncModelLink(id, selected, connected, error, optional = false) {
  const link = $(id);
  if (!selected) link.textContent = optional ? "OPTIONAL · NOT SELECTED" : "NOT SELECTED";
  else if (connected) link.textContent = "LINKED";
  else if (error) link.textContent = optional ? "OPTIONAL · ERROR" : "ERROR";
  else link.textContent = optional ? "OPTIONAL · NOT CONNECTED" : "SELECTED · NOT CONNECTED";
  link.classList.toggle("selected", !!selected);
  link.title = error || "";
}

function syncRlStageCharts(status) {
  const stage = $("stage");
  if (!stage) return;
  const rl = (status && status.rl) || {};
  const replay = S.ACTIVE_TAB === "rl" && LIVE.replayMode && LIVE.replayOwner === "rl";
  stage.classList.toggle("rl-live", S.ACTIVE_TAB === "rl" && !replay);
  stage.classList.toggle("rl-replay", replay);
  stage.classList.toggle("rl-critic-active", !!S.rlCritic && !!rl.active);
}

function renderRlConfig() {
  const cfg = (S.CFG && S.CFG.rl) || { enabled: false };
  if (!cfg.enabled) return;

  const task = $("rl-task-list");
  fillSelect(
    task,
    (cfg.tasks || []).map((name) => ({ name, value: name })),
    "SELECT TASK",
    S.rlTask,
  );
  setCommandMetadata(task, "web:rl_select_task:{task}", true);
  task.onchange = () => {
    S.rlTask = task.value;
    rlPendingTask = task.value;
    task.classList.toggle("has-value", !!task.value);
    syncTaskChip(task);
    S.STATUS.last_error = "";
    queueRlSelection("/api/rl/select_task", { task: task.value });
    updateGuide();
  };
  syncTaskChip(task);

  const policy = $("rl-policy-list");
  fillSelect(policy, cfg.policies || [], "SELECT POLICY", S.rlPolicy);
  setCommandMetadata(policy, "web:rl_select_policy:{slot}", true);
  policy.onchange = () => {
    S.rlPolicy = policy.value;
    rlPendingPolicy = policy.value;
    policy.classList.toggle("has-value", !!policy.value);
    S.STATUS.last_error = "";
    queueRlSelection("/api/rl/select_policy", { slot: Number(policy.value) });
    updateGuide();
  };

  const critic = $("rl-critic-list");
  fillSelect(critic, cfg.critics || [], "SELECT CRITIC", S.rlCritic);
  setCommandMetadata(critic, "web:rl_select_critic:{slot}", true);
  critic.onchange = () => {
    if (!S.STATUS.is_setup_done) return;
    S.rlCritic = critic.value;
    critic.classList.toggle("has-value", !!critic.value);
    if (critic.value !== "") apiPost("/api/rl/select_critic", { slot: Number(critic.value) });
    updateGuide();
  };

  $("rl-mode").textContent = String(cfg.cli_mode || "—").toUpperCase();
  $("rl-strategy").textContent = String(cfg.inference_strategy || "—").toUpperCase();
  $("rl-data-format").value = cfg.data && cfg.data.format === "lerobot" ? "lerobot" : "";
  $("rl-dataset-dir").textContent = (cfg.data && cfg.data.dataset_dir) || "—";
  $("rl-dataset-dir").title = (cfg.data && cfg.data.dataset_dir) || "";
  renderRlStatus(S.STATUS || {});
  scheduleRlSetup();
}

function episodeId(item) {
  if (!item || item.status !== "saved") return null;
  const value = Number(item.episode_index);
  return Number.isFinite(value) ? value : null;
}

function episodeTone(item) {
  if (item.status === "queued") return "cq-queued";
  if (item.status === "saving") return "cq-busy";
  if (item.status === "failed") return "cq-fail";
  if (item.qc_verdict === "pass") return "cq-ok";
  if (item.qc_verdict === "fail" || item.quality === "red") return "cq-fail";
  return "cq-queued";
}

function episodeIssue(item) {
  if (item.error) return item.error;
  if (item.qc_verdict) return `qc ${item.qc_verdict}`;
  const issues = item.quality_issues || [];
  return issues.length
    ? issues.map((issue) => `${issue.code || "issue"}${Number(issue.count || 1) > 1 ? ` ×${issue.count}` : ""}`).join(", ")
    : (item.status || "ok");
}

function renderSavedEpisodes(items) {
  const host = $("rl-save-tiles");
  host.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("span");
    empty.className = "collect-empty";
    empty.textContent = "no saved episodes";
    host.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const tile = document.createElement("button");
    const id = episodeId(item);
    tile.type = "button";
    tile.className = `collect-tile ${episodeTone(item)}`;
    if (id != null) {
      tile.classList.add("replayable");
      tile.classList.toggle("selected", selectedEpisode === id);
      tile.title = `episode ${id} · ${item.length || 0} frames`;
      tile.onclick = () => {
        selectSavedEpisode(item, items);
      };
    } else {
      tile.disabled = true;
    }
    host.appendChild(tile);
  });
}

function renderSavedEpisodeList(items) {
  const host = $("rl-save-list");
  host.style.display = S.rlSaveExpanded ? "block" : "none";
  $("rl-save-expand").textContent = S.rlSaveExpanded ? "COLLAPSE" : "EXPAND";
  host.innerHTML = "";
  if (!items.length) return;
  items.slice().reverse().forEach((item) => {
    const id = episodeId(item);
    const row = document.createElement("div");
    row.className = `collect-row ${id != null ? "replayable" : ""}${id === selectedEpisode ? " selected" : ""}`;
    const episode = document.createElement("span");
    const frames = document.createElement("span");
    const issue = document.createElement("span");
    episode.textContent = id == null ? "—" : `#${String(id).padStart(3, "0")}`;
    frames.textContent = `${item.length || 0}f`;
    issue.className = "issue";
    issue.textContent = episodeIssue(item);
    row.append(episode, frames, issue);
    if (id != null) {
      row.onclick = () => {
        selectSavedEpisode(item, items);
      };
    }
    host.appendChild(row);
  });
}

function renderSavedData(items, force = false) {
  const key = JSON.stringify({
    selectedEpisode,
    expanded: S.rlSaveExpanded,
    items: items.map((item) => ({
      episode_index: item.episode_index,
      length: item.length,
      status: item.status,
      quality: item.quality,
      qc_verdict: item.qc_verdict,
      quality_issue_count: item.quality_issue_count,
      quality_issues: (item.quality_issues || []).map((issue) => ({
        code: issue.code,
        count: issue.count,
      })),
      error: item.error,
    })),
  });
  if (!force && key === rlSavedRenderKey) return;
  rlSavedRenderKey = key;
  renderSavedEpisodes(items);
  renderSavedEpisodeList(items);
}

function selectSavedEpisode(item, items) {
  const id = episodeId(item);
  if (id == null) return;
  if (rlReplayRequestPending && selectedEpisode === id) return;
  selectedEpisode = id;
  S.rlQcEpisode = id;
  renderSavedData(items, true);
  $("rl-b-replay").disabled = false;
  replaySelectedEpisode();
}

async function submitRlQc(verdict) {
  const episode = S.rlQcEpisode;
  const status = $("rl-qc-status");
  if (episode == null) {
    if (status) status.textContent = "✗ select an episode first";
    return;
  }
  const rollout = S.STATUS.rollout || {};
  const note = $("rl-qc-note").value || "";
  clientTrace("rl.qc.begin", { episode, verdict, dataset_dir: rollout.dataset_dir || "" });
  const response = await apiPost("/api/qc_mark", {
    dataset_dir: rollout.dataset_dir || "",
    episode: String(episode),
    verdict,
    note,
  });
  if (!response.ok) {
    if (status) status.textContent = `✗ ${response.error || "QC failed"}`;
    return;
  }
  if (status) {
    status.textContent = verdict
      ? `episode ${episode} marked ${verdict}`
      : `episode ${episode} note saved`;
  }
  S.STATUS = await apiGet("/api/status");
  renderRlStatus(S.STATUS);
}

function renderRlSource(status) {
  const readout = $("rl-control-source");
  const value = $("rl-control-source-value");
  if (!readout || !value) return;
  let source = "idle";
  if (LIVE.replayMode && LIVE.replayOwner === "rl" && LIVE.n) {
    const cursor = LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor;
    const index = Math.max(0, Math.min(Math.round(Number(cursor) || 0), LIVE.n - 1));
    source = LIVE.controlSource[index] === "intervention" ? "intervention" : "policy";
  } else if (status.rollout_intervention_active) {
    source = "intervention";
  } else if (status.session_status === "running" && status.is_setup_done) {
    source = "policy";
  }
  readout.dataset.source = source;
  value.textContent = source === "intervention" ? "INTERVENTION" : source.toUpperCase();
}

function replayCriticSource(timestamp) {
  if (LIVE.replayOwner !== "rl" || !LIVE.n || LIVE.controlSource.length !== LIVE.n) {
    return "replay";
  }
  const timeline = LIVE.timestamp;
  let low = 0;
  let high = timeline.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (Number(timeline[middle]) < Number(timestamp)) low = middle;
    else high = middle - 1;
  }
  const before = low;
  const after = Math.min(low + 1, timeline.length - 1);
  const nearest = Math.abs(Number(timeline[after]) - Number(timestamp))
    < Math.abs(Number(timeline[before]) - Number(timestamp)) ? after : before;
  return LIVE.controlSource[nearest] === "intervention" ? "intervention" : "policy";
}

function clearRlCriticSeries() {
  LIVE.criticTimestamp = [];
  LIVE.criticValue = [];
  LIVE.criticSource = [];
  LIVE.criticGeneration += 1;
  drawLiveCharts();
}

function clearRlCriticQueue() {
  rlCriticGeneration += 1;
  rlCriticPendingFrame = null;
}

function flushRlCriticQueue() {
  if (rlCriticInFlight || rlCriticPendingFrame == null) return;
  const generation = rlCriticGeneration;
  const frame = rlCriticPendingFrame;
  rlCriticPendingFrame = null;
  rlCriticInFlight = true;
  fetch("/api/rl/replay_critic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame }),
  })
    .then(async (response) => {
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        clientTrace("rl.replay_critic.error", { frame, error: payload.error || response.status });
      }
    })
    .catch((error) => clientTrace("rl.replay_critic.error", { frame, error: String(error) }))
    .finally(() => {
      rlCriticInFlight = false;
      if (generation === rlCriticGeneration) flushRlCriticQueue();
    });
}

async function postRlReplay(path, body) {
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    return response.ok ? payload : { ok: false, error: payload.error || response.status };
  } catch (error) {
    return { ok: false, error: String(error) };
  }
}

async function replaySelectedEpisode() {
  if (selectedEpisode == null) return;
  const requestId = ++replayRequestId;
  rlReplayRequestPending = true;
  clearRlCriticQueue();
  if (LIVE.replayMode && LIVE.replayOwner === "rl") exitReplayMode();
  clearRlCriticSeries();
  try {
    const rollout = (S.STATUS && S.STATUS.rollout) || {};
    lastQueuedReplayFrame = -1;
    const response = await postRlReplay("/api/rl/review_episode", {
      dataset_dir: rollout.dataset_dir || "",
      episode: selectedEpisode,
    });
    if (requestId !== replayRequestId || S.ACTIVE_TAB !== "rl") return;
    if (!response.ok) {
      $("rl-save-error").textContent = response.error || "replay load failed";
      return;
    }
    const started = await loadReviewPlayback(
      { ...response, dataset_dir: rollout.dataset_dir || "" },
      "rl",
    );
    if (requestId !== replayRequestId) return;
    if (!started) $("rl-save-error").textContent = LIVE.replayError || "replay failed";
  } finally {
    if (requestId === replayRequestId) rlReplayRequestPending = false;
  }
}

function renderRlStatus(status) {
  if (!S.CFG || !S.CFG.rl || !S.CFG.rl.enabled) return;
  const rl = status.rl || {};
  const rollout = status.rollout || {};
  if (
    rlSaveSetupPending
    && status.is_setup_done
    && status.session_status === "ready"
    && Number(status.step_index || 0) === 0
    && !rollout.save_ready
  ) {
    rlSaveSetupPending = false;
  }
  if (rlSaveSetupPending && status.last_error) rlSaveSetupPending = false;
  if (status.setup_stage || status.is_setup_done || status.last_error) rlSetupRequestPending = false;
  if (status.selected_task && rl.active) {
    S.rlTask = status.selected_task;
    if (rlPendingTask === S.rlTask) rlPendingTask = null;
  }
  if (rl.selected_policy_slot != null) {
    S.rlPolicy = String(rl.selected_policy_slot);
    if (rlPendingPolicy === S.rlPolicy) rlPendingPolicy = null;
  } else if (rlPendingPolicy == null) {
    S.rlPolicy = "";
  }
  if (rl.selected_critic_slot != null) S.rlCritic = String(rl.selected_critic_slot);
  else S.rlCritic = "";

  const task = $("rl-task-list");
  const policy = $("rl-policy-list");
  const critic = $("rl-critic-list");
  if (task && S.rlTask && task.value !== S.rlTask) task.value = S.rlTask;
  if (policy && S.rlPolicy !== "" && policy.value !== S.rlPolicy) policy.value = S.rlPolicy;
  if (critic && critic.value !== S.rlCritic) critic.value = S.rlCritic;
  [task, policy, critic].forEach((select) => {
    if (select) select.classList.toggle("has-value", !!select.value);
  });
  if (task) syncTaskChip(task);
  syncModelLink("rl-policy-link", !!S.rlPolicy, !!status.policy_connected, status.policy_error);
  syncModelLink("rl-critic-link", !!S.rlCritic, !!rl.critic_connected, rl.critic_error, true);
  renderRlSource(status);

  const selected = !!S.rlTask && S.rlPolicy !== "";
  const selectionConfirmed = status.selected_task === S.rlTask
    && rl.selected_policy_slot != null
    && String(rl.selected_policy_slot) === S.rlPolicy;
  const setupError = status.last_error || status.policy_error;
  const setup = !!status.is_setup_done && !!status.policy_connected;
  const setupBusy = !!status.setup_stage;
  const criticChoice = $("rl-critic-choice");
  if (criticChoice) criticChoice.style.display = setup ? "" : "none";
  syncRlStageCharts(status);
  updateScrub();
  const retry = $("rl-b-setup");
  retry.disabled = !selected || setupBusy || status.session_status === "running";
  retry.style.display = setupError && !setup ? "" : "none";
  const setupMsg = $("rl-auto-setup-msg");
  if (rlSaveSetupPending && !status.setup_stage) setupMsg.textContent = "SAVE · QUEUING DATA…";
  else if (status.last_error && !setup) setupMsg.textContent = "SETUP FAILED";
  else if (status.policy_error && !setup) setupMsg.textContent = "POLICY OFFLINE · SETUP REQUIRED";
  else if (setup) setupMsg.textContent = "ROBOT READY";
  else if (setupBusy) setupMsg.textContent = `AUTO · ${String(status.setup_stage).toUpperCase()}`;
  else if (rlSetupRequestPending) setupMsg.textContent = "AUTO · PREPARING ROBOT…";
  else if (selected) setupMsg.textContent = "READY · SETUP REQUIRED";
  else setupMsg.textContent = "SELECT TASK + POLICY";
  $("rl-setup-state").textContent = setupError
    ? `ERROR · ${setupError}`
    : (setup
        ? (rl.critic_connected ? "ROBOT + POLICY + CRITIC READY" : "ROBOT + POLICY READY · CRITIC OPTIONAL")
        : (setupBusy ? String(status.setup_stage).toUpperCase() : "Ready to setup"));

  const running = status.session_status === "running";
  const intervention = !!status.rollout_intervention_active;
  const hilSupported = !!status.hil_supported;
  const hilEnabled = !!status.rollout_intervention_enabled;
  $("rl-hil-enable").checked = hilEnabled;
  $("rl-hil-enable").disabled = !setup || !hilSupported || intervention;
  $("rl-hil-label").textContent = hilSupported ? (hilEnabled ? "HIL ON" : "HIL OFF") : "HIL N/A";
  $("rl-hil-gate").classList.toggle("on", hilEnabled);
  $("rl-hil-gate").classList.toggle("disabled", $("rl-hil-enable").disabled);
  $("rl-b-run").disabled = !setup || running || intervention || rlSaveSetupPending;
  $("rl-b-reset").disabled = !setup || intervention || setupBusy;
  $("rl-run-label").textContent = status.step_index > 0 ? "CONTINUE ▶▶" : "RUN ▶";
  $("rl-b-run").classList.toggle("recording", running);
  $("rl-b-intervene").disabled = !running;
  $("rl-b-intervene").textContent = hilEnabled ? "INTERVENE ■" : "STOP ■";
  $("rl-b-accept").disabled = !intervention;
  $("rl-b-abandon").disabled = !intervention;
  $("rl-run-error").textContent = setupError || rl.critic_error || "";

  const progress = Math.max(0, Math.min(1, Number(rollout.progress || 0)));
  const items = (rollout.episodes || []).concat(rollout.queue || []);
  $("rl-save-count").textContent = `${(rollout.episodes || []).length}/${items.length}`;
  $("rl-save-eta").textContent = rollout.eta_sec == null ? "—" : `${Number(rollout.eta_sec).toFixed(1)}s`;
  $("rl-save-hint").style.display = items.some((item) => episodeId(item) != null) ? "" : "none";
  $("rl-save-progress").style.width = `${progress * 100}%`;
  $("rl-save-pipeline").textContent = String(rollout.pipeline_state || "IDLE").toUpperCase();
  $("rl-save-pipeline").dataset.state = String(rollout.pipeline_state || "idle").toUpperCase();
  $("rl-b-save").disabled = !rollout.enabled || !rollout.save_ready || intervention || rlSaveSetupPending;
  $("rl-b-replay").disabled = selectedEpisode == null;
  const qcBox = $("rl-qc-box");
  const qcSelected = S.rlQcEpisode != null;
  if (qcBox) qcBox.style.display = qcSelected ? "" : "none";
  $("rl-b-qc-pass").disabled = !qcSelected;
  $("rl-b-qc-fail").disabled = !qcSelected;
  $("rl-b-qc-note-save").disabled = !qcSelected;
  $("rl-save-dir").textContent = rollout.dataset_dir ? `saved to ${rollout.dataset_dir}` : "";
  $("rl-save-error").textContent = rollout.save_blocked_by_intervention
    ? "accept or abandon the active intervention before saving"
    : "";
  renderSavedData(items);
  renderRlGripper(setup);
  if (selectionConfirmed && !setup && !setupBusy && !setupError) scheduleRlSetup();
  updateGuide();
}

async function pollRlSeries() {
  if (S.ACTIVE_TAB !== "rl" || seriesPolling || rlReplayRequestPending) return;
  seriesPolling = true;
  try {
    const criticGeneration = LIVE.criticGeneration;
    const actionSince = LIVE.replayMode ? 0 : rlLiveActionSince;
    const criticSince = LIVE.criticValue.length;
    const response = await apiGet(`/api/rl/series?since=${actionSince}&critic_since=${criticSince}`);
    if (criticGeneration !== LIVE.criticGeneration) return;
    if (!LIVE.replayMode) rlLiveActionSince = Number(response.n || 0);
    const critic = response.critic || {};
    if (Number(critic.n || 0) < LIVE.criticValue.length) {
      LIVE.criticTimestamp = [];
      LIVE.criticValue = [];
      LIVE.criticSource = [];
    }
    LIVE.criticTimestamp.push(...(critic.timestamp || []));
    LIVE.criticValue.push(...(critic.value || []));
    LIVE.criticSource.push(...(critic.source || []).map((source, index) => (
      source === "replay" ? replayCriticSource(critic.timestamp[index]) : source
    )));
    updateScrub();
    drawLiveCharts();
  } finally {
    seriesPolling = false;
  }
}

function queueReplayCritic(frame) {
  if (S.ACTIVE_TAB !== "rl" || LIVE.replayOwner !== "rl" || !S.STATUS.rl?.critic_connected) return;
  const index = Math.max(0, Math.round(Number(frame) || 0));
  if (index === lastQueuedReplayFrame) return;
  lastQueuedReplayFrame = index;
  rlCriticPendingFrame = index;
  flushRlCriticQueue();
}

window.addEventListener("eva:replay-frame", (event) => {
  if (event.detail && event.detail.owner === "rl") queueReplayCritic(event.detail.frame);
});

$("rl-b-setup").onclick = () => {
  rlSetupRequestPending = false;
  requestRlSetup();
};
$("rl-b-run").onclick = () => {
  if (LIVE.replayOwner === "rl") exitReplayMode();
  return apiPost("/api/rl/run");
};
$("rl-b-reset").onclick = () => apiPost("/api/rl/reset");
$("rl-b-intervene").onclick = () => apiPost("/api/rl/intervene");
$("rl-b-accept").onclick = () => apiPost("/api/rl/accept");
$("rl-b-abandon").onclick = () => apiPost("/api/rl/abandon");
$("rl-b-save").onclick = () => {
  if (rlSaveSetupPending) return;
  rlSaveSetupPending = true;
  renderRlStatus(S.STATUS || {});
  return apiPost("/api/rl/save").then((response) => {
    if (!response || response.ok === false) {
      rlSaveSetupPending = false;
      renderRlStatus(S.STATUS || {});
    }
    return response;
  });
};
$("rl-b-replay").onclick = replaySelectedEpisode;
$("rl-b-qc-pass").onclick = () => submitRlQc("pass");
$("rl-b-qc-fail").onclick = () => submitRlQc("fail");
$("rl-b-qc-note-save").onclick = () => submitRlQc("");
$("rl-save-expand").onclick = () => {
  S.rlSaveExpanded = !S.rlSaveExpanded;
  renderRlStatus(S.STATUS || {});
};
$("rl-hil-enable").onchange = () => apiPost("/api/rl/hil_enabled", {
  enabled: $("rl-hil-enable").checked,
});

export { pollRlSeries, renderRlConfig, renderRlStatus };
