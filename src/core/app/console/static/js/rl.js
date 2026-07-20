// rl.js: RL rollout/HIL controls, saved-episode replay, and Critic telemetry.
import { $, LIVE, S, apiGet, apiPost } from "./core.js";
import { buildLiveDims, drawLiveCharts, resetLiveSeries, updateScrub } from "./charts.js";
import { exitReplayMode, loadReviewPlayback } from "./replay.js";
import { renderRlGripper, updateGuide } from "./run.js";

let selectedEpisode = null;
let seriesPolling = false;
let lastQueuedReplayFrame = -1;
let replayRequestId = 0;

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

function syncModelLink(id, selected, connected, error) {
  const link = $(id);
  if (!selected) link.textContent = "NOT SELECTED";
  else if (connected) link.textContent = "LINKED";
  else if (error) link.textContent = "ERROR";
  else link.textContent = "SELECTED · NOT CONNECTED";
  link.classList.toggle("selected", !!selected);
  link.title = error || "";
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
  task.onchange = () => {
    S.rlTask = task.value;
    task.classList.toggle("has-value", !!task.value);
    syncTaskChip(task);
    apiPost("/api/rl/select_task", { task: task.value });
    updateGuide();
  };
  syncTaskChip(task);

  const policy = $("rl-policy-list");
  fillSelect(policy, cfg.policies || [], "SELECT POLICY", S.rlPolicy);
  policy.onchange = () => {
    S.rlPolicy = policy.value;
    policy.classList.toggle("has-value", !!policy.value);
    apiPost("/api/rl/select_policy", { slot: Number(policy.value) });
    updateGuide();
  };

  const critic = $("rl-critic-list");
  fillSelect(critic, cfg.critics || [], "SELECT CRITIC", S.rlCritic);
  critic.onchange = () => {
    S.rlCritic = critic.value;
    critic.classList.toggle("has-value", !!critic.value);
    apiPost("/api/rl/select_critic", { slot: Number(critic.value) });
    updateGuide();
  };

  $("rl-mode").textContent = String(cfg.cli_mode || "—").toUpperCase();
  $("rl-strategy").textContent = String(cfg.inference_strategy || "—").toUpperCase();
  $("rl-data-format").value = cfg.data && cfg.data.format === "lerobot" ? "lerobot" : "";
  $("rl-dataset-dir").textContent = (cfg.data && cfg.data.dataset_dir) || "—";
  $("rl-dataset-dir").title = (cfg.data && cfg.data.dataset_dir) || "";
  renderRlStatus(S.STATUS || {});
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
  if (item.qc_verdict === "fail" || item.quality === "red") return "cq-fail";
  if (item.qc_verdict === "pass") return "cq-ok";
  return "cq-queued";
}

function episodeIssue(item) {
  if (item.error) return item.error;
  if (item.qc_verdict) return `qc ${item.qc_verdict}`;
  const issues = item.quality_issues || [];
  return issues.length ? issues.map((issue) => issue.code || issue.detail || "issue").join(", ") : (item.status || "ok");
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

function selectSavedEpisode(item, items) {
  const id = episodeId(item);
  if (id == null) return;
  selectedEpisode = id;
  renderSavedEpisodes(items);
  renderSavedEpisodeList(items);
  $("rl-b-replay").disabled = false;
  replaySelectedEpisode();
}

async function replaySelectedEpisode() {
  if (selectedEpisode == null) return;
  const requestId = ++replayRequestId;
  if (LIVE.replayMode && LIVE.replayOwner === "rl") exitReplayMode();
  const rollout = (S.STATUS && S.STATUS.rollout) || {};
  lastQueuedReplayFrame = -1;
  const response = await apiPost("/api/rl/review_episode", {
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
}

function renderRlStatus(status) {
  if (!S.CFG || !S.CFG.rl || !S.CFG.rl.enabled) return;
  const rl = status.rl || {};
  const rollout = status.rollout || {};
  if (status.selected_task && rl.active) S.rlTask = status.selected_task;
  if (rl.selected_policy_slot != null) S.rlPolicy = String(rl.selected_policy_slot);
  if (rl.selected_critic_slot != null) S.rlCritic = String(rl.selected_critic_slot);

  const task = $("rl-task-list");
  const policy = $("rl-policy-list");
  const critic = $("rl-critic-list");
  if (task && S.rlTask && task.value !== S.rlTask) task.value = S.rlTask;
  if (policy && S.rlPolicy !== "" && policy.value !== S.rlPolicy) policy.value = S.rlPolicy;
  if (critic && S.rlCritic !== "" && critic.value !== S.rlCritic) critic.value = S.rlCritic;
  [task, policy, critic].forEach((select) => {
    if (select) select.classList.toggle("has-value", !!select.value);
  });
  if (task) syncTaskChip(task);
  syncModelLink("rl-policy-link", !!S.rlPolicy, !!status.policy_connected, status.policy_error);
  syncModelLink("rl-critic-link", !!S.rlCritic, !!rl.critic_connected, rl.critic_error);

  const selected = !!S.rlTask && S.rlPolicy !== "" && S.rlCritic !== "";
  const setup = !!status.is_setup_done && !!rl.critic_connected;
  const setupBusy = !!status.setup_stage;
  $("rl-b-setup").disabled = !selected || setupBusy || status.session_status === "running";
  $("rl-setup-state").textContent = status.last_error
    ? `ERROR · ${status.last_error}`
    : (setup ? "ROBOT + POLICY + CRITIC READY" : (setupBusy ? String(status.setup_stage).toUpperCase() : "Ready to setup"));

  const running = status.session_status === "running";
  const intervention = !!status.rollout_intervention_active;
  const hilSupported = !!status.hil_supported;
  const hilEnabled = !!status.rollout_intervention_enabled;
  $("rl-hil-enable").checked = hilEnabled;
  $("rl-hil-enable").disabled = !setup || !hilSupported || intervention;
  $("rl-hil-label").textContent = hilSupported ? (hilEnabled ? "HIL ON" : "HIL OFF") : "HIL N/A";
  $("rl-hil-gate").classList.toggle("on", hilEnabled);
  $("rl-hil-gate").classList.toggle("disabled", $("rl-hil-enable").disabled);
  $("rl-b-run").disabled = !setup || running || intervention;
  $("rl-run-label").textContent = status.step_index > 0 ? "CONTINUE ▶▶" : "RUN ▶";
  $("rl-b-run").classList.toggle("recording", running);
  $("rl-b-intervene").disabled = !running;
  $("rl-b-intervene").textContent = hilEnabled ? "INTERVENE ■" : "STOP ■";
  $("rl-b-accept").disabled = !intervention;
  $("rl-b-abandon").disabled = !intervention;
  $("rl-run-error").textContent = status.last_error || rl.critic_error || "";

  const progress = Math.max(0, Math.min(1, Number(rollout.progress || 0)));
  const items = (rollout.episodes || []).concat(rollout.queue || []);
  $("rl-save-count").textContent = `${(rollout.episodes || []).length}/${items.length}`;
  $("rl-save-eta").textContent = rollout.eta_sec == null ? "—" : `${Number(rollout.eta_sec).toFixed(1)}s`;
  $("rl-save-hint").style.display = items.some((item) => episodeId(item) != null) ? "" : "none";
  $("rl-save-progress").style.width = `${progress * 100}%`;
  $("rl-save-pipeline").textContent = String(rollout.pipeline_state || "IDLE").toUpperCase();
  $("rl-save-pipeline").dataset.state = String(rollout.pipeline_state || "idle").toUpperCase();
  $("rl-b-save").disabled = !rollout.enabled || !rollout.save_ready || intervention;
  $("rl-b-replay").disabled = selectedEpisode == null;
  $("rl-save-dir").textContent = rollout.dataset_dir ? `saved to ${rollout.dataset_dir}` : "";
  $("rl-save-error").textContent = rollout.save_blocked_by_intervention
    ? "accept or abandon the active intervention before saving"
    : "";
  renderSavedEpisodes(items);
  renderSavedEpisodeList(items);
  renderRlGripper(setup);
  updateGuide();
}

async function pollRlSeries() {
  if (S.ACTIVE_TAB !== "rl" || seriesPolling) return;
  seriesPolling = true;
  try {
    const actionSince = LIVE.replayMode ? 0 : LIVE.n;
    const criticSince = LIVE.criticValue.length;
    const response = await apiGet(`/api/rl/series?since=${actionSince}&critic_since=${criticSince}`);
    if (!LIVE.replayMode) {
      if (Number(response.n || 0) < LIVE.n) resetLiveSeries();
      for (let i = 0; i < (response.timestamp || []).length; i++) {
        LIVE.timestamp.push(response.timestamp[i]);
        LIVE.action.push(response.action[i]);
        LIVE.state.push(response.state[i]);
        LIVE.controlSource.push(response.control_source[i]);
        LIVE.intervention.push(response.intervention[i]);
        LIVE.interventionSegmentIndex.push(response.intervention_segment_index[i]);
      }
      LIVE.n = LIVE.timestamp.length;
      if (!LIVE.dimsBuilt && LIVE.n) buildLiveDims();
      if (LIVE.n) LIVE.cursor = LIVE.n - 1;
    }
    const critic = response.critic || {};
    if (Number(critic.n || 0) < LIVE.criticValue.length) {
      LIVE.criticTimestamp = [];
      LIVE.criticValue = [];
    }
    LIVE.criticTimestamp.push(...(critic.timestamp || []));
    LIVE.criticValue.push(...(critic.value || []));
    updateScrub();
    drawLiveCharts();
  } finally {
    seriesPolling = false;
  }
}

function queueReplayCritic(frame) {
  if (S.ACTIVE_TAB !== "rl" || LIVE.replayOwner !== "rl") return;
  const index = Math.max(0, Math.round(Number(frame) || 0));
  if (index === lastQueuedReplayFrame) return;
  lastQueuedReplayFrame = index;
  apiPost("/api/rl/replay_critic", { frame: index });
}

window.addEventListener("eva:replay-frame", (event) => {
  if (event.detail && event.detail.owner === "rl") queueReplayCritic(event.detail.frame);
});

$("rl-b-setup").onclick = () => apiPost("/api/rl/setup");
$("rl-b-run").onclick = () => {
  if (LIVE.replayOwner === "rl") exitReplayMode();
  return apiPost("/api/rl/run");
};
$("rl-b-intervene").onclick = () => apiPost("/api/rl/intervene");
$("rl-b-accept").onclick = () => apiPost("/api/rl/accept");
$("rl-b-abandon").onclick = () => apiPost("/api/rl/abandon");
$("rl-b-save").onclick = () => apiPost("/api/rl/save");
$("rl-b-replay").onclick = replaySelectedEpisode;
$("rl-save-expand").onclick = () => {
  S.rlSaveExpanded = !S.rlSaveExpanded;
  renderRlStatus(S.STATUS || {});
};
$("rl-hil-enable").onchange = () => apiPost("/api/rl/hil_enabled", {
  enabled: $("rl-hil-enable").checked,
});

export { pollRlSeries, renderRlConfig, renderRlStatus };
