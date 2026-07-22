// collect.js: data-collection tab (collect) + QC/annotation review &
// stage-video playback control (review).
import { $, LIVE, S, apiGet, apiPost, clientTrace } from "./core.js";
import { updateScrub } from "./charts.js";
import { collectTaskValue, setPanel, applyStatus, uiMode } from "./run.js";
import {
  exitReplayMode, loadReviewPlayback, refreshCameraStreams, replayStop,
} from "./replay.js";
import { setActiveTab } from "./main.js";

// ===== collect =====

async function startCollectFromTab() {
    if (S.reviewKind === "collect") {
      returnReviewToLive();
    }
    const task = collectTaskValue();
    if (task && task !== S.STATUS.selected_collect_task) {
      S.STATUS.selected_collect_task = task;
      await apiPost("/api/select_collect_task", { task });
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

function collectTone(item) {
    if (item.status === "queued") return "cq-queued";
    if (item.status === "saving") return "cq-busy";
    if (item.status === "failed") return "cq-fail";
    if (item.qc_verdict === "pass") return "cq-ok";
    if (item.qc_verdict === "fail") return "cq-fail";
    if (item.quality === "red") return "cq-fail";
    return "cq-queued";
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
    if (reviewTask !== collectTaskValue()) return null;
    const collect = S.STATUS.collect || {};
    const items = (collect.episodes || []).concat(collect.queue || []);
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
        if (episode === S.collectReplayEpisode) tile.classList.add("selected");
        tile.onpointerdown = (event) => selectCollectEpisodePointer(event, item);
        tile.onclick = () => selectCollectEpisode(item);
      } else {
        tile.title = `episode ${item.episode_index} · ${item.status}`;
      }
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
      row.className = `collect-row ${episode != null ? "replayable" : ""}${episode === S.collectReplayEpisode ? " selected" : ""}`;
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
    const episodes = rollout.episodes || [];
    const queue = rollout.queue || [];
    const items = episodes.concat(queue);
    const hideRolloutSave = ["sim", "step"].includes(uiMode(S.STATUS.cli_mode)) &&
      items.length === 0 && S.reviewKind !== "rollout";
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
    $("rollout-save-count").textContent = `${episodes.length}/${items.length}`;
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

    renderRolloutSaveTiles(items);
    renderRolloutSaveList(items);

    if (S.rolloutSaveEpisode == null) {
      $("rollout-review-title").textContent = "no rollout selected";
    }
  }

function renderCollect() {
    if (!$("collect-control-col")) return;
    const collect = S.STATUS.collect || {};
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
    const hasPrompt = !!prompt;
    const queueFull = collect.pipeline_state === "QUEUE_FULL";
    const episodes = collect.episodes || [];
    const queue = collect.queue || [];
    const items = episodes.concat(queue);
    const progress = Math.max(0, Math.min(1, Number(collect.progress || 0)));

    const collectFps = S.CFG && S.CFG.collection ? S.CFG.collection.fps : null;
    $("collect-fps").textContent = collectFps ? `${collectFps} FPS` : "";
    $("collect-count").textContent = `${episodes.length}/${items.length}`;
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
    if (armLabel) armLabel.textContent = S.collectArmEnabled ? "ARM ON" : "ARM OFF";

    const toggle = $("b-collect-toggle");
    toggle.disabled = toggleBusy ||
      (collecting ? false : (!enabled || !hasPrompt || queueFull || !S.collectArmEnabled));
    toggle.classList.toggle("recording", collecting);
    toggle.classList.toggle("primary", !collecting);
    toggle.querySelector(".rec-label").textContent = collecting ? "END / SAVE" : "START RECORD";
    $("b-collect-cancel").disabled = !collecting;
    const selectedEpisode = selectedCollectEpisodeItem();
    const selectedEpisodeSaved = savedEpisodeId(selectedEpisode) != null;
    $("b-collect-qc-pass").disabled = !enabled || !selectedEpisodeSaved;
    $("b-goto-qc").disabled = !enabled || !selectedEpisodeSaved;
    $("b-collect-note-save").disabled = S.collectReplayEpisode == null;

    const recordState = collecting || (hasPrompt && !S.collectArmEnabled)
      ? "active"
      : (hasPrompt ? "done" : "pending");
    setPanel("collect-panel-task", enabled && hasPrompt ? "done" : "active");
    setPanel("collect-panel-record", recordState);
    const queueEnabled = enabled && (S.collectQueueEnabled || episodes.length > 0 || queue.length > 0);
    setPanel("collect-panel-queue", queue.length ? "active" : (queueEnabled ? "done" : "pending"));
    setPanel("collect-panel-replay", S.collectReplayEpisode == null ? "pending" : "active");

    renderCollectTiles(items);
    renderCollectList(items);

    const replayStatus = $("collect-replay-status");
    if (S.reviewKind === "collect" && LIVE.replayOwner === "collect") {
      replayStatus.textContent = LIVE.replayError
        ? `episode ${S.collectReplayEpisode} · error · ${LIVE.replayError}`
        : (LIVE.replayLoading
            ? `episode ${S.collectReplayEpisode} · loading`
            : `episode ${S.collectReplayEpisode} · review`);
      replayStatus.style.display = S.ACTIVE_TAB === "collect" ? "" : "none";
    } else if (S.collectReplayEpisode != null) {
      replayStatus.textContent = `episode ${S.collectReplayEpisode} selected`;
      replayStatus.style.display = S.ACTIVE_TAB === "collect" ? "" : "none";
    } else if (S.collectReplayEpisode == null) {
      replayStatus.textContent = "";
      replayStatus.style.display = "none";
    }
  }

function dotClass(kind) { return "dot " + kind; }

// ===== review =====

let reviewDatasetDir = "";

let reviewTask = "";

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

function reviewActiveInCurrentTab() {
    return (S.reviewKind === "collect" && S.ACTIVE_TAB === "collect") ||
      (S.reviewKind === "rollout" && S.ACTIVE_TAB === "debug");
  }

function clearReviewPlayback() {
    reviewRequestId += 1;
    S.reviewKind = "";
    reviewDatasetDir = "";
    reviewTask = "";
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
    applyStatus(await apiGet("/api/status"));
  }

async function submitEpisodeNote(kind) {
    const episode = kind === "rollout" ? S.rolloutSaveEpisode : S.collectReplayEpisode;
    const status = kind === "rollout" ? $("rollout-save-err") : $("collect-qc-status");
    if (episode == null) { if (status) status.textContent = "✗ select an episode first"; return; }
    S.reviewKind = kind;
    reviewDatasetDir = reviewDatasetFor(kind);
    reviewEpisodeId = episode;
    if (status) status.textContent = "saving…";
    const r = await apiPost(episodeQcEndpoint(kind), {
      dataset_dir: reviewDatasetDir,
      task: reviewTask,
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
  saveAnnotation, submitEpisodeNote, submitEpisodeQc, submitQc,
};
