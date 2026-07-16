// main.js: app entry — wires DOM events, exposes inline handlers, boots polling
// (main); tab switching + active-tab thumb + per-tab render dispatch (tabs).
import { $, LIVE, S, apiGet, apiPost } from "./core.js";
import { closeChartModal, drawLiveCharts, liveDimsAll, onScrubInput, openChartModal, resetLiveSeries } from "./charts.js";
import { applyTune, applyManualTune, renderConfig, manualConnect, manualDisconnect, manualDispatchToggle, enterManualSim, applyStatus, pauseSetup, replayIsLocalMode, resumeSetup, retrySetup, startRunFromDebug, updateGuide } from "./run.js";
import { collectConfigured, renderCollect, renderRolloutSave, returnReviewToLive, startCollectFromTab, saveAnnotation, submitEpisodeNote, submitEpisodeQc, submitQc, clearReviewPlayback, reviewActiveInCurrentTab } from "./collect.js";
import { evalEnabled, evalReset, evalSetup, evalRunToggle, evalResumeOnEnter, submitEvalScore, loadEvalResults, renderEvalSelectors, loadResultsAll, tpSeek, tpToggle, trialPopClose } from "./eval.js";
import { replayPlay, replayStop, replayToggle, seekReplay, loop, pollFrame, pollScene, refreshCameraStreams, exitReplayMode } from "./replay.js";
import { refreshCameras, reloadCameras, renderCalibratePanels, applyBoard, startCalibrate, resetCalibrate, setMode, captureSample, currentCalibCam, solveAll, saveCalibration, moveToPose, addPoseFromCurrent, commitEditPose, cancelEditPose, estop } from "./calibrate.js";

// ===== tabs =====
function moveTabThumb() {
    const thumb = $("tab-thumb");
    const active = document.querySelector(".tab.active");
    if (!thumb || !active) return;
    thumb.style.setProperty("--thumb-x", active.offsetLeft + "px");
    thumb.style.setProperty("--thumb-w", active.offsetWidth + "px");
    thumb.classList.add("ready");
  }

function relocateCanvas(tab) {
    const stage = $("stage");
    if (!stage) return;
    // RESULT is a browser-style tab with no live stage; park the stage in the hidden
    // DEBUG view so its single WebGL context stays alive. EVAL hosts the live stage in
    // its right column (#eval-replay-col), showing the same live 3D + cameras as DEBUG.
    let host;
    if (tab === "result") host = $("view-debug");
    else if (tab === "eval") host = $("eval-replay-col");
    else host = $("view-" + tab);
    if (host && stage.parentElement !== host) host.appendChild(stage);
    const showSeries = tab === "replay" ||
      (tab === "collect" && LIVE.replayOwner === "collect");
    stage.classList.toggle("no-series", !showSeries);
    if (window.Scene3D && Scene3D.resize) requestAnimationFrame(() => Scene3D.resize());
    if (typeof drawLiveCharts === "function") requestAnimationFrame(() => drawLiveCharts());
  }

function borrowDebugPanels() {}

function restoreDebugPanels() {}

function disarmCollectArm() {
    if (!S.collectArmEnabled) return;
    S.collectArmEnabled = false;
    S.collectToggleBusy = null;
  }

function afterWindowLoad(fn) {
    if (document.readyState === "complete") {
      fn();
      return;
    }
    window.addEventListener("load", fn, { once: true });
  }

function closeMediaStreams() {
    document.querySelectorAll("img.cam").forEach((img) => {
      img.removeAttribute("src");
    });
    document.querySelectorAll("video.cam, video.tp-cam").forEach((video) => {
      video.pause();
      video.removeAttribute("src");
      video.load();
    });
  }

// #trial-pop is a single shared node used only by RESULT now: its detail view docks it
// into #rv-detail-replay; anything else parks it on .workspace as a floating popup
// fallback. Move it to the right host for the active tab, dropping .open so no stale
// popup lingers. (EVAL no longer uses it — its right column shows the live #stage.)
function relocateTrialPop(tab) {
    const pop = $("trial-pop");
    if (!pop) return;
    const host = tab === "result" ? $("rv-detail-replay")
      : document.querySelector(".workspace");
    if (host && pop.parentElement !== host) {
      pop.classList.remove("open");
      host.appendChild(pop);
    }
  }

function setActiveTab(tab) {
    if (tab !== "collect") disarmCollectArm();
    const leavingCollectReview = S.ACTIVE_TAB === "collect" && tab !== "collect" &&
      LIVE.replayOwner === "collect";
    S.ACTIVE_TAB = tab;
    if (S.reviewKind && !reviewActiveInCurrentTab()) {
      clearReviewPlayback();
    }
    if (leavingCollectReview) exitReplayMode();
    // Leaving REPLAY: drop the one-shot replay series + scrub clock so DEBUG/MANUAL/
    // EVAL start from a clean live buffer instead of replaying the loaded episode. The
    // backend tab_switch also unmounts replay_source whenever the destination isn't
    // REPLAY, so a loaded QC episode can't hijack EVAL/DEBUG via is_replay().
    if (tab !== "replay" && (LIVE.replayMode || S.replaySeriesKey !== null)) {
      S.replaySeriesKey = null;
      S._replayModePushed = false;
      exitReplayMode();
    }
    document.querySelectorAll(".tab").forEach((x) => {
      x.classList.toggle("active", x.dataset.tab === tab);
    });
    moveTabThumb();
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    // COLLECT keeps the DEBUG canvas/observation layout but owns a separate left
    // control column. REPLAY has its own control view and only reuses the WebGL canvas.
    const viewTab = tab === "collect" ? "debug" : tab;
    $("view-" + viewTab).classList.add("active");
    if (viewTab === "debug") {
      $("view-debug").classList.toggle("collect-mode", tab === "collect");
    }
    relocateCanvas(viewTab);
    relocateTrialPop(tab);
    // EVAL borrows DEBUG's GRIPPER panel; any other tab restores it to DEBUG first so it
    // renders in its native column.
    if (tab === "eval") borrowDebugPanels();
    else restoreDebugPanels();
    // The MJPEG <img>s persist across tabs; force them to reconnect so a tab whose
    // source has no live frames doesn't keep painting the previous tab's last frame.
    refreshCameraStreams();
    renderConfig();
    // MANUAL opens straight into SIM-debug: entering the tab activates manual mode
    // so the sliders drive the sim/3D preview immediately. Connecting to the REAL
    // robot is a separate, optional step gated on the backend's live link state.
    if (tab === "manual") enterManualSim();
    if (tab === "calibrate") renderCalibratePanels();
    if (tab === "eval") { renderEvalSelectors(); loadEvalResults().then(evalResumeOnEnter); }
    if (tab === "result") { loadResultsAll(); }
    updateGuide();
    renderCollect();
  }

// ===== main =====

"use strict";

// Expose handlers referenced by inline on* attributes in index.html.
Object.assign(window, { tpToggle, tpSeek, trialPopClose, replayToggle });

async function boot() {
    S.CFG = await apiGet("/api/config");
    renderConfig();
    refreshCameras();
    // EVAL/RESULT use inline onclick handlers; expose them.
    window.tpToggle = tpToggle; window.tpSeek = tpSeek;
    window.trialPopClose = trialPopClose;
    const s = await apiGet("/api/status");
    applyStatus(s);
    if (!collectConfigured()) {
      document.querySelector('.tab[data-tab="collect"]').classList.add("disabled");
    }
    // An eval config opens straight on the EVAL tab. Without one the console still boots on
    // DEBUG, but EVAL/RESULT stay reachable as a read-only viewer over recorded results.
    if (evalEnabled()) { setActiveTab("eval"); }
    // stage scrub bar + per-chart ALL toggles
    $("scrub-range").addEventListener("input", (e) => onScrubInput(e.target.value));
    $("action-all").addEventListener("click", () => liveDimsAll("a"));
    $("state-all").addEventListener("click", () => liveDimsAll("s"));
    // expand a chart into the big-view modal
    $("action-expand").addEventListener("click", () => openChartModal("a"));
    $("state-expand").addEventListener("click", () => openChartModal("s"));
    $("chart-modal-all").addEventListener("click", () => { if (S.chartModalWhich) liveDimsAll(S.chartModalWhich); });
    $("chart-modal-close").addEventListener("click", closeChartModal);
    $("chart-modal").addEventListener("click", (e) => { if (e.target === $("chart-modal")) closeChartModal(); });
    // keep the 3D + charts crisp when the stage row geometry changes
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        if (window.Scene3D && Scene3D.resize) Scene3D.resize();
        drawLiveCharts();
      });
      ro.observe($("stage"));
    }
    resetLiveSeries();
    requestAnimationFrame(moveTabThumb);
    window.addEventListener("resize", moveTabThumb);
    window.addEventListener("pagehide", closeMediaStreams);
    window.addEventListener("beforeunload", closeMediaStreams);
    loop(async () => { try { applyStatus(await apiGet("/api/status")); } catch (e) {} }, 250);
    afterWindowLoad(() => loop(pollFrame, 200));
    loop(pollScene, 80);
  }

document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      if (t.classList.contains("disabled")) return;
      const tab = t.dataset.tab;
      if (tab === S.ACTIVE_TAB) return;
      // A direct tab click is plain browsing, so clear QC mode here.
      S.qcMode = false;
      S.pendingQcLoad = null;
      // Soft reset on every tab switch. COLLECT teleop is armed only after the
      // local motion gate is switched on; START RECORD still controls recording.
      if (tab !== "collect") disarmCollectArm();
      apiPost("/api/tab_switch", {
        tab,
        collect_teleop_armed: tab === "collect" && S.collectArmEnabled,
      });
      // Leaving MANUAL: drop both the SIM-debug flag and any real-robot link so
      // re-entering starts fresh (SIM debug auto-arms, REAL needs CONNECT again).
      S.manualActive = false;
      S.realRequested = false;
      S.realConnected = false;
      S.manualDispatching = false;
      S.manualDispatchStopPending = false;
      // Switching away mid-setup: abort the in-flight setup (halt interrupts the running
      // reset/warmup motion and clears it) so the new tab starts from a clean idle state
      // instead of finishing a setup for the tab we just left. _setupFired stays true so
      // it doesn't immediately re-fire; clear any pause latch so the new tab can auto-setup.
      if (S.STATUS && S.STATUS.setup_stage) { apiPost("/api/halt"); S._setupFired = true; }
      S._setupPaused = false;
      setActiveTab(tab);
    });
  });
$("b-run").onclick    = () => {
    if (S.runToggleBusy !== null) return;
    const live = S.STATUS.session_status === "running";
    S.runToggleBusy = !live;
    applyStatus(S.STATUS);
    setTimeout(() => {
      if (S.runToggleBusy !== null) { S.runToggleBusy = null; applyStatus(S.STATUS); }
    }, 1500);
    return live ? apiPost("/api/operator_action", { intent: "start" }) : startRunFromDebug();
  };
$("b-reset").onclick  = () => apiPost("/api/reset");
$("b-step").onclick   = () => apiPost("/api/step_infer");
$("b-tune-apply").onclick = applyTune;
$("b-commit").onclick = () => apiPost("/api/step_commit");
$("b-step-halt").onclick = () => apiPost("/api/halt");
$("b-step-reset").onclick = () => apiPost("/api/reset");
$("b-replay-run").onclick    = () => {
    if (replayIsLocalMode()) {
      LIVE.playing ? replayStop() : replayPlay();
      return;
    }
    return S.STATUS.session_status === "running" ? apiPost("/api/halt") : apiPost("/api/run");
  };
$("b-replay-reset").onclick  = () => {
    if (replayIsLocalMode()) {
      replayStop();
      seekReplay(0);
      return;
    }
    return apiPost("/api/reset");
  };
$("be-run").onclick    = () => evalRunToggle();
$("be-setup").onclick  = () => evalSetup(true);
$("be-reset").onclick  = () => evalReset();
$("be-submit").onclick = () => submitEvalScore();
$("b-collect-toggle").onclick = () => {
    if (S.collectToggleBusy !== null) return;
    const live = !!(S.STATUS.collect && S.STATUS.collect.collecting);
    if (!live && !S.collectArmEnabled) {
      renderCollect();
      updateGuide();
      return;
    }
    if (live) S.collectQueueEnabled = true;
    S.collectToggleBusy = !live;
    renderCollect();
    // Normal release is by the status poll once `collecting` matches the target. This
    // fallback covers a backend refusal (queue full, logging off) where it never flips,
    // so the button can't latch disabled forever.
    setTimeout(() => {
      if (S.collectToggleBusy !== null) { S.collectToggleBusy = null; renderCollect(); }
    }, 1500);
    return live ? apiPost("/api/operator_action", { intent: "accept" }) : startCollectFromTab();
  };
$("collect-arm-enable").onchange = () => {
    const enabled = $("collect-arm-enable").checked;
    S.collectArmEnabled = enabled && S.ACTIVE_TAB === "collect";
    if (!enabled) S.collectToggleBusy = null;
    renderCollect();
    updateGuide();
    apiPost("/api/tab_switch", {
      tab: S.ACTIVE_TAB,
      collect_teleop_armed: S.ACTIVE_TAB === "collect" && S.collectArmEnabled,
    });
  };
$("hil-intervention-enable").onchange = () => {
    const enabled = $("hil-intervention-enable").checked;
    if (S.STATUS) {
      S.STATUS.rollout_intervention_enabled = enabled;
      applyStatus(S.STATUS);
    }
    apiPost("/api/rollout_intervention_enabled", { enabled });
  };
$("b-collect-cancel").onclick = () => {
    S.collectQueueEnabled = false;
    renderCollect();
    return apiPost("/api/operator_action", { intent: "cancel" });
  };
$("b-collect-qc-pass").onclick = () => submitEpisodeQc("collect", "pass");
$("b-goto-qc").onclick = () => submitEpisodeQc("collect", "fail");
$("b-collect-note-save").onclick = () => submitEpisodeNote("collect");
$("b-rollout-save").onclick = () => apiPost("/api/rollout_save");
$("b-rollout-intervention-abandon").onclick = () => apiPost("/api/operator_action", { intent: "cancel" });
$("b-rollout-qc-pass").onclick = () => submitEpisodeQc("rollout", "pass");
$("b-rollout-qc-fail").onclick = () => submitEpisodeQc("rollout", "fail");
$("review-return-live").onclick = returnReviewToLive;
$("replay-b-qc-pass").onclick = () => submitQc("pass");
$("replay-b-qc-fail").onclick = () => submitQc("fail");
$("replay-b-anno-save").onclick = () => saveAnnotation();
$("collect-queue-toggle").onclick = () => {
    S.collectQueueExpanded = !S.collectQueueExpanded;
    renderCollect();
  };
$("rollout-save-queue-toggle").onclick = () => {
    S.rolloutSaveQueueExpanded = !S.rolloutSaveQueueExpanded;
    renderRolloutSave();
  };
document.querySelector("#panel-setup .auto-setup-row").onclick = () => {
    if ($("panel-setup").dataset.st === "error") retrySetup();
  };
const replaySetupRow = document.querySelector("#replay-panel-config .auto-setup-row");
if (replaySetupRow) {
  replaySetupRow.onclick = () => {
    if ($("replay-panel-config").dataset.st === "error") retrySetup();
  };
}
document.querySelector("#eval-panel-setup .auto-setup-row").onclick = () => {
    if ($("eval-panel-setup").dataset.st === "error") evalSetup(true);
  };
$("b-setup-pause").onclick = pauseSetup;
$("b-setup-resume").onclick = resumeSetup;
$("b-setup-retry").onclick = retrySetup;
if ($("b-replay-setup-pause")) $("b-replay-setup-pause").onclick = pauseSetup;
if ($("b-replay-setup-resume")) $("b-replay-setup-resume").onclick = resumeSetup;
if ($("b-replay-setup-retry")) $("b-replay-setup-retry").onclick = retrySetup;
$("bm-connect").onclick = () => (S.realRequested ? manualDisconnect() : manualConnect());
$("bm-send").onclick = manualDispatchToggle;
$("bm-home").onclick  = () => apiPost("/api/manual_home");
$("b-manual-tune-apply").onclick = applyManualTune;
$("manual-tune-publish-rate").onkeydown = (e) => { if (e.key === "Enter") applyManualTune(); };
const _calibReload = $("b-calib-reload");
if (_calibReload) _calibReload.onclick = () => reloadCameras();
// CALIBRATE tab wire-ups (07 CALIBRATE)
const _cbBoard = $("b-cal-board-apply"); if (_cbBoard) _cbBoard.onclick = applyBoard;
const _cbStart = $("b-cal-start"); if (_cbStart) _cbStart.onclick = startCalibrate;
const _cbReset = $("b-cal-reset"); if (_cbReset) _cbReset.onclick = resetCalibrate;
const _cbCapture = $("b-cal-capture"); if (_cbCapture) _cbCapture.onclick = () => captureSample(currentCalibCam());
const _cbSolve = $("b-cal-solve"); if (_cbSolve) _cbSolve.onclick = () => solveAll($("cal-hand-eye-method").value);
const _cbSave = $("b-cal-save"); if (_cbSave) _cbSave.onclick = saveCalibration;
const _cbAdd = $("b-cal-add-current"); if (_cbAdd) _cbAdd.onclick = addPoseFromCurrent;
const _cbPoseSave = $("b-cal-pose-save"); if (_cbPoseSave) _cbPoseSave.onclick = commitEditPose;
const _cbPoseCancel = $("b-cal-pose-cancel"); if (_cbPoseCancel) _cbPoseCancel.onclick = cancelEditPose;
const _cbEStop = $("b-cal-estop"); if (_cbEStop) _cbEStop.onclick = estop;
const _cbSim = $("b-cal-mode-sim"); if (_cbSim) _cbSim.onclick = () => setMode("sim");
const _cbReal = $("b-cal-mode-real"); if (_cbReal) _cbReal.onclick = () => setMode("real");
export { setActiveTab };
boot();
