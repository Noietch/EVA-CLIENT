// run.js: run lifecycle / status / UI-mode / panel orchestration (run);
// config & tuning panel (config); manual sim/connection control (manual).
import { $, LIVE, RUN_CONTROLS, S, apiPost, setCommandMetadata } from "./core.js";
import { updateScrub } from "./charts.js";
import { renderDeviceSettings } from "./device.js";
import {
  collectEnabled, dotClass, loadAnnotation, renderCollect, renderRolloutSave,
  selectCollectionDataset,
} from "./collect.js";
import { applyEvalStatus, evalCfg, evalEnabled } from "./eval.js";
import { maybeSyncReplayPlayer, loadMountedReplaySeries, replayStop } from "./replay.js";

// ===== run =====

function formatImageHz(hz) {
    const value = Number(hz);
    return Number.isFinite(value) ? `${value.toFixed(1)} hz` : "— hz";
  }

function startRunFromDebug() {
    apiPost("/api/run");
  }

function replayIsLocalMode() {
    if (S.ACTIVE_TAB !== "replay") return false;
    const m = uiMode((S.STATUS && S.STATUS.cli_mode) || "sim");
    return m !== "real";
  }

function applyRunControlStatus(ids, s) {
    if (ids.run === "b-replay-run") {
      const loaded = LIVE.replayMode && LIVE.n > 0 && !LIVE.replayLoading;
      const ready = !!s.is_setup_done;
      const atEnd = LIVE.cursor >= LIVE.n - 1;
      const resumable = !LIVE.playing && LIVE.cursor > 0 && !atEnd;
      const run = $(ids.run);
      const stepGroup = $(ids.stepGroup);
      const reset = $("b-replay-reset");
      $(ids.runGroup).style.display = "grid";
      if (stepGroup) stepGroup.style.display = "none";
      if (replayIsLocalMode()) {
        run.textContent = LIVE.playing ? "STOP ■" : (resumable ? "CONTINUE ▶▶" : "REPLAY ▶");
        run.classList.toggle("primary", !LIVE.playing);
        run.classList.toggle("danger", LIVE.playing);
        run.disabled = !loaded || !ready;
        if (reset) reset.disabled = !loaded || LIVE.playing;
        return;
      }
      const running = s.session_status === "running";
      const continueRun = !running && (s.step_index || 0) > 0;
      run.textContent = running ? "STOP ■" : (continueRun ? "CONTINUE ▶▶" : "REPLAY ▶");
      run.classList.toggle("primary", !running);
      run.classList.toggle("danger", running);
      run.disabled = !loaded || (!running && !ready);
      if (reset) reset.disabled = !loaded || running;
      return;
    }
    const m = uiMode(s.cli_mode);
    if (S.ACTIVE_TAB === "manual") renderControl();
    const isStep = m === "step";
    $(ids.runGroup).style.display = isStep ? "none" : "grid";
    $(ids.stepGroup).style.display = isStep ? "block" : "none";

    const armed = s.pending_chunk != null;
    const ss = $(ids.stepState);
    ss.classList.remove("armed", "ok");
    if (armed) {
      ss.textContent = "SIM preview done · " + (s.pending_chunk || "?") + " actions · press REAL ▶ to dispatch";
      ss.classList.add("armed");
    } else if (s.is_setup_done) {
      ss.textContent = "READY · press SIM ↻ to infer one chunk";
      ss.classList.add("ok");
    } else {
      ss.textContent = "IDLE · SETUP FIRST";
    }

    const ready = s.is_setup_done;
    const running = s.session_status === "running";
    const intervention = !!s.rollout_intervention_active;
    const continueRun = !running && ((s.step_index || 0) > 0 || intervention);
    if (ids.run === "b-run") {
      // Single RUN/STOP toggle (mirrors the collect record toggle): one click runs,
      // the next halts. `running` lags the click by a poll, so hold the button busy
      // until the polled state matches the target to stop a stale double-fire.
      if (S.runToggleBusy !== null && running === S.runToggleBusy) S.runToggleBusy = null;
      const busy = S.runToggleBusy !== null;
      const run = $(ids.run);
      run.querySelector(".rec-label").textContent = running
        ? "STOP ■"
        : (intervention ? "RESUME ▶" : (continueRun ? "CONTINUE ▶▶" : "RUN ▶"));
      run.classList.toggle("recording", running);
      run.classList.toggle("primary", !running);
      run.disabled = busy || (!running && !ready);
      const abandon = $("b-rollout-intervention-abandon");
      if (abandon) {
        abandon.style.display = intervention ? "" : "none";
        abandon.disabled = !intervention;
      }
      $(ids.step).disabled = !ready || running;
      $(ids.commit).disabled = !armed;
      $(ids.stepHalt).disabled = !armed && !running;
      $(ids.stepReset).disabled = false;
      return;
    }
    $(ids.run).textContent = continueRun ? "CONTINUE ▶▶" : "RUN ▶";
    $(ids.run).disabled = !ready || running;
    $(ids.run).classList.toggle("live", running);
    $(ids.halt).disabled = !running;
    $(ids.step).disabled = !ready || running;
    $(ids.commit).disabled = !armed;
    $(ids.stepHalt).disabled = !armed && !running;
    $(ids.stepReset).disabled = false;
  }

function syncHilInterventionEnabled(status) {
    const toggle = $("hil-intervention-enable");
    const label = $("hil-intervention-label");
    if (!toggle || !label) return;
    const enabled = !!status.rollout_intervention_enabled;
    const supported = !!status.hil_supported;
    toggle.checked = enabled;
    toggle.disabled = !supported || !!status.rollout_intervention_active;
    label.textContent = supported ? (enabled ? "HIL ON" : "HIL OFF") : "HIL N/A";
    toggle.title = status.hil_error || "Enable rollout HIL intervention";
    const gate = toggle.closest(".collect-arm-gate");
    if (gate) {
      gate.classList.toggle("on", enabled);
      gate.classList.toggle("disabled", toggle.disabled);
    }
  }

function applyStatus(s) {
    S.STATUS = s;
    if (!s.setup_stage) S.collectHomeBusy = false;
    S.collectArmEnabled = !!s.collection_teleop_armed;
    // RUN follows the live tail; any other state unlocks the scrub bar for history
    // review. Re-entering RUN re-arms following and keeps accumulating. REPLAY owns
    // its own scrub clock (replayMode), so the live-follow toggle is bypassed there.
    const isRunning = s.session_status === "running" ||
      (S.ACTIVE_TAB === "rl" && !!s.rollout_intervention_active);
    if (!LIVE.replayMode) {
      if (isRunning && !LIVE.following) { LIVE.following = true; updateScrub(); }
      else if (!isRunning && LIVE.following) { LIVE.following = false; updateScrub(); }
    }
    // telemetry
    $("t-tx").textContent = s.transport_type || "—";
    $("t-tx-dot").className = dotClass(s.transport_connected ? "ok" : "idle");
    // Policy link is only meaningful when inference is actually used (a strategy is
    // selected, or the EVAL tab is open). REPLAY/MANUAL/COLLECT never hit the policy
    // server, so the chip stays hidden there instead of showing a misleading DOWN.
    const policyRelevant = !!s.selected_strategy || ["eval", "rl"].includes(S.ACTIVE_TAB);
    const gbPolicy = $("gb-policy");
    if (gbPolicy) {
      gbPolicy.style.display = policyRelevant ? "" : "none";
      gbPolicy.textContent = "policy " + (s.policy_connected ? "LINKED" : "DOWN");
      gbPolicy.classList.toggle("err", policyRelevant && !s.policy_connected);
    }
    const st = s.session_status || "unset";
    $("t-state").textContent = st.toUpperCase();
    $("t-state-dot").className = dotClass(st === "running" ? "busy" : (st === "ready" ? "ok" : "idle"));
    $("t-infer").textContent = (s.last_infer_ms || 0) + "ms";
    $("t-step").textContent = s.step_index || 0;
    $("t-elapsed").textContent = ((s.run_elapsed_ms || 0) / 1000).toFixed(1) + "s";
    $("t-image-hz").textContent = formatImageHz(s.image_min_hz);

    // active selections
    mark("prompt-list", "prompt", s.selected_task);
    if (collectTask == null && s.selected_collect_task) {
      collectTask = s.selected_collect_task;
      collectSet = s.selected_collect_set || null;
      collectTaskIndex = Number.isInteger(s.selected_collect_task_index)
        ? s.selected_collect_task_index
        : null;
    }
    if (collectTask == null) {
      const defaultSet = collectTaskSets()[0];
      if (defaultSet && defaultSet.tasks.length) {
        collectSet = defaultSet.name;
        collectTask = defaultSet.tasks[0].prompt;
        collectTaskIndex = 0;
      }
    }
    syncCollectTaskNavigation();
    if (S.ACTIVE_TAB === "replay" || S.ACTIVE_TAB === "eval" || s.replay_loaded || S.CFG.is_replay) {
      const input = $("replay-episode-input");
      if (document.activeElement !== input && input.value === "" && s.replay_loaded) {
        input.value = String(s.replay_episode_id || 0);
      }
      if (s.replay_loaded) {
        $("replay-episode-task").textContent =
          `episode ${s.replay_episode_id} · ${s.replay_task || "∅ no task"}`;
      } else if (S.CFG.is_replay && s.current_episode != null) {
        if (document.activeElement !== input && input.value === "") input.value = String(s.current_episode);
        $("replay-episode-task").textContent =
        `episode ${s.current_episode} · ${s.selected_task || "∅ no task"}`;
      }
      const fpsInput = $("replay-tune-fps");
      if (document.activeElement !== fpsInput && s.replay_fps) {
        fpsInput.value = String(s.replay_fps);
      }
      maybeSyncReplayPlayer(s);
    }
    // default the mode highlight to SIM until the operator picks another
    const modeMark = (!s.cli_mode || s.cli_mode === "select") ? "sim" : uiMode(s.cli_mode);
    mark("mode-list", "mode", modeMark);
    mark("replay-mode-list", "mode", modeMark);
    mark("strategy-list", "strategy", s.selected_strategy);
    syncHilInterventionEnabled(s);
    updateGuide();

    // canvas readout
    {
      $("readout").style.display = "block";
      $("r-step").textContent = s.step_index || 0;
      $("r-chunk").textContent = s.chunk_index || 0;
    }

    const m = uiMode(s.cli_mode);
    // send gating (real-robot MANUAL tab)
    if (m === "manual") {
      // The REAL link is only as real as the backend's live-transport flag. Reflect
      // it whenever the operator has asked to connect, so a dead/absent robot shows
      // "waiting" instead of a fake LIVE and SEND TO REAL stays locked.
      if (S.ACTIVE_TAB === "manual") {
        const live = !!s.transport_connected;
        if (live !== S.realConnected) { S.realConnected = live; renderManualConn(); }
        syncManualDispatchState(s);
      }
      renderManualTarget(s.manual_qpos);
    }

    applyRunControlStatus(RUN_CONTROLS.debug, s);
    applyRunControlStatus(RUN_CONTROLS.replay, s);

    const nextEp = $("replay-b-episode-next");
    if (nextEp) {
      const total = s.replay_n_episodes || 0;
      nextEp.disabled = !(s.replay_loaded && total > 0 && (s.replay_episode_id || 0) + 1 < total);
    }

    $("err").textContent = s.last_error || "";
    $("replay-err").textContent = s.last_error || "";
    renderCollect();
    renderRolloutSave();
    applyEvalStatus(s);
  }

function uiMode(m) { return m === "debug_sim_real" ? "step" : m; }

// Map each selector list to the header chip that mirrors its current value.
const SEL_CHIP = {
    "prompt-list": "prompt-chip",
    "mode-list": "mode-chip",
    "replay-mode-list": "replay-mode-chip",
    "strategy-list": "strategy-chip",
  };

// Read the human label off a .seg: the text-bearing span that is neither the
// 2-digit index (.mk) nor a trailing rate badge (.epb-rate).
function segLabel(seg) {
    const spans = [...seg.querySelectorAll("span")]
      .filter((s) => !s.classList.contains("mk") && !s.classList.contains("epb-rate"));
    return (spans.length ? spans[spans.length - 1] : seg).textContent.trim();
}

// Mirror the active selection of `listId` into its header chip (show only when set).
function syncChip(listId) {
    const host = $(listId);
    let label = "";
    if (host && host.tagName === "SELECT") {
      const opt = host.selectedOptions && host.selectedOptions[0];
      label = opt && opt.value ? (opt.dataset.label || opt.textContent).trim() : "";
      host.classList.toggle("has-value", !!label);
    } else {
      const seg = document.querySelector(`#${listId} .seg.active`);
      label = seg ? segLabel(seg) : "";
    }
    const chip = $(SEL_CHIP[listId]);
    if (!chip) return;
    let v = chip.querySelector(".v");
    if (!v) { v = document.createElement("span"); v.className = "v"; chip.appendChild(v); }
    v.textContent = label;
    chip.title = label;
    chip.classList.toggle("set", !!label);
  }

function mark(listId, attr, val) {
    const host = $(listId);
    if (host && host.tagName === "SELECT") {
      host.value = val || "";
      syncChip(listId);
      return;
    }
    const items = document.querySelectorAll(`#${listId} [data-${attr}]`);
    items.forEach((b, i) => {
      const on = b.dataset[attr] === val;
      b.classList.toggle("active", on);
      // slide the segmented-control thumb to the active cell
      if (on && host && host.classList.contains("row-2")) {
        host.style.setProperty("--mode-i", i);
      }
    });
    syncChip(listId);
  }

function setPanel(id, st) { const p = $(id); if (p) p.dataset.st = st; }

const WORKFLOW_PANELS = {
    debug: {
      prompt: "panel-prompt", config: "panel-config", setup: "panel-setup", control: "panel-control",
      setupMsg: "auto-setup-msg", pause: "b-setup-pause", resume: "b-setup-resume", retry: "b-setup-retry",
    },
    replay: {
      prompt: "replay-panel-episode", config: "replay-panel-config", setup: "replay-panel-config", control: "replay-panel-control",
      setupMsg: "replay-auto-setup-msg", pause: "b-replay-setup-pause", resume: "b-replay-setup-resume", retry: "b-replay-setup-retry",
    },
  };

function activeWorkflowPanels() {
    return S.ACTIVE_TAB === "replay" ? WORKFLOW_PANELS.replay : WORKFLOW_PANELS.debug;
  }

let _prevSetupDone = false;

function retrySetup() { S._setupFired = true; apiPost("/api/setup"); }

function pauseSetup() {
    S._setupPaused = true;
    S._setupFired = true;  // block auto-restart until RESUME
    apiPost("/api/halt");
    renderSetupCtl();
  }

function resumeSetup() {
    S._setupPaused = false;
    S._setupFired = false;  // re-arm so auto-setup fires again
    renderSetupCtl();
  }

function renderSetupCtl() {
    const s = S.STATUS || {};
    const ids = activeWorkflowPanels();
    const inFlight = !!s.setup_stage;
    const done = !!s.is_setup_done;
    const errored = $(ids.setup).dataset.st === "error";
    const pauseBtn = $(ids.pause);
    const resumeBtn = $(ids.resume);
    const retryBtn = $(ids.retry);
    if (pauseBtn) pauseBtn.style.display = (inFlight && !done && !S._setupPaused && !errored) ? "" : "none";
    if (resumeBtn) resumeBtn.style.display = (S._setupPaused && !done) ? "" : "none";
    if (retryBtn) retryBtn.style.display = (errored && !S._setupPaused) ? "" : "none";
  }

const TELEOP_CLIENT_LABELS = { vr_webxr: "VR" };

function collectInputSourceSuffix(status) {
    const teleopCfg = S.CFG && S.CFG.collection && S.CFG.collection.teleop;
    if (!teleopCfg || teleopCfg.control_source !== "client" || !teleopCfg.client_type) return "";
    const label = TELEOP_CLIENT_LABELS[teleopCfg.client_type] || "INPUT";
    const teleop = status && status.teleop;
    const faulted = !!(teleop && (teleop.last_fault || teleop.source_error));
    const state = faulted ? "ERROR" : (teleop && teleop.connected ? "LINKED" : "DOWN");
    const stateClass = state === "LINKED" ? "linked" : (state === "DOWN" ? "down" : "error");
    return ` | <span class="vr-input-status ${stateClass}">${label} ${state}</span>`;
  }

function autoSetup(ready, done, errored) {
    const msg = $(activeWorkflowPanels().setupMsg);
    const stage = (S.STATUS && S.STATUS.setup_stage) ? String(S.STATUS.setup_stage) : "";
    if (_prevSetupDone && !done) S._setupFired = false;
    _prevSetupDone = done;
    if (done) { S._setupFired = true; S._setupPaused = false; renderSetupCtl(); if (msg) msg.textContent = "ROBOT READY"; return; }
    renderSetupCtl();
    if (S._setupPaused) { if (msg) msg.textContent = "PAUSED — CLICK RESUME"; return; }
    if (errored) {
      // Failed: stop spinning, surface the error (full text shows in #err); RETRY button offers a re-run.
      if (msg) msg.textContent = "SETUP FAILED";
      return;
    }
    if (!ready) { S._setupFired = false; if (msg) msg.textContent = "AWAITING CONFIG…"; return; }
    if (stage) { if (msg) msg.textContent = "AUTO · " + stage.toUpperCase(); return; }
    if (!S._setupFired) { S._setupFired = true; apiPost("/api/setup"); }
    // Show the live setup sub-stage (connecting / resetting / warming up…) so the
    // user can see what setup is doing instead of an opaque spinner.
    if (msg) msg.textContent = stage ? ("AUTO · " + stage.toUpperCase()) : "AUTO · PREPARING ROBOT…";
  }

function updateGuide() {
    if (!S.CFG) return;
    const s = S.STATUS || {};
    // The replay status line lives in the banner; only show it on the REPLAY tab.
    const epTask = $("replay-episode-task");
    if (epTask) epTask.style.display = S.ACTIVE_TAB === "replay" ? "" : "none";
    const collectStatus = $("collect-replay-status");
    const collectReplayActive = S.reviewKind === "collect" && LIVE.replayOwner === "collect";
    const showCollectStatus = S.ACTIVE_TAB === "collect" &&
      (S.collectReplayEpisode != null || collectReplayActive);
    if (collectStatus) collectStatus.style.display = showCollectStatus ? "" : "none";
    // The manual connection status also lives in the banner; renderManualConn shows
    // it on the MANUAL tab, so hide it everywhere else.
    const mConn = $("manual-conn");
    if (mConn && S.ACTIVE_TAB !== "manual") mConn.style.display = "none";
    if (S.ACTIVE_TAB === "dashboard") return;
    if (S.ACTIVE_TAB === "rl") {
      const hasTask = !!$("rl-task-list").value;
      const hasPolicy = !!$("rl-policy-list").value;
      const setup = !!s.is_setup_done && !!s.policy_connected;
      const running = s.session_status === "running";
      const intervention = !!s.rollout_intervention_active;
      const rolloutHistory = S.episodeHistory && S.episodeHistory.rollout;
      const saved = Number(s.rollout && s.rollout.completed_episodes || 0) > 0 ||
        !!(rolloutHistory && rolloutHistory.loaded && rolloutHistory.episodes.length);
      setPanel("rl-panel-task", hasTask ? "done" : "active");
      setPanel("rl-panel-models", hasPolicy ? "done" : (hasTask ? "active" : "pending"));
      setPanel("rl-panel-data", "done");
      setPanel("rl-panel-setup", s.last_error && !setup ? "error" : (setup ? "done" : (hasTask && hasPolicy ? "active" : "pending")));
      setPanel("rl-panel-rollout", setup ? (running || intervention ? "active" : "done") : "pending");
      setPanel("rl-panel-saved", saved ? "done" : (setup ? "active" : "pending"));
      setPanel("rl-panel-gripper", setup ? "done" : "pending");
      const bar = $("guidebar");
      if (bar) bar.classList.toggle("done", saved && !running && !intervention);
      const guideStep = !hasTask || !hasPolicy ? 1 : (!setup ? 2 : (running || intervention ? 4 : 3));
      $("gb-step").textContent = `STEP ${guideStep}/4`;
      if (!hasTask || !hasPolicy) {
        $("gb-msg").innerHTML = "Select <b>task and Policy</b>";
        $("gb-hint").textContent = "Critic is optional and adds value telemetry when available";
      } else if (!setup) {
        $("gb-msg").innerHTML = s.setup_stage
          ? `SETUP · <b>${String(s.setup_stage).toUpperCase()}</b>`
          : "SETUP starts <b>automatically</b>";
        $("gb-hint").textContent = s.last_error || "Critic is optional";
      } else if (intervention) {
        $("gb-msg").innerHTML = "HIL intervention is <b>recording</b>";
        $("gb-hint").textContent = "ACCEPT resumes rollout; ABANDON rolls the segment back";
      } else if (running) {
        $("gb-msg").innerHTML = "Rollout is <b>running</b>";
        $("gb-hint").textContent = "The Critic curve and rollout/intervention track update live";
      } else {
        $("gb-msg").innerHTML = saved ? "Saved episode is ready for <b>REPLAY</b>" : "Ready to <b>RUN</b>";
        $("gb-hint").textContent = "Stop, save, then select an episode to replay";
      }
      return;
    }
    if (S.ACTIVE_TAB === "collect") {
      const collect = s.collect || {};
      const enabled = collectEnabled();
      const collecting = !!collect.collecting;
      const hasPrompt = !!collectTaskValue();
      const activeQueue = (collect.queue || []).filter(
        (item) => item.status === "queued" || item.status === "saving"
      );
      let message = "";
      let hint = "";
      const done = enabled && hasPrompt && S.collectArmEnabled &&
        !collecting && activeQueue.length === 0;
      const bar = $("guidebar");
      if (bar) bar.classList.toggle("done", done);
      if ($("gb-step")) $("gb-step").textContent = collectReplayActive ? "QUALITY CHECK" : "COLLECT";
      if (!enabled) {
        message = "Collection disabled in <b>config</b>";
        hint = "Enable collection + logging before recording";
      } else if (!hasPrompt) {
        message = "Select a <b>TASK</b> before recording";
        hint = "Choose a task before recording";
      } else if (!S.collectArmEnabled) {
        message = "Collection motion is <b>locked</b>";
        hint = "Switch ARM on before START RECORD";
      } else if (collecting) {
        message = `Recording — <b>${collect.current_episode_frames || 0}</b> frames`;
        hint = "END/SAVE queues the episode · CANCEL discards it";
      } else if (activeQueue.length) {
        message = `Converting — <b>${activeQueue.length}</b> item(s) in queue`;
        hint = "Select a green item, then press REPLAY";
      } else {
        message = "Ready — click <b>START RECORD</b>";
        hint = "Queue is collapsed by default; expand for details";
      }
      if ($("gb-msg")) $("gb-msg").innerHTML = message + collectInputSourceSuffix(s);
      if ($("gb-hint")) $("gb-hint").textContent = hint;
      return;
    }
    const isReplay = S.ACTIVE_TAB === "replay";
    const isManual = S.ACTIVE_TAB === "manual" || uiMode(s.cli_mode) === "manual";
    const hasPrompt = isReplay
      ? !!(s.replay_loaded || (S.CFG.is_replay && s.current_episode != null))
      : !!s.selected_task;
    // SIM is the default run mode, so replay always has a mode even before the
    // backend echoes the auto-select — otherwise the merged CONFIG&SETUP card stays
    // "pending" and dims its own MODE buttons into an unclickable deadlock. This is a
    // VISUAL default only; setup must wait for the backend to actually hold a mode
    // (hasRealMode), or it fires /api/setup while cli_mode is still "select" and the
    // backend rejects it — leaving the spinner stuck on "AUTO · PREPARING ROBOT…".
    const hasRealMode = !!s.cli_mode && s.cli_mode !== "select";
    const hasMode = hasRealMode
      || (isReplay && !!(S.CFG.modes && S.CFG.modes.includes("sim")));
    // Replay/manual need no inference strategy.
    const hasStrategy = isReplay || isManual || !!s.selected_strategy;
    const hasConfig = hasMode && hasStrategy;
    const setupDone = !!s.is_setup_done;
    const running = s.session_status === "running";

    // Manual mode is a free-drive console: no prompt / strategy / setup. Once the
    // mode is picked the control panel is live immediately.
    if (isManual) {
      const ids = activeWorkflowPanels();
      setPanel(ids.prompt, "done");
      setPanel(ids.config, "done");
      setPanel(ids.setup, "done");
      setPanel(ids.control, "active");
      const bar = $("guidebar");
      if (bar) bar.classList.toggle("done", !!s.transport_connected);
      if ($("gb-step")) $("gb-step").textContent = "DEVICE";
      if ($("gb-msg")) $("gb-msg").textContent = s.transport_connected ? "ROBOT CONNECTED" : "ROBOT DISCONNECTED";
      if ($("gb-hint")) $("gb-hint").textContent = s.collection_teleop_armed || s.manual_publish_active ? "CONTROL ENABLED" : "CONTROL LOCKED";
      return;
    }

    // per-panel state: done / active (current step) / pending (locked)
    const ids = activeWorkflowPanels();
    setPanel(ids.prompt, hasPrompt ? "done" : "active");
    setPanel(ids.config, hasConfig ? "done" : (hasPrompt ? "active" : "pending"));
    // A setup error (server unreachable) shows as a distinct failed state so the
    // spinner stops and RETRY is offered, instead of spinning forever.
    const setupErrored = !setupDone && hasPrompt && hasConfig && !!s.last_error;
    setPanel(ids.setup, setupDone ? "done" : (setupErrored ? "error" : (hasPrompt && hasConfig ? "active" : "pending")));
    setPanel(ids.control, setupDone ? "active" : "pending");

    // SETUP runs itself: once prompt+config are ready, fire it once and reflect
    // progress through the spinner/bar instead of asking the user to click.
    // Self-heal: a freshly loaded replay can have the backend mode still on "select"
    // (the tab_switch soft-reset clears it). Push the default SIM so the mode settles
    // and auto-setup can proceed, instead of sitting on "AWAITING CONFIG…" forever.
    if (isReplay && hasPrompt && !hasRealMode && hasMode && !S._replayModePushed) {
      S._replayModePushed = true;
      apiPost("/api/select_mode", { mode: "sim" });
    }
    if (hasRealMode) S._replayModePushed = false;
    autoSetup(hasPrompt && hasRealMode && hasStrategy, setupDone, setupErrored);

    // top guide bar message for the current step
    const pick = isReplay ? "an <b>EPISODE</b>" : "a <b>TASK</b>";
    const setupStage = s.setup_stage ? String(s.setup_stage) : "";
    let step = 1, msg = `Step 1 — select ${pick}`, hint = "Follow steps 1 → 4 on the left to run";
    let done = false;
    if (!hasPrompt) { step = 1; msg = `Step 1 — select ${pick} on the left`; }
    else if (!hasMode) { step = 2; msg = "Step 2 — pick a <b>MODE</b> under CONFIG"; }
    else if (!hasStrategy) { step = 2; msg = "Step 2 — pick a <b>STRATEGY</b> under CONFIG"; }
    else if (setupErrored) { step = 3; msg = "Step 3 — <b>setup failed</b>"; hint = "Check the error, then click RETRY under SETUP"; }
    else if (!setupDone) {
      // Surface the live setup sub-stage (resetting / validating policy / warming up…)
      // straight in the banner so the operator sees what setup is doing, not an opaque spinner.
      step = 3;
      msg = setupStage
        ? `Step 3 — <b>${setupStage}</b>…`
        : "Step 3 — <b>preparing the robot</b>…";
      hint = "Setting up automatically — this can take a few seconds";
    }
    else if (!running) { step = 4; msg = "Ready — click <b>RUN ▶</b> to start"; hint = "STOP to halt · RESET to home"; done = true; }
    else { step = 4; msg = "Running — click <b>STOP ■</b> to halt"; hint = "Live observation on the right"; done = true; }

    const bar = $("guidebar");
    if (bar) bar.classList.toggle("done", done);
    if ($("gb-step")) $("gb-step").textContent = `STEP ${step}/4`;
    if ($("gb-msg")) $("gb-msg").innerHTML = msg;
    if ($("gb-hint")) $("gb-hint").textContent = hint;
  }

// ===== config =====

let collectTask = null;

let collectSet = null;

let collectTaskIndex = null;

let replayDefaultKeys = {};

let replayActionCandidates = [];

let replayInspectedDir = "";

function renderPromptButtons(listId) {
    const host = $(listId);
    if (!host) return;
    host.innerHTML = "";
    if (host.tagName === "SELECT") {
      setCommandMetadata(host, "web:switch_task:{task}", true);
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.disabled = true;
      placeholder.textContent = "SELECT TASK";
      host.appendChild(placeholder);
      S.CFG.tasks.forEach((p, i) => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.dataset.label = p || "∅ empty";
        opt.textContent = `${String(i + 1).padStart(2, "0")} ${p || "∅ empty"}`;
        host.appendChild(opt);
      });
      host.onchange = () => {
        const task = host.value;
        mark(listId, "prompt", task);
        S.STATUS.selected_task = task;
        apiPost("/api/select_task", { task });
        updateGuide();
      };
      mark(listId, "prompt", S.STATUS && S.STATUS.selected_task);
      return;
    }
    S.CFG.tasks.forEach((p, i) => {
      const b = document.createElement("button");
      b.className = "seg";
      b.dataset.prompt = p;
      setCommandMetadata(b, "web:switch_task:{task}", true);
      b.innerHTML = `<span class="mk">${String(i + 1).padStart(2, "0")}</span><span>${p || "∅ empty"}</span>`;
      b.onclick = () => {
        mark(listId, "prompt", p);
        S.STATUS.selected_task = p;
        apiPost("/api/select_task", { task: p });
        updateGuide();
      };
      host.appendChild(b);
    });
  }

function collectTaskValue() {
    return collectTask || "";
  }

function collectSetValue() {
    return collectSet || "";
  }

function collectTaskIndexValue() {
    return Number.isInteger(collectTaskIndex) ? collectTaskIndex : null;
  }

function collectTaskSets() {
    const sets = Object.entries((S.CFG && S.CFG.collect_tasks) || {}).map(
      ([name, entries], sourceIndex) => {
        const tasks = (Array.isArray(entries) ? entries : [])
          .filter((entry) => Array.isArray(entry) && entry.length === 2)
          .map(([prompt, target]) => ({ prompt: String(prompt), target: Number(target) }));
        return {
          name, tasks, prompts: tasks.map((task) => task.prompt), sourceIndex,
        };
      }
    ).filter((entry) => entry.prompts.length);
    return orderCollectTaskSets(sets);
  }

function collectTaskPlan(prompt) {
    return ((S.SCENE_PLAN && S.SCENE_PLAN.tasks) || []).find(
      (task) => task.prompt_en === prompt
    ) || null;
  }

function collectSetSceneIds(set) {
    const sceneIds = [];
    (set.tasks || []).forEach((task) => {
      const plan = collectTaskPlan(task.prompt);
      (plan ? plan.scene_ids : []).forEach((sceneId) => {
        if (!sceneIds.includes(sceneId)) sceneIds.push(sceneId);
      });
    });
    const encoded = String(set.name || "").match(/(?:^|_)scene[_-]?([0-9]+)/i);
    if (!sceneIds.length && encoded) sceneIds.push(`SC-${encoded[1].padStart(3, "0")}`);
    return sceneIds;
  }

function collectSceneSortKey(sceneId) {
    const value = String(sceneId || "");
    const number = value.match(/(\d+)(?!.*\d)/);
    return [value.replace(/\d+/g, "").toUpperCase(), number ? Number(number[1]) : Number.MAX_SAFE_INTEGER, value];
  }

function compareCollectSceneIds(left, right) {
    const leftKey = collectSceneSortKey(left);
    const rightKey = collectSceneSortKey(right);
    return leftKey[0].localeCompare(rightKey[0]) || leftKey[1] - rightKey[1] ||
      leftKey[2].localeCompare(rightKey[2]);
  }

function collectSceneDistance(left, right) {
    if (!left || !right) return 0;
    const leftKey = collectSceneSortKey(left);
    const rightKey = collectSceneSortKey(right);
    if (leftKey[0] !== rightKey[0]) return 1000000;
    return Math.abs(leftKey[1] - rightKey[1]);
  }

function orderCollectTaskSets(sets) {
    const remaining = sets.map((set) => ({ ...set, sceneIds: collectSetSceneIds(set) }));
    const ordered = [];
    let previousScene = "";
    while (remaining.length) {
      remaining.sort((left, right) => {
        const leftScene = left.sceneIds[0] || "";
        const rightScene = right.sceneIds[0] || "";
        return (previousScene ? collectSceneDistance(previousScene, leftScene) : 0) -
          (previousScene ? collectSceneDistance(previousScene, rightScene) : 0) ||
          compareCollectSceneIds(leftScene, rightScene) || left.sourceIndex - right.sourceIndex;
      });
      const next = remaining.shift();
      ordered.push(next);
      previousScene = next.sceneIds[next.sceneIds.length - 1] || previousScene;
    }
    return ordered;
  }

function applyCollectTaskSelection(prompt, setName, taskIndex, preferredSceneId = "") {
    const sets = collectTaskSets();
    const selectedSet = sets.find((set) => set.name === setName);
    const selectedTask = selectedSet && selectedSet.tasks[taskIndex];
    if (!selectedTask || selectedTask.prompt !== prompt) return false;
    collectTask = prompt;
    collectSet = setName;
    collectTaskIndex = taskIndex;
    S.STATUS.selected_collect_task = prompt;
    S.STATUS.selected_collect_set = setName;
    S.STATUS.selected_collect_task_index = taskIndex;
    const linked = collectTaskPlan(prompt);
    if (linked) {
      const sceneIndex = preferredSceneId ? linked.scene_ids.indexOf(preferredSceneId) : -1;
      S.scenePlanTaskId = linked.task_id;
      S.scenePlanSceneIndex = sceneIndex >= 0 ? sceneIndex : 0;
      S.scenePlanSceneId = linked.scene_ids[S.scenePlanSceneIndex] || "";
      S.scenePlanRoundIndex = 0;
      S.scenePlanTaskPromptKey = prompt;
    }
    return true;
  }

function stepCollectSet(delta) {
    const sets = collectTaskSets();
    const setIndex = sets.findIndex((set) => set.name === collectSet);
    const nextIndex = (setIndex < 0 ? 0 : setIndex) + delta;
    if (nextIndex < 0 || nextIndex >= sets.length) return;
    selectCollectionDataset(sets[nextIndex].name);
}

function syncCollectTaskNavigation() {
    const sets = collectTaskSets();
    const setIndex = sets.findIndex((set) => set.name === collectSet);
    const currentSet = setIndex >= 0 ? sets[setIndex] : null;
    const collecting = !!(S.STATUS.collect && S.STATUS.collect.collecting);
    const locked = collecting || S.collectTaskSelectionPending;
    const setSelect = $("collect-set-list");
    if (setSelect) setSelect.value = currentSet ? currentSet.name : "";
    const controls = {
      "b-collect-prev-set": setIndex <= 0,
      "b-collect-next-set": setIndex < 0 || setIndex >= sets.length - 1,
    };
    Object.entries(controls).forEach(([id, atBoundary]) => {
      const button = $(id);
      if (button) button.disabled = locked || atBoundary;
    });
    if (setSelect) setSelect.disabled = locked || !sets.length;
  }

function renderCollectTaskButtons() {
    const setHost = $("collect-set-list");
    if (!setHost) return;
    setHost.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.disabled = true;
    placeholder.textContent = "SELECT SET";
    setHost.appendChild(placeholder);
    collectTaskSets().forEach((set) => {
      const option = document.createElement("option");
      option.value = set.name;
      option.textContent = set.name;
      setHost.appendChild(option);
    });
    setHost.onchange = () => {
      const selected = collectTaskSets().find((set) => set.name === setHost.value);
      if (selected) selectCollectionDataset(selected.name);
    };
    $("b-collect-prev-set").onclick = () => stepCollectSet(-1);
    $("b-collect-next-set").onclick = () => stepCollectSet(1);
    syncCollectTaskNavigation();
  }

function renderModeButtons(listId) {
    const host = $(listId);
    if (!host) return;
    host.innerHTML = "";
    const MODE_LABEL = { real: "REAL", sim: "SIM", step: "STEP", manual: "MANUAL" };
    // MANUAL is its own top-level tab now; keep it out of run-mode pickers.
    const modes = S.CFG.modes.filter((m) => m !== "manual");
    setCommandMetadata(host, "web:select_mode:{mode}", true);
    host.style.setProperty("--mode-n", modes.length);
    modes.forEach((m) => {
      const b = document.createElement("button");
      b.className = "seg";
      b.dataset.mode = m;
      setCommandMetadata(b, "web:select_mode:{mode}", true);
      b.innerHTML = `<span>${MODE_LABEL[m] || m}</span>`;
      b.onclick = () => {
        mark(listId, "mode", m);
        S.STATUS.cli_mode = m;
        apiPost("/api/select_mode", { mode: m });
        updateGuide();
      };
      host.appendChild(b);
    });
    // default selection: SIM (fall back to the first available mode)
    mark(listId, "mode", modes.includes("sim") ? "sim" : modes[0]);
  }

const GRIP_STATE = {};

let GRIP_FORCE = true;

function gripperStateFromValue(control, value) {
    const current = Number(value);
    const openValue = Number(control.open_value);
    const closeValue = Number(control.close_value);
    if (![current, openValue, closeValue].every(Number.isFinite)) return null;
    return Math.abs(current - openValue) <= Math.abs(current - closeValue) ? "open" : "close";
  }

function syncGripperState(qpos) {
    if (!Array.isArray(qpos)) return;
    const controls = (S.CFG && S.CFG.gripper_controls) || [];
    let changed = false;
    controls.forEach((control) => {
      const state = gripperStateFromValue(control, qpos[control.qpos_index]);
      if (state && GRIP_STATE[control.side] !== state) {
        GRIP_STATE[control.side] = state;
        changed = true;
      }
    });
    if (changed) syncGripperCaps();
  }

function buildGripperCaps(host) {
    if (!host) return;
    host.innerHTML = "";
    const controls = (S.CFG && S.CFG.gripper_controls) || [];
    if (!controls.length) return;
    const caps = document.createElement("div");
    caps.className = "grip-caps";
    if (controls.length === 1) caps.style.gridTemplateColumns = "1fr";
    controls.forEach((control) => {
      if (!GRIP_STATE[control.side]) {
        GRIP_STATE[control.side] = gripperStateFromValue(
          control,
          control.initial_value,
        ) || "close";
      }
      const cap = document.createElement("div");
      cap.className = "grip-cap";
      const lbl = document.createElement("div");
      lbl.className = "grip-cap-lbl";
      lbl.textContent = controls.length > 1 ? control.label : "GRIPPER";
      cap.appendChild(lbl);

      const sw = document.createElement("div");
      sw.className = "grip-sw";
      sw.dataset.grip = control.side;
      setCommandMetadata(sw, "web:gripper:{side}:{state}:{lock}", true);
      const isOpen = GRIP_STATE[control.side] === "open";
      sw.classList.toggle("on", isOpen);
      sw.innerHTML = `<span class="grip-knob">${isOpen ? "open" : "close"}</span>`;
      sw.onclick = () => {
        const state = GRIP_STATE[control.side] === "open" ? "close" : "open";
        GRIP_STATE[control.side] = state;
        apiPost("/api/gripper", { side: control.side, state, lock: GRIP_FORCE });
        syncGripperCaps();
      };
      cap.appendChild(sw);
      caps.appendChild(cap);
    });
    host.appendChild(caps);

    const force = document.createElement("label");
    force.className = "grip-force";
    const txt = document.createElement("span");
    txt.className = "grip-force-txt";
    const title = document.createElement("span");
    title.className = "grip-force-title";
    title.textContent = "force lock";
    const sub = document.createElement("span");
    sub.className = "grip-force-sub";
    sub.textContent = "override policy during run";
    txt.appendChild(title);
    txt.appendChild(sub);
    const fsw = document.createElement("span");
    fsw.className = "ios-sw" + (GRIP_FORCE ? " on" : "");
    force.appendChild(txt);
    force.appendChild(fsw);
    force.onclick = () => { GRIP_FORCE = !GRIP_FORCE; fsw.classList.toggle("on", GRIP_FORCE); };
    host.appendChild(force);
  }

function syncGripperCaps() {
    document.querySelectorAll(".grip-sw").forEach((sw) => {
      const isOpen = GRIP_STATE[sw.dataset.grip] === "open";
      sw.classList.toggle("on", isOpen);
      const knob = sw.querySelector(".grip-knob");
      if (knob) knob.textContent = isOpen ? "open" : "close";
    });
  }

function renderGripperButtons() {
    buildGripperCaps($("gripper-buttons"));
  }

function renderEvalGripper() {
    buildGripperCaps($("eval-gripper-buttons"));
  }

function renderRlGripper(enabled) {
    const host = $("rl-gripper-buttons");
    const state = $("rl-gripper-state");
    if (!host || !state) return;
    if (!enabled) {
      host.innerHTML = "";
      host.dataset.ready = "";
      state.style.display = "";
      return;
    }
    if (host.dataset.ready !== "1") {
      buildGripperCaps(host);
      host.dataset.ready = "1";
    }
    state.style.display = "none";
  }

function renderConfig() {
    $("t-robot").textContent = S.CFG.robot_type || "—";
    renderGripperButtons();
    $("prompt-title").textContent = "TASK";
    $("prompt-n").textContent = S.CFG.tasks.length;
    renderPromptButtons("prompt-list");
    renderCollectTaskButtons();
    renderModeButtons("mode-list");
    renderModeButtons("replay-mode-list");
    // Pick the default run mode on the backend if nothing is chosen yet. Under eval the
    // mode is dictated by the config's cli_mode (e.g. REAL for table_tennis): push that so
    // a directly-entered eval runs on the real arm, instead of the SIM fallback below
    // which (when STATUS is still empty at boot) would silently run trials in SIM — no
    // real-arm motion, instant reset — until a model A->B switch re-selected REAL.
    if (evalEnabled()) {
      const wantMode = evalCfg().cli_mode || "real";
      if ((!S.STATUS || !S.STATUS.cli_mode || S.STATUS.cli_mode === "select") && S.CFG.modes.includes(wantMode)) {
        if (S.STATUS) S.STATUS.cli_mode = wantMode;
        apiPost("/api/select_mode", { mode: wantMode });
      }
    } else if (S.ACTIVE_TAB !== "rl" && (!S.STATUS || !S.STATUS.cli_mode || S.STATUS.cli_mode === "select") && S.CFG.modes.includes("sim")) {
      // SIM is the default run mode for the plain console tabs.
      if (S.STATUS) S.STATUS.cli_mode = "sim";
      apiPost("/api/select_mode", { mode: "sim" });
    }

    $("strategy-h").style.display = "block";
    $("strategy-list").style.display = "block";
    const sl = $("strategy-list"); sl.innerHTML = "";
    setCommandMetadata(sl, "web:select_strategy:{strategy}", true);
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.disabled = true;
    placeholder.textContent = "SELECT STRATEGY";
    sl.appendChild(placeholder);
    S.CFG.strategies.forEach((s, i) => {
      const opt = document.createElement("option");
      opt.value = s.key;
      opt.dataset.label = `${s.key} · ${s.type}`;
      opt.textContent = `${String(i + 1).padStart(2, "0")} ${s.key} · ${s.type}`;
      sl.appendChild(opt);
    });
    sl.onchange = () => {
      const key = sl.value;
      if (!key) return;
      mark("strategy-list", "strategy", key);
      S.STATUS.selected_strategy = key;
      apiPost("/api/select_strategy", { strategy: key });
      renderTune(key);
      updateGuide();
    };
    mark("strategy-list", "strategy", S.STATUS && S.STATUS.selected_strategy);
    $("panel-gripper").style.display = "block";
    $("panel-tune").style.display = "block";
    renderReplayConfig();
    renderTune(S.STATUS && S.STATUS.selected_strategy);
    renderManualTune();
    // MANUAL tab is always reachable now; its CONNECT button arms the live feed.
    // manual_capable just tells the operator whether the launch transport is a real
    // arm (zmq/ros) before they try to connect.
    renderManualConn();
    updateGuide();
  }

function renderReplayConfig() {
    const n = S.STATUS.replay_n_episodes || S.CFG.n_episodes || 0;
    $("replay-episode-n").textContent = n;
    $("replay-qc-box").style.display = S.qcMode ? "flex" : "none";
    const input = $("replay-episode-input");
    if (n > 0) input.max = String(Math.max(0, n - 1));
    const datasetInput = $("replay-dataset-input");
    if (document.activeElement !== datasetInput && !datasetInput.value && S.STATUS.replay_dataset_dir) {
      datasetInput.value = S.STATUS.replay_dataset_dir;
    }

    const inferReplayVideoKeys = (candidates, selected) => {
      const keys = {};
      const cams = S.CFG.camera_keys || [];
      cams.forEach((cam) => {
        const match = (candidates || []).find((c) => c === `observation.images.${cam}` || c.endsWith(`.${cam}`));
        if (match) keys[cam] = match;
      });
      if (!Object.keys(keys).length && selected) {
        keys[cams[0] || "cam_high"] = selected;
      }
      return keys;
    };

    const replayActionKeyForMode = (mode) => {
      if (mode === "eef") {
        const match = replayActionCandidates.find((key) => {
          const normalized = key.toLowerCase().replace(/[./]/g, "_");
          return normalized === "action_eef" || normalized === "actions_eef" ||
            (normalized.startsWith("action") && normalized.includes("eef"));
        });
        if (match) return match;
      }
      return replayDefaultKeys.action || "action";
    };

    const inspectDataset = async (dir) => {
      $("replay-episode-task").textContent = `inspecting ${dir}…`;
      const r = await apiPost(
        "/api/inspect_dataset",
        { dataset_dir: dir },
        { timeoutMs: 0 },
      );
      if (!r.ok) {
        if (S.pendingQcLoad && S.pendingQcLoad.dir === dir) S.pendingQcLoad = null;
        $("replay-episode-task").textContent = `✗ ${r.error || "cannot read dataset"}`;
        return false;
      }
      replayActionCandidates = r.keys.action.candidates || [];
      replayDefaultKeys = {
        action: r.keys.action.default || "action",
        state: r.keys.state.default || "observation.state",
      };
      S.replayVideoKeys = inferReplayVideoKeys(r.keys.image.candidates, r.keys.image.default);
      replayInspectedDir = dir;
      S.STATUS.replay_n_episodes = r.n_episodes;
      $("replay-episode-n").textContent = r.n_episodes;
      return true;
    };

    const confirm = async () => {
      const id = parseInt(input.value, 10);
      if (Number.isNaN(id) || id < 0) {
        $("replay-episode-task").textContent = "✗ invalid episode id";
        return;
      }
      S._setupFired = false;
      S._setupPaused = false;
      const dir = (datasetInput.value || "").trim();
      if (!dir && S.CFG.is_replay) {
        if (n > 0 && id >= n) { $("replay-episode-task").textContent = `✗ episode id must be in 0..${n - 1}`; return; }
        $("replay-episode-task").textContent = `episode ${id} · loading…`;
        apiPost("/api/select_episode", { episode: id });
        updateGuide();
        return;
      }
      if (!dir) { $("replay-episode-task").textContent = "✗ fill the dataset dir first"; return; }
      const hasInspectedDir = replayInspectedDir === dir;
      const total = hasInspectedDir ? (S.STATUS.replay_n_episodes || 0) : 0;
      if (total > 0 && id >= total) {
        $("replay-episode-task").textContent = `✗ episode id must be in 0..${total - 1}`;
        return;
      }
      $("replay-episode-task").textContent = `loading ${dir} · episode ${id}…`;
      S.qcEpisode = id;
      $("replay-qc-note").value = "";
      $("replay-qc-status").textContent = `QC episode ${id}: mark pass / fail`;
      if (S.qcMode) loadAnnotation(dir, id);
      const actionMode = $("replay-key-action-mode").value || "joint";
      const loadKeys = hasInspectedDir
        ? { ...replayDefaultKeys, action: replayActionKeyForMode(actionMode) }
        : {};
      const loadVideoKeys = hasInspectedDir ? S.replayVideoKeys : {};
      S.replaySeriesKey = null;
      S.replayLoadPending = true;
      replayStop();
      try {
        const load = await apiPost("/api/load_replay_dataset",
          {
            dataset_dir: dir,
            episode: id,
            keys: loadKeys,
            video_keys: loadVideoKeys,
            action_mode: actionMode,
          },
          { timeoutMs: 0 },
        );
        if (!load.ok) {
          $("replay-episode-task").textContent = `✗ ${load.error || "cannot load episode"}`;
          return;
        }
        if (load.n_episodes != null) {
          S.STATUS.replay_n_episodes = load.n_episodes;
          $("replay-episode-n").textContent = load.n_episodes;
        }
        await loadMountedReplaySeries(load);
      } finally {
        S.replayLoadPending = false;
      }
      updateGuide();
    };

    // QC deep-link: inspect then auto-load the requested episode in one go.
    if (S.pendingQcLoad && S.pendingQcLoad.dir) {
      const ql = S.pendingQcLoad;
      S.pendingQcLoad = null;
      (async () => {
        datasetInput.value = ql.dir;
        if (!(await inspectDataset(ql.dir))) return;
        const total = S.STATUS.replay_n_episodes || 0;
        input.value = String(Math.min(ql.episode, Math.max(0, total - 1)));
        if (total > 0) confirm();
      })();
    }

    $("replay-b-episode-confirm").onclick = confirm;
    input.onkeydown = (e) => { if (e.key === "Enter") confirm(); };

    // Load the next episode in the dataset; reuses confirm()'s inspect+load path.
    $("replay-b-episode-next").onclick = () => {
      const cur = S.STATUS && typeof S.STATUS.replay_episode_id === "number" ? S.STATUS.replay_episode_id : -1;
      const total = (S.STATUS && S.STATUS.replay_n_episodes) || 0;
      const next = cur + 1;
      if (cur < 0 || (total > 0 && next >= total)) return;
      input.value = String(next);
      confirm();
    };

    const applyReplayTune = async () => {
      const fps = parseInt($("replay-tune-fps").value, 10);
      if (Number.isNaN(fps) || fps < 1) return;
      $("replay-tune-status").textContent = "applying…";
      const r = await apiPost("/api/update_infer_params", {
        replay_fps: fps,
      });
      $("replay-tune-status").textContent = r.ok ? "applied" : "failed: " + (r.error || "unknown");
    };
    $("b-replay-tune-apply").onclick = applyReplayTune;
    $("replay-tune-fps").onkeydown = (e) => { if (e.key === "Enter") applyReplayTune(); };
  }

function renderTune(strategyKey) {
    if (!S.CFG) return;
    $("tune-inference-rate").value = S.CFG.inference_rate != null ? S.CFG.inference_rate : "";
    $("tune-publish-rate").value = S.CFG.publish_rate != null ? S.CFG.publish_rate : "";
    const strat = (S.CFG.strategies || []).find((s) => s.key === strategyKey);
    const box = $("tune-strategy-args"); box.innerHTML = "";
    $("tune-strategy-key").textContent = strat ? strat.key : "—";
    if (!strat) { $("tune-strategy-h").style.display = "none"; return; }
    $("tune-strategy-h").style.display = "block";
    const fields = strat.fields || [];
    const args = strat.args || {};
    fields.forEach(({ key, label, min, step }) => {
      const row = document.createElement("div"); row.className = "tune-row";
      const span = document.createElement("span"); span.textContent = label;
      const inp = document.createElement("input");
      inp.dataset.arg = key;
      if (step === null) { inp.type = "text"; } else { inp.type = "number"; inp.min = min; inp.step = step; }
      inp.value = args[key] != null ? args[key] : "";
      row.appendChild(span); row.appendChild(inp); box.appendChild(row);
    });
  }

function renderManualTune() {
    if (!S.CFG) return;
    const input = $("manual-tune-publish-rate");
    if (input && document.activeElement !== input) {
      input.value = S.CFG.publish_rate != null ? S.CFG.publish_rate : "";
    }
  }

async function applyTune() {
    const key = S.STATUS && S.STATUS.selected_strategy;
    const body = {
      inference_rate: parseFloat($("tune-inference-rate").value),
      publish_rate: parseInt($("tune-publish-rate").value, 10),
    };
    if (key) {
      const args = {};
      $("tune-strategy-args").querySelectorAll("input").forEach((inp) => {
        const v = inp.value.trim();
        if (v === "") return;
        args[inp.dataset.arg] = inp.type === "number" ? Number(v) : v;
      });
      body.strategies = { [key]: args };
    }
    $("tune-status").textContent = "applying…";
    try {
      const r = await apiPost("/api/update_infer_params", body);
      if (r.ok) {
        S.CFG.inference_rate = body.inference_rate;
        S.CFG.publish_rate = body.publish_rate;
      }
      $("tune-status").textContent = r.ok ? "applied" : "failed: " + (r.error || "unknown");
    } catch (e) {
      $("tune-status").textContent = "failed: " + e;
    }
  }

async function applyManualTune() {
    const publishRate = parseInt($("manual-tune-publish-rate").value, 10);
    if (Number.isNaN(publishRate) || publishRate < 1) return;
    $("manual-tune-status").textContent = "applying…";
    try {
      const r = await apiPost("/api/update_infer_params", { publish_rate: publishRate });
      if (r.ok) {
        S.CFG.publish_rate = publishRate;
        const debugInput = $("tune-publish-rate");
        if (debugInput && document.activeElement !== debugInput) debugInput.value = String(publishRate);
      }
      $("manual-tune-status").textContent = r.ok ? "applied" : "failed: " + (r.error || "unknown");
    } catch (e) {
      $("manual-tune-status").textContent = "failed: " + e;
    }
  }

// ===== manual =====

function enterManualSim() {
    S.manualActive = !S.STATUS.collection_teleop_armed;
    if (S.manualActive) apiPost("/api/select_mode", { mode: "manual" });
    renderManualConn();
    renderControl();
  }

function renderControl() {
    const config = S.CFG;
    if (!config) return false;
    renderDeviceSettings();
    const source = config.collection.teleop;
    const teleop = config.device_selection.teleop !== "joint";
    const name = source.client_type === "vr_webxr" ? "VR" : source.client_type || "LEADER";
    $("control-teleop-controls").hidden = !teleop;
    document.querySelectorAll(".control-joints").forEach(panel => { panel.hidden = false; });
    const motionOwnedByTeleop = !!S.STATUS.collection_teleop_armed;
    S.manualActive = !motionOwnedByTeleop;
    renderManualConn();
    if (motionOwnedByTeleop) {
      $("bm-connect").disabled = true;
      $("bm-send").disabled = true;
      $("bm-home").disabled = true;
    }
    buildManualSliders(S.STATUS.manual_qpos);
    document.querySelectorAll("#manual-sliders-m input").forEach(input => {
      input.disabled = motionOwnedByTeleop;
    });
    if (!teleop) return false;
    const status = S.STATUS;
    const armed = !!status.collection_teleop_armed;
    $("control-arm-enable").checked = armed;
    $("control-arm-enable").disabled = !armed && !status.transport_connected;
    $("manual-conn").textContent = `${name} · ${armed ? "ENABLED" : "LOCKED"}`;
    return true;
  }

function renderManualConn() {
    const btn = $("bm-connect");
    const conn = $("manual-conn");
    const send = $("bm-send");
    setCommandMetadata(btn, S.realRequested ? "web:disconnect" : "web:connect");
    setCommandMetadata(send, S.manualDispatching ? "web:halt" : "web:manual_send");
    const capable = !!(S.CFG && S.CFG.manual_capable);
    conn.style.display = "";
    conn.classList.remove("ok", "armed", "err");
    if (!capable) {
      S.manualDispatching = false;
      // No real-robot transport at all: SIM debug still works, REAL is unavailable.
      conn.textContent = "SIM DEBUG · no real robot (transport not zmq/ros)";
      btn.textContent = "CONNECT REAL";
      btn.disabled = true;
      btn.classList.add("primary");
    } else if (S.realRequested && S.realConnected) {
      conn.textContent = "REAL ROBOT · LIVE";
      conn.classList.add("ok");
      btn.textContent = "DISCONNECT";
      btn.disabled = false;
      btn.classList.remove("primary");
    } else if (S.realRequested && !S.realConnected) {
      conn.textContent = "WAITING FOR ROBOT… (no live data)";
      conn.classList.add("armed");
      btn.textContent = "CANCEL";
      btn.disabled = false;
      btn.classList.remove("primary");
    } else {
      S.manualDispatching = false;
      conn.textContent = "SIM DEBUG · press CONNECT REAL to drive hardware";
      btn.textContent = "CONNECT REAL";
      btn.disabled = false;
      btn.classList.add("primary");
    }
    // HOME drives the sim preview — available in SIM debug. SEND TO REAL
    // needs a live hardware link.
    $("bm-home").disabled = !S.manualActive;
    send.textContent = S.manualDispatching ? "STOP ■" : "SEND TO REAL ▶";
    send.classList.toggle("primary", !S.manualDispatching);
    send.classList.toggle("danger", S.manualDispatching);
    send.disabled = S.manualDispatching ? !S.realRequested : !(S.realRequested && S.realConnected);
  }

function syncManualDispatchState(status) {
    const active = !!status.manual_publish_active;
    if (!active) S.manualDispatchStopPending = false;
    if (S.manualDispatchStopPending && active) return;
    if (S.manualDispatching !== active) {
      S.manualDispatching = active;
      renderManualConn();
    }
  }

function manualConnect() {
    S.realRequested = true;
    renderManualConn();
  }

function manualDisconnect() {
    S.realRequested = false;
    S.realConnected = false;
    S.manualDispatching = false;
    S.manualDispatchStopPending = false;
    renderManualConn();
  }

async function manualDispatchToggle() {
    if (S.manualDispatching) {
      S.manualDispatching = false;
      S.manualDispatchStopPending = true;
      renderManualConn();
      return apiPost("/api/halt");
    }
    S.manualDispatching = true;
    S.manualDispatchStopPending = false;
    renderManualConn();
    await flushManualQpos();
    return apiPost("/api/manual_send");
  }

let _manualQpos = [];
let _manualCurrentQpos = [];

let _manualSendTimer = null;

// Sliders are a command input the server echoes back one round-trip late. Suppress
// echo writes for a short window after the operator last touched a slider, so a stale
// (or lagging real-robot) value can't fight the drag — activeElement alone misses
// touch/trackpad drags that never focus the input, and the instant after mouseup.
let _manualEditTs = 0;
const MANUAL_ECHO_SUPPRESS_MS = 400;

async function flushManualQpos() {
    if (_manualSendTimer) {
      clearTimeout(_manualSendTimer);
      _manualSendTimer = null;
    }
    if (!_manualQpos.length) return;
    await apiPost("/api/manual_qpos", { qpos: _manualQpos.slice() });
  }

function sendManualQpos() {
    if (_manualSendTimer) return;
    _manualSendTimer = setTimeout(() => {
      _manualSendTimer = null;
      flushManualQpos();
    }, 50);  // debounce slider drags
  }

function buildManualSliders(qpos) {
    if (S._manualSlidersBuilt || !qpos || !qpos.length) return;
    _manualQpos = qpos.slice();
    const limits = S.CFG.manual_qpos_limits;
    if (!limits || limits.length !== qpos.length) {
      const nLimits = limits ? limits.length : 0;
      throw new Error(`manual_qpos_limits length mismatch: got ${nLimits}, qpos ${qpos.length}`);
    }
    const host = $("manual-sliders-m");
    host.innerHTML = "";
    const joints = S.CFG.manual_joint_groups.flatMap(group =>
      group.joints.map((name, index) => ({name, group: group.name, first: index === 0})));
    qpos.forEach((v, i) => {
      const limit = limits[i];
      const joint = joints[i];
      if (joint.first) {
        const heading = document.createElement("div");
        heading.className = "ms-group";
        heading.textContent = joint.group;
        host.appendChild(heading);
      }
      const row = document.createElement("div"); row.className = "ms-row";
      row.innerHTML =
        `<label class="ms-j" for="ms-joint-${i}"></label>` +
        `<input id="ms-joint-${i}" type="range" min="${limit.min}" max="${limit.max}" step="${limit.step}" value="${v}" data-j="${i}">` +
        `<span class="ms-x"><span id="ms-target-${i}">${v.toFixed(3)}</span>` +
        `<span class="ms-sep">/</span><span class="ms-current" id="ms-current-${i}">—</span></span>`;
      const range = row.querySelector("input");
      row.querySelector("label").textContent = joint.name;
      range.setAttribute("aria-label", `${joint.group}: ${joint.name}`);
      range.oninput = (e) => {
        const idx = i, val = parseFloat(e.target.value);
        _manualQpos[idx] = val;
        _manualEditTs = Date.now();
        $(`ms-target-${idx}`).textContent = val.toFixed(3);
        sendManualQpos();
      };
      host.appendChild(row);
    });
    S._manualSlidersBuilt = true;
  }

function syncManualSliders(qpos) {
    if (!S._manualSlidersBuilt || !qpos) return;
    if (Date.now() - _manualEditTs < MANUAL_ECHO_SUPPRESS_MS) return;
    qpos.forEach((v, i) => {
      const range = document.querySelector(`#manual-sliders-m input[data-j="${i}"]`);
      if (!range || document.activeElement === range) return;
      if (Math.abs(parseFloat(range.value) - v) > 1e-4) {
        range.value = v;
        _manualQpos[i] = v;
        const x = $(`ms-target-${i}`); if (x) x.textContent = v.toFixed(3);
      }
    });
  }

function renderManualCurrent(qpos) {
    if (!S.manualActive || uiMode(S.STATUS.cli_mode) !== "manual" || !qpos) return;
    _manualCurrentQpos = qpos.slice();
    _manualCurrentQpos.forEach((v, i) => {
      const current = $(`ms-current-${i}`);
      if (current) current.textContent = Number(v).toFixed(3);
    });
  }

function renderManualTarget(qpos) {
    if (!S.manualActive || uiMode(S.STATUS.cli_mode) !== "manual") return;
    buildManualSliders(qpos);
    syncManualSliders(qpos);
  }

export {
  applyRunControlStatus, applyStatus, mark, pauseSetup, replayIsLocalMode, resumeSetup,
  retrySetup, setPanel, startRunFromDebug, syncChip, uiMode, updateGuide,
  applyTune, applyManualTune, collectSetValue, collectTaskIndexValue,
  collectTaskValue, renderConfig, applyCollectTaskSelection,
  renderEvalGripper,
  renderRlGripper,
  enterManualSim, manualConnect, manualDisconnect, manualDispatchToggle,
  renderManualConn, renderManualCurrent, renderManualTarget,
  syncGripperState,
};
