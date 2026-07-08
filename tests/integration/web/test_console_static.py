from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "static"


def console_source() -> str:
    """Concatenate the console markup, styles, and all ES modules into one string.

    The console was split from a single index.html into per-feature ES modules
    (js/*.js) plus css/console.css. These tests assert on feature markers that
    now live across those files, so they search the combined source. Files are
    joined in a fixed order; within-file ordering is preserved, which is what the
    relative-order assertions below rely on (their paired markers share a file).
    """
    parts = [(STATIC_DIR / "index.html").read_text()]
    parts += [p.read_text() for p in sorted(STATIC_DIR.glob("js/*.js"))]
    parts += [p.read_text() for p in sorted(STATIC_DIR.glob("css/*.css"))]
    return "\n".join(parts)


def test_scene3d_initialization_errors_are_rendered_in_overlay():
    html = console_source()

    assert "function show3DError" in html
    assert "boot3D();" in html
    assert "init();\n  load();" not in html
    assert "ghosts[arm]" not in html
    assert "ghostMat[arm]" not in html
    assert "baseGroups[arm]" not in html
    assert "proxies[arm]" not in html
    assert "triads[arm]" not in html


def test_live_camera_streams_do_not_block_page_load_or_reload():
    html = console_source()

    assert "function afterWindowLoad(fn)" in html
    assert "afterWindowLoad(() => loop(pollFrame, 200));" in html
    assert 'window.addEventListener("pagehide", closeMediaStreams);' in html
    assert 'window.addEventListener("beforeunload", closeMediaStreams);' in html
    assert 'img.removeAttribute("src");' in html
    assert 'video.removeAttribute("src");' in html


def test_console_post_requests_are_serialized():
    html = console_source()

    assert "let postQueue = Promise.resolve();" in html
    assert "const result = postQueue.catch(() => {}).then(request);" in html
    assert "postQueue = result.then(() => undefined, () => undefined);" in html


def test_telemetry_bar_renders_image_hz_metric():
    html = console_source()

    assert 'id="t-image-hz"' in html
    assert "function formatImageHz(hz)" in html
    assert '$("t-image-hz").textContent = formatImageHz(s.image_min_hz);' in html


def test_manual_target_qpos_renders_from_status_without_frame_qpos():
    html = console_source()

    assert "function renderManualTarget(qpos)" in html
    assert "renderManualTarget(s.manual_qpos);" in html
    assert (
        "renderManualTarget(S.STATUS.manual_qpos || "
        "(S._manualSlidersBuilt ? null : f.qpos));" in html
    )


def test_manual_sliders_use_configured_qpos_limits():
    html = console_source()

    assert "manual_qpos_limits" in html
    assert "const limits = S.CFG.manual_qpos_limits;" in html
    assert 'min="${limit.min}" max="${limit.max}" step="${limit.step}"' in html
    assert 'min="-3.2" max="3.2" step="0.005"' not in html


def test_manual_scene_transforms_keep_per_mesh_easing():
    html = console_source()

    assert "function applyLayer(layer, arms)" in html
    assert "const instant = !!payload.instant;" not in html
    assert "applyLayer(meshes, command);" in html
    assert "applyLayer(ghostMeshes, payload.ghost);" in html
    assert "mesh.position.lerp(u.tPos, alpha);" in html


def test_collect_scene_poll_uses_normal_cadence_during_recording():
    html = console_source()

    assert "const COLLECT_SCENE_POLL_MS = 80;" in html
    assert "function scenePollMinIntervalMs()" in html
    assert "S.STATUS.collect.collecting" in html
    assert "now - lastScenePollAt < minInterval" in html


def test_manual_dispatch_uses_single_send_stop_toggle():
    html = console_source()

    assert '<button class="btn primary full" id="bm-send">SEND TO REAL ▶</button>' in html
    assert 'id="bm-pause"' not in html
    assert '$("bm-send").onclick = manualDispatchToggle;' in html
    assert 'send.textContent = S.manualDispatching ? "STOP ■" : "SEND TO REAL ▶";' in html
    assert "syncManualDispatchState(s);" in html
    assert "const active = !!status.manual_publish_active;" in html
    assert 'return apiPost("/api/manual_send");' in html
    assert 'return apiPost("/api/halt");' in html


def test_manual_tuning_exposes_publish_rate_only():
    html = console_source()

    assert 'id="manual-tune-publish-rate"' in html
    assert 'id="b-manual-tune-apply"' in html
    assert 'id="manual-tune-status"' in html
    assert '$("b-manual-tune-apply").onclick = applyManualTune;' in html
    assert 'apiPost("/api/update_infer_params", { publish_rate: publishRate })' in html
    assert 'id="manual-tune-inference-rate"' not in html


def test_collect_qc_does_not_jump_to_replay_tab():
    html = console_source()

    assert "function reviewEpisode(kind, item)" in html
    assert "function submitEpisodeQc(kind, verdict)" in html
    assert 'apiPost("/api/review_episode"' in html
    assert '"b-goto-qc").onclick = () => submitEpisodeQc("collect", "fail");' in html
    assert '"b-goto-qc").onclick = () => openBatchQc(' not in html


def test_review_episode_uses_selected_dataset_episode_and_video_query():
    html = console_source()

    assert "reviewDatasetDir" in html
    assert "reviewEpisodeId" in html
    assert 'function renderReviewVideos(videoKeys) {\n    const strip = $("cam-strip");' in html
    assert "dataset_dir: reviewDatasetDir," in html
    assert "episode: String(reviewEpisodeId)," in html
    assert 'src="/api/replay_video?${params.toString()}"' in html
    assert 'id="collect-review-videos"' not in html
    assert 'id="rollout-review-videos"' not in html
    assert "collectReplayEpisode" in html
    assert "latestEpisodeIndex" not in html


def test_review_episode_videos_follow_collection_replay_frame():
    html = console_source()

    assert "function syncReviewVideosToFrame(frameIndex)" in html
    assert "syncReviewVideosToFrame(replay.frame_index);" in html
    assert 'reviewKind === "collect" && replay.active' in html


def test_review_episode_waits_for_video_before_starting_motion():
    html = console_source()

    assert "function waitForStageVideosReady(kind)" in html
    assert "function waitForStageVideosPainted(kind)" in html
    assert 'setStageVideoLoading(true, "loading video")' in html
    assert 'await waitForStageVideosReady("review");' in html
    assert "syncReviewVideosToFrame(0);" in html
    assert 'await waitForStageVideosPainted("review");' in html
    assert 'apiPost("/api/review_replay_start"' in html
    assert "playStageVideos();" in html
    assert html.index('await waitForStageVideosPainted("review");') < html.index(
        'apiPost("/api/review_replay_start"'
    )


def test_video_paint_wait_uses_browser_rendered_frame_before_motion():
    html = console_source()

    assert "function waitForVideoPainted(v)" in html
    assert "requestVideoFrameCallback" in html
    assert "requestAnimationFrame" in html
    assert "await waitForStageVideosPainted" in html


def test_replay_waits_for_canplay_instead_of_loadeddata():
    html = console_source()

    start = html.index("function videoReady(v)")
    end = html.index("async function waitForStageVideosReady", start)
    body = html[start:end]
    assert "v.readyState >= 3" in body
    assert '"canplay"' in body
    assert '"loadeddata"' not in body


def test_replay_playback_buffers_until_video_has_future_data():
    html = console_source()

    assert "master.readyState < 3" in html
    assert "master.readyState < 2" not in html


def test_replay_local_play_clock_drives_smooth_playback():
    html = console_source()

    # The whole-episode transforms blob is fetched once and decoded locally.
    assert 'await fetch("/api/replay_transforms")' in html
    assert 'magic !== "EVAXFRM1"' in html
    assert "function replayApplyTransformFrame(frame)" in html
    # Local play clock + scrub play button, no per-frame network in the fast path.
    assert "function replayPlay()" in html
    assert "function replayStop()" in html
    assert 'onclick="replayToggle()"' in html
    assert "Scene3D.applyTransformFrame(" in html
    assert "function buildReplayPlayTimeline(timestamps, nFrames)" in html
    assert "LIVE.playTime = buildReplayPlayTimeline(LIVE.timestamp, LIVE.n);" in html
    assert "function replayFrameAtTime(timeSec)" in html
    assert "function setReplayCursorFrame(frame, syncVideos = false)" in html
    assert "LIVE.cursorFrac = frac;" in html
    assert "function drawReplayCharts(force = false)" in html
    assert "now - _lastReplayChartDraw < 100" in html
    assert "LIVE.replayLoading = true;" in html
    assert 'await waitForStageVideosReady("replay");' in html
    assert "seekReplay(0);" in html
    assert 'await waitForStageVideosPainted("replay");' in html
    assert html.index("seekReplay(0);") < html.index('await waitForStageVideosPainted("replay");')
    assert html.index('await waitForStageVideosPainted("replay");') < html.index(
        "LIVE.replayLoading = false;"
    )
    assert "if (LIVE.replayLoading) return;" in html
    assert "const master = replayMasterVideo();" in html
    assert "targetTime = master.currentTime || 0;" in html
    assert "const framePos = replayFrameAtTime(targetTime);" in html
    assert "setReplayCursorFrame(framePos, false);" in html
    assert "function syncRealReplayVisual(frame)" in html
    assert "function realReplayVisualFrame()" in html
    assert "REAL_REPLAY_MAX_EXTRAPOLATE_S" in html
    assert "setReplayCursorFrame(frame, false);" in html
    assert "syncReplayVideos(frame);" in html


def test_replay_page_uses_qc_style_frame_playback_not_chunks():
    html = console_source()

    assert '<div class="panel-h"><span class="step-badge">2</span>CONFIG &amp; SETUP<span class="sel-chip" id="replay-mode-chip"></span></div>' in html
    assert '<div class="sub-h">MODE</div>\n            <div class="row-2" id="replay-mode-list"></div>' in html
    assert 'id="replay-auto-setup-msg"' in html
    assert '<div class="panel-h"><span class="step-badge">4</span>CONTROL</div>' in html
    assert 'id="b-replay-step"' not in html
    assert 'id="b-replay-commit"' not in html
    assert 'id="b-replay-halt"' not in html
    assert 'id="replay-tune-exec-steps"' not in html
    assert 'if (replayIsLocalMode()) {' in html
    assert 'apiPost("/api/run")' in html
    assert 'apiPost("/api/halt")' in html
    assert 'uiMode(s.cli_mode) === "real"' in html


def test_replay_mode_control_is_rendered_and_synced():
    html = console_source()

    assert 'renderModeButtons("replay-mode-list");' in html
    assert 'mark("replay-mode-list", "mode", modeMark);' in html
    assert '"replay-mode-list": "replay-mode-chip",' in html


def test_replay_setup_controls_are_guarded_for_console_boot():
    html = console_source()

    assert 'id="b-replay-setup-pause"' in html
    assert 'id="b-replay-setup-resume"' in html
    assert 'id="b-replay-setup-retry"' in html
    assert 'const replaySetupRow = document.querySelector("#replay-panel-config .auto-setup-row");' in html
    assert 'if (replaySetupRow) {' in html
    assert 'if ($("b-replay-setup-pause")) $("b-replay-setup-pause").onclick = pauseSetup;' in html
    assert 'if ($("b-replay-setup-resume")) $("b-replay-setup-resume").onclick = resumeSetup;' in html
    assert 'if ($("b-replay-setup-retry")) $("b-replay-setup-retry").onclick = retrySetup;' in html


def test_replay_scene_interpolates_fractional_transform_frames():
    html = console_source()

    assert "function setMeshInterpolatedTarget(mesh, floats, o0, o1, a)" in html
    assert "const frame0 = Math.floor(frame);" in html
    assert "const frame1 = Math.min(frame0 + 1, nFrames - 1);" in html
    assert "u.tPos.copy(_p0).lerp(_p1, a);" in html
    assert "u.tQuat.copy(_q0).slerp(_q1, a);" in html
    assert "setMeshInterpolatedTarget(mesh, floats, o0, base1 + g * 16, a);" in html


def test_result_replay_uses_local_clock_and_default_all_chart_dims():
    html = console_source()

    assert "/api/episode_transforms?episode_index=" in html
    assert "ReplayScene.loadEpisode(epi, model)" in html
    assert "function tpBuildPlayTimeline(timestamps, nFrames)" in html
    assert "function tpFrameAtTime(timeSec)" in html
    assert "function tpPlayFrame()" in html
    assert "requestAnimationFrame(tpPlayFrame)" in html
    assert "TP.raf = null" in html
    assert 'preload="auto"' in html
    assert "master && master.readyState < 3" in html
    assert "drawSeriesChart($(\"tp-achart-cv\"), TP.series.action, TP.playTime, TP.dimsOnA, TP.i);" in html
    assert "drawSeriesChart($(\"tp-chart-cv\"), TP.series.state, TP.playTime, TP.dimsOn, TP.i);" in html
    assert "const dimsOn = {}; for (let d = 0; d < sd; d++) dimsOn[d] = true;" in html
    assert "const dimsOnA = {}; for (let d = 0; d < ad; d++) dimsOnA[d] = true;" in html
    result_body = html[html.index("function tpSetup(") : html.index("function tpVideos()", html.index("function tpSetup("))]
    assert "d < 7" not in result_body


def test_collect_manual_qc_submit_note():
    html = console_source()

    assert 'id="collect-qc-note"' in html
    assert "function reviewNoteFor(kind)" in html
    assert 'note: reviewNoteFor(kind).value || "",' in html
    assert 'note: "",' not in html


def test_qc_notes_use_console_textarea_style():
    html = console_source()

    assert '<textarea class="review-note" id="collect-qc-note"' in html
    assert '<textarea id="replay-qc-note" class="review-note"' in html
    assert ".review-note {" in html


def test_batch_qc_requires_selected_episode():
    html = console_source()

    assert '"b-goto-qc").disabled =' in html
    assert "collectReplayEpisode == null" in html


def test_manual_qc_fail_marks_episode_tile_red():
    html = console_source()

    assert 'item.qc_verdict === "fail"' in html


def test_saved_collect_episode_is_gray_until_manual_qc():
    html = console_source()
    start = html.index("function collectTone(item)")
    body = html[start:html.index("function collectIssueText", start)]

    assert 'if (item.qc_verdict === "pass") return "cq-ok";' in body
    assert 'if (item.qc_verdict === "fail") return "cq-fail";' in body
    assert 'item.quality === "green"' not in body
    assert 'if (item.quality === "red") return "cq-fail";' in body
    assert 'if (savedEpisodeId(item) != null' not in body
    assert body.rstrip().endswith('return "cq-queued";\n  }')
    assert "Green means saved" not in html
    assert "Pass green / fail red" in html


def test_selected_batch_qc_episode_is_highlighted():
    html = console_source()

    assert ".collect-tile.selected" in html
    assert 'if (episode === S.collectReplayEpisode) tile.classList.add("selected");' in html
    assert 'episode === S.collectReplayEpisode ? " selected" : ""' in html


def test_collect_queue_click_switches_active_review_episode():
    html = console_source()
    start = html.index("function selectCollectEpisode(item)")
    body = html[start:html.index("function selectRolloutSaveEpisode", start)]

    assert "const replay = S.STATUS.collection_replay || {};" in body
    assert 'S.reviewKind === "collect" && replay.active' in body
    assert 'reviewEpisode("collect", item);' in body


def test_collect_review_ignores_stale_async_switches():
    html = console_source()
    start = html.index("async function reviewEpisode(kind, item)")
    body = html[start:html.index("async function submitEpisodeQc", start)]

    assert "let reviewRequestId = 0;" in html
    assert "const requestId = ++reviewRequestId;" in body
    assert "requestId !== reviewRequestId" in body


def test_collect_replay_toggle_starts_and_stops_selected_episode():
    html = console_source()

    assert "function selectCollectEpisode(item)" in html
    assert 'id="b-collect-replay-toggle"' in html
    assert 'id="b-collect-replay-start"' not in html
    assert 'id="b-collect-replay-stop"' not in html
    assert "function toggleSelectedCollectReplay()" in html
    assert "function stopCollectReplay()" in html
    assert "function selectCollectEpisodePointer(event, item)" in html
    assert "tile.onpointerdown = (event) => selectCollectEpisodePointer(event, item);" in html
    assert "row.onpointerdown = (event) => selectCollectEpisodePointer(event, item);" in html
    assert "tile.onclick = () => selectCollectEpisode(item);" in html
    assert "row.onclick = () => selectCollectEpisode(item);" in html
    assert '$("b-collect-replay-toggle").onclick = () => toggleSelectedCollectReplay();' in html
    assert "if (replay.playing)" in html
    assert "const replayActive = !!replay.playing;" in html
    assert 'replayToggle.classList.toggle("collect-replay-stop", replayActive);' in html
    assert ".btn.collect-replay-stop { color: var(--danger); }" in html
    assert 'replayToggle.textContent = replayActive ? "STOP ■" : "REPLAY ▶";' in html
    assert "ondblclick" not in html
    assert "double-click to replay" not in html


def test_collect_fourth_stage_is_quality_check():
    html = console_source()

    assert '<div class="panel-h"><span class="step-badge">4</span>QUALITY CHECK</div>' in html
    assert 'const collectReplayActive = S.reviewKind === "collect" || collectReplay.active;' in html
    assert 'collectReplayActive ? "QUALITY CHECK" : "COLLECT"' in html


def test_stopped_run_button_switches_to_continue_icon():
    html = console_source()

    assert "const intervention = !!s.rollout_intervention_active;" in html
    assert "const continueRun = !running && ((s.step_index || 0) > 0 || intervention);" in html
    assert 'run.querySelector(".rec-label").textContent = running' in html
    assert '? "STOP ■"' in html
    assert ': (intervention ? "RESUME ▶" : (continueRun ? "CONTINUE ▶▶" : "RUN ▶"));' in html


def test_rollout_intervention_abandon_controls_are_wired():
    html = console_source()
    server_source = (
        Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "server.py"
    ).read_text()

    assert 'id="b-rollout-intervention-abandon"' in html
    assert 'apiPost("/api/operator_action", { intent: "cancel" })' in html
    assert (
        '"/api/rollout_intervention_abandon": "web:rollout_intervention_abandon"'
        in server_source
    )
    assert '"/api/rollout_stop": "web:rollout_stop"' in server_source
    assert "save_blocked_by_intervention" in html
    assert "accepted_intervention_segments" in html


def test_debug_hil_control_mode_is_wired():
    html = console_source()
    server_source = (
        Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "server.py"
    ).read_text()

    assert 'id="hil-control-mode"' in html
    assert 'apiPost("/api/rollout_intervention_mode"' in html
    assert "hil_control_mode" in html
    assert "hil_control_modes" in html
    assert '"hil_control_mode": r.hil_control_mode' in server_source
    assert '"/api/rollout_intervention_mode"' in server_source


def test_operator_action_api_is_wired():
    server_source = (
        Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "server.py"
    ).read_text()

    assert '"/api/operator_action": ConsoleRequestHandler._post_operator_action' in server_source
    assert 'self._enqueue_ok(f"web:operator_action:{intent}:ui")' in server_source


def test_collect_and_rollout_controls_use_operator_action():
    html = console_source()

    assert 'apiPost("/api/operator_action", { intent: "start" })' in html
    assert 'apiPost("/api/operator_action", { intent: "accept" })' in html
    assert 'apiPost("/api/operator_action", { intent: "cancel" })' in html
    assert (
        '$("b-rollout-save").onclick = () => '
        'apiPost("/api/operator_action", { intent: "accept" });'
    ) in html
    assert (
        '$("b-rollout-intervention-abandon").onclick = () => '
        'apiPost("/api/operator_action", { intent: "cancel" });'
    ) in html


def test_step_control_has_single_dynamic_instruction_line():
    html = console_source()

    assert "SIM ↻ preview · review then REAL ▶ dispatch" not in html
    assert 'ss.textContent = "READY · press SIM ↻ to infer one chunk";' in html


def test_tab_order_places_manual_and_collect_before_replay():
    html = console_source()

    debug = html.index('data-tab="debug"><span class="idx">01</span> DEBUG')
    manual = html.index('data-tab="manual"><span class="idx">02</span> MANUAL')
    collect = html.index('data-tab="collect"><span class="idx">03</span> COLLECT')
    replay = html.index('data-tab="replay"><span class="idx">04</span> REPLAY')

    assert debug < manual < collect < replay


def test_task_and_strategy_controls_are_dropdown_selects():
    html = console_source()

    assert (
        '<select class="choice-select task-select" id="prompt-list" '
        'aria-label="Task"></select>'
        in html
    )
    assert (
        '<select class="choice-select task-select" id="collect-prompt-list" '
        'aria-label="Collection task"></select>'
        in html
    )
    assert (
        '<select class="choice-select task-select" id="eval-prompt-list" '
        'aria-label="Evaluation task"></select>'
        in html
    )
    assert (
        '<select class="choice-select task-select" id="eval-model-list" aria-label="Evaluation model"></select>'
        in html
    )
    assert (
        '<select class="choice-select strategy-select" id="strategy-list" aria-label="Strategy"></select>'
        in html
    )


def test_eval_model_selector_is_dropdown():
    html = console_source()

    assert 'if (list.tagName === "SELECT") {' in html
    assert 'opt.value = String(slot);' in html
    assert 'list.onchange = () => {' in html
    assert 'const slot = Number(list.value);' in html
    assert "if (evalBusy() || EVAL_SWITCHING)" in html
    assert "evalSwitchCkpt(slot);" in html
    assert 'if (list && list.tagName === "SELECT") list.disabled = true;' in html
    assert 'if (modelList && modelList.tagName === "SELECT") modelList.disabled = lockCkpt;' in html
    assert "-webkit-appearance: none" in html
    assert "background-image: url(\"data:image/svg+xml" in html
    assert 'placeholder.textContent = "SELECT TASK";' in html
    assert 'apiPost("/api/select_task", { task });' in html
    assert 'apiPost("/api/select_collect_task", { task });' in html
    assert 'const opt = document.createElement("option");' in html
    assert 'sl.onchange = () => {' in html
    assert 'apiPost("/api/select_strategy", { strategy: key });' in html


def test_collect_start_requires_motion_switch():
    html = console_source()

    switch = 'id="collect-arm-enable"'
    start = 'id="b-collect-toggle"'
    assert switch in html
    assert html.index(switch) < html.index(start)
    assert "collectArmEnabled: false" in html
    assert "|| !S.collectArmEnabled" in html
    assert "function disarmCollectArm()" in html
    assert 'if (tab !== "collect") disarmCollectArm();' in html
    assert "collect_teleop_armed: tab === \"collect\" && S.collectArmEnabled" in html
    assert "collect_teleop_armed: S.ACTIVE_TAB === \"collect\" && S.collectArmEnabled" in html
    assert (
        'apiPost("/api/tab_switch", {' in html
    )
    assert (
        'S.collectArmEnabled = enabled && S.ACTIVE_TAB === "collect";'
        in html
    )


def test_collect_task_selection_refreshes_record_gate_immediately():
    html = console_source()

    assert (
        'S.STATUS.selected_collect_task = task;\n'
        '        apiPost("/api/select_collect_task", { task });\n'
        '        mark("collect-prompt-list", "prompt", task);\n'
        '        renderCollect();'
    ) in html
    assert (
        'S.STATUS.selected_collect_task = p;\n'
        '        apiPost("/api/select_collect_task", { task: p });\n'
        '        mark("collect-prompt-list", "prompt", p);\n'
        '        renderCollect();'
    ) in html


def test_collect_queue_unlocks_after_end_save_click():
    html = console_source()

    assert "collectQueueEnabled: false" in html
    assert "if (live) S.collectQueueEnabled = true;" in html
    assert "S.collectQueueEnabled || episodes.length > 0 || queue.length > 0" in html


def test_eval_task_select_defaults_to_configured_task():
    html = console_source()

    assert "const prompts = evalCfg().tasks || [];" in html
    assert "prompts.some((p) => p.prompt_en === EVAL_PROMPT)" in html
    assert "EVAL_PROMPT = prompts[0].prompt_en;" in html
    assert "opt.value = p.prompt_en;" in html
    assert "eval-prompt-chip" not in html
    assert "eval-model-chip" not in html
    assert "eval-prompt-n" not in html
    assert "eval-model-n" not in html
    assert html.index('host.classList.toggle("has-value", !!label);') < html.index("if (!chip) return;")


def test_eval_unscored_trial_hint_is_english_and_not_shown_after_save_next():
    html = console_source()

    assert "运行该 trial 后才能打分" not in html
    assert "Run this trial before scoring" in html
    assert "（无备注）" not in html
    assert "(no note)" in html
    assert "EVAL_SHOW_UNSCORABLE_HINT && EVAL_PROMPT && !scorable" in html
    assert (
        "EVAL_SHOW_UNSCORABLE_HINT = false;\n"
        "        EVAL_TRIAL = saved < n ? saved + 1 : saved;"
        in html
    )


def test_eval_stop_latches_stopping_until_backend_settles():
    html = console_source()

    assert 'S.evalRunToggleBusy === "stop"' in html
    assert 'if (live) setEvalPending("stopping");' in html
    assert 'if (S.evalRunToggleBusy === "start")' in html
    assert 'apiPost("/api/eval_stop").catch' in html


def test_debug_manual_collect_hide_action_state_charts():
    html = console_source()

    assert '<div class="stage no-series" id="stage">' in html
    assert ".stage.no-series .stage-charts" in html
    assert ".stage.no-series .stage-scrub" in html
    assert 'const showSeries = S.ACTIVE_TAB === "replay";' in html
    assert 'stage.classList.toggle("no-series", !showSeries);' in html
    assert 'if (LIVE.replayMode || S.ACTIVE_TAB !== "replay" || liveSeriesPolling) return;' in html
    assert "loop(pollLiveSeries, 200);" not in html


def test_replay_charts_use_series_dimension_names():
    html = console_source()

    assert "LIVE.actionNames = r.action_names || [];" in html
    assert "LIVE.stateNames = r.state_names || [];" in html
    assert 'const names = tag === "a" ? LIVE.actionNames : LIVE.stateNames;' in html
    assert 'el.textContent = names[d] || ((tag === "a" ? "a" : "q") + d);' in html


def test_replay_chart_dimension_names_are_scrollable():
    html = console_source()

    assert "overflow-x: auto" in html
    assert "scrollbar-width: thin" in html
    assert ".chart-dims::-webkit-scrollbar" in html
    assert "flex: 0 0 auto; font-family: var(--mono)" in html


def test_replay_series_key_tracks_action_mode_and_key():
    html = console_source()

    assert "s.replay_action_mode || \"\"" in html
    assert "s.replay_action_key || \"\"" in html
    assert "S.replaySeriesKey = null;" in html


def test_replay_video_src_uses_loaded_dataset_episode_and_video_key():
    html = console_source()

    assert 'let replayLoadedDatasetDir = "";' in html
    assert "dataset_dir: replayLoadedDatasetDir," in html
    assert "episode: String(replayLoadedEpisodeId)," in html
    assert 'if (videoKey) params.set("video_key", videoKey);' in html
    assert 'src="/api/replay_video?${params.toString()}"' in html
    assert 'src="/api/replay_video?cam=${encodeURIComponent(k)}"' not in html
