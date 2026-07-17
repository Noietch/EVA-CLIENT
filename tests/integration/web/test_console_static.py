from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "static"
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_replay_perf_probe_measures_from_user_click_to_visible_videos():
    source = (REPO_ROOT / "tests" / "manual" / "replay_perf_probe.py").read_text()

    assert "Input.dispatchMouseEvent" in source
    assert "Page.captureScreenshot" in source
    assert "--headless=new" not in source
    assert "ignoredVideos = new WeakSet" in source
    assert "state.start = performance.now();" in source
    assert "el.click();" not in source
    assert "visible" in source
    assert "VIDEOS_VISIBLE_READY_EXPR" in source
    assert "start_probe(cdp)" not in source


def test_replay_perf_probe_repeats_all_user_visible_review_paths():
    source = (REPO_ROOT / "tests" / "manual" / "replay_perf_probe.py").read_text()

    assert "--replay-next-count" in source
    assert "--collect-count" in source
    assert "--debug-count" in source
    assert "--max-visible-ms" in source
    assert "select_clickable_tile" in source
    assert 'human_click_target_and_start_probe(cdp, target, f"collect-tile-{i}")' in source
    assert 'human_click_and_start_probe(cdp, "#b-collect-replay-toggle")' not in source
    assert "summarize_resources" in source
    assert "replay-next-" in source
    assert "collect-" in source
    assert "debug-" in source


def test_replay_and_review_abort_obsolete_video_downloads_before_replacing_strip():
    html = console_source()

    assert "function replaceCamStripContent(html)" in html
    assert 'v.removeAttribute("src");' in html
    assert "try { v.load(); } catch (e) {}" in html
    assert "function mountEpisodeVideos({ datasetDir, episodeId, videoKeys })" in html
    assert "replaceCamStripContent(cams.map((k) => {" in html
    assert "replaceCamStripContent('<div class=\"cam-empty\">awaiting frame…</div>');" in html


def test_replay_and_review_cells_show_real_video_poster_before_video_canplay():
    html = console_source()

    assert "/api/replay_poster?${params.toString()}" in html
    assert 'class="cam cam-poster"' in html
    assert 'class="cam cam-video"' in html
    assert "onload=\"this.closest('.cam-cell').classList.remove('loading')\"" in html
    assert ".cam-cell.video-ready .cam-poster" in html
    assert ".cam-cell.video-ready .cam-video" in html


def test_video_loading_overlay_does_not_cover_loaded_poster():
    html = console_source()
    start = html.index("function setVideosLoading(videos, on, text)")
    body = html[start : html.index("function setStageVideoLoading", start)]

    assert 'const poster = cell.querySelector(".cam-poster");' in body
    assert "const posterReady =" in body
    assert "const videoPainted =" in body
    assert 'cell.classList.toggle("loading", on && !posterReady && !videoPainted);' in body


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


def test_collect_tile_click_starts_review_immediately():
    html = console_source()
    start = html.index("function selectCollectEpisode(item)")
    body = html[start : html.index("function selectRolloutSaveEpisode", start)]

    assert 'reviewEpisode("collect", item);' in body
    assert "switchActiveReview" not in body


def test_collect_review_uses_shared_local_replay_engine():
    html = console_source()

    assert "async function loadReviewPlayback(info, owner)" in html
    assert "function installReplaySeries(series)" in html
    assert "LIVE.replayOwner = owner;" in html
    assert "replayTransformsUrl = `/api/review_transforms?${params.toString()}`;" in html
    assert "await videosReady;" in html
    assert "transformsReady.then" in html
    assert "await Promise.all([videosReady, transformsReady]);" not in html
    assert "seekReplay(0);" in html
    assert "replayPlay();" in html


def test_collect_review_has_explicit_local_return_to_live():
    html = console_source()

    assert 'id="review-return-live"' in html
    assert 'id="b-collect-replay-toggle"' not in html
    assert '$("review-return-live").onclick = returnReviewToLive;' in html
    assert 'apiPost("/api/exit_collect_replay")' not in html


def test_collect_review_controls_remain_available_during_real_collection():
    html = console_source()

    assert 'const hardwareOwnsCursor = LIVE.replayOwner === "replay" && realRun;' in html
    assert "LIVE.replayMode && !LIVE.replayLoading && !hardwareOwnsCursor" in html


def test_return_to_live_clears_selected_history_locally():
    html = console_source()
    start = html.index("function returnReviewToLive()")
    body = html[start : html.index("function reviewEpisode", start)]

    assert "S.collectReplayEpisode = null;" in body
    assert "clearReviewPlayback();" in body
    assert "exitReplayMode();" in body
    assert "refreshCameraStreams();" in body
    assert "apiPost(" not in body


def test_collect_review_is_not_driven_by_collection_replay_status():
    html = console_source()
    start = html.index("function renderCollect()")
    body = html[start : html.index("function dotClass", start)]

    assert "S.STATUS.collection_replay" not in body
    assert 'LIVE.replayOwner === "collect"' in html
    assert "if (strip && !LIVE.replayMode)" in html


def test_review_episode_uses_selected_dataset_episode_and_video_query():
    html = console_source()

    assert "reviewDatasetDir" in html
    assert "reviewEpisodeId" in html
    assert "function mountEpisodeVideos({ datasetDir, episodeId, videoKeys })" in html
    assert "dataset_dir: datasetDir," in html
    assert "episode: String(episodeId)," in html
    assert 'data-src="/api/replay_video?${params.toString()}"' in html
    assert 'id="collect-review-videos"' not in html
    assert 'id="rollout-review-videos"' not in html
    assert "collectReplayEpisode" in html
    assert "latestEpisodeIndex" not in html


def test_review_episode_videos_are_not_seeked_by_status_poll_during_playback():
    html = console_source()

    render_collect_start = html.index("function renderCollect()")
    render_collect_end = html.index("function dotClass", render_collect_start)
    render_collect = html[render_collect_start:render_collect_end]
    render_rollout_start = html.index("function renderRolloutSave()")
    render_rollout_end = html.index("function renderCollect()", render_rollout_start)
    render_rollout = html[render_rollout_start:render_rollout_end]

    assert "function syncReviewVideosToFrame(frameIndex)" not in html
    assert "replay.frame_index" not in render_collect
    assert "replay.frame_index" not in render_rollout
    assert 'LIVE.replayOwner === "collect"' in html


def test_review_episode_starts_after_video_without_waiting_for_transforms():
    html = console_source()

    assert "function waitForStageVideosReady()" in html
    assert "function waitForStageVideosPainted()" in html
    local_start = html.index("async function loadReviewPlayback(info, owner)")
    local_body = html[local_start : html.index("function replayApplyTransformFrame", local_start)]
    assert "const videosReady = waitForStageVideosReady();" in local_body
    assert "const transformsReady = loadReplayTransforms(loadSeq);" in local_body
    assert "await videosReady;" in local_body
    assert "transformsReady.then" in local_body
    assert "await Promise.all([videosReady, transformsReady]);" not in local_body
    assert local_body.index("seekReplay(0);") < local_body.index("replayPlay();")
    assert local_body.index("replayPlay();") < local_body.index(
        "const transformsReady = loadReplayTransforms(loadSeq);"
    )
    assert local_body.index("replayPlay();") < local_body.index("transformsReady.then")
    assert '["collect", "rollout"].includes(LIVE.replayOwner) && REPLAY_XF_LOADING' in html

    rollout_start = html.index("async function reviewRolloutEpisode(item)")
    rollout_body = html[rollout_start : html.index("async function submitEpisodeQc", rollout_start)]
    assert 'apiPost("/api/review_episode"' in rollout_body
    assert 'loadReviewPlayback({ ...r, dataset_dir: reviewDatasetDir }, "rollout")' in rollout_body
    assert 'apiPost("/api/review_replay_start"' not in rollout_body


def test_replay_scrub_marks_rollout_intervention_ranges():
    html = console_source()

    assert "LIVE.intervention = series.intervention || [];" in html
    assert "function interventionTrackGradient()" in html
    assert 'const color = active ? "var(--accent)" : "var(--rule-strong)";' in html
    assert 'range.classList.toggle("has-intervention", !!controlTrack);' in html
    assert 'LIVE.replayOwner = "rollout";' in html
    assert 'loadReviewPlayback({ ...r, dataset_dir: reviewDatasetDir }, "rollout")' in html
    assert "S.rolloutSaveEpisode = null;" in html


def test_browser_diagnostics_trace_api_errors_and_review_nodes():
    html = console_source()

    assert "function clientTrace(event, details = {}, traceId = null)" in html
    assert 'fetch("/api/client_trace"' in html
    assert 'clientTrace("api.post.begin"' in html
    assert 'clientTrace("api.post.end"' in html
    assert 'clientTrace("api.post.error"' in html
    assert 'window.addEventListener("error"' in html
    assert 'window.addEventListener("unhandledrejection"' in html
    assert 'clientTrace("review.collect.select"' in html
    assert 'clientTrace("review.playback.begin"' in html
    assert 'clientTrace("review.transforms.end"' in html
    assert 'clientTrace("review.qc.end"' in html


def test_video_paint_wait_uses_browser_rendered_frame_before_motion():
    html = console_source()
    start = html.index("async function waitForVideoPainted(v)")
    body = html[start : html.index("async function waitForStageVideosReady", start)]

    assert "function waitForVideoPainted(v)" in html
    assert "requestVideoFrameCallback" in html
    assert "requestAnimationFrame" in html
    assert "await waitForStageVideosPainted" in html
    assert "setTimeout(finish, 500)" not in body
    assert "!v.paused && typeof v.requestVideoFrameCallback" in body


def test_replay_waits_for_canplay_instead_of_loadeddata():
    html = console_source()

    start = html.index("function videoReady(v)")
    end = html.index("async function waitForStageVideosReady", start)
    body = html[start:end]
    assert "v.readyState >= 3" in body
    assert '"canplay"' in body
    assert '"loadeddata"' not in body
    assert "setTimeout(finish, 8000)" not in body


def test_replay_playback_buffers_until_video_has_future_data():
    html = console_source()

    assert "master.readyState < 3" in html
    assert "master.readyState < 2" not in html


def test_replay_keeps_slave_cameras_on_the_master_clock():
    html = console_source()
    sync_start = html.index("function syncReplayVideos(frame, master = null, force = false)")
    sync_body = html[sync_start : html.index("let _replayUrdfSeq", sync_start)]
    play_start = html.index("function replayPlay()")
    play_body = html[play_start : html.index("function replayStop()", play_start)]
    seek_start = html.index("function seekReplay(i, syncVideos = true)")
    seek_body = html[seek_start : html.index("function setReplayCursorFrame", seek_start)]

    assert "const tolerance = 0.5 / replayVideoFps;" in sync_body
    assert "replayVideoFps = Math.max(1, Number(s.replay_fps) || REPLAY_DEFAULT_FPS);" in html
    assert "replayVideoFps = Math.max(1, Number(info.fps) || REPLAY_DEFAULT_FPS);" in html
    assert "if (v === master) return;" in sync_body
    assert "force || Math.abs((v.currentTime || 0) - t) > tolerance" in sync_body
    assert "syncReplayVideos(framePos, master);" in play_body
    assert "syncReplayVideos(LIVE.cursor, null, true);" in seek_body
    assert "0.12" not in sync_body


def test_replay_local_play_clock_drives_smooth_playback():
    html = console_source()
    load_start = html.index("async function loadReplaySeries()")
    load_body = html[load_start : html.index("async function loadReplayTransforms", load_start)]
    xf_start = html.index("async function loadReplayTransforms(loadSeq)")
    xf_body = html[xf_start : html.index("function replayApplyTransformFrame", xf_start)]
    set_frame_start = html.index("function replaySetUrdfFrame(frame)")
    set_frame_body = html[set_frame_start : html.index("function exitReplayMode", set_frame_start)]

    # The whole-episode transforms blob is fetched once and decoded locally.
    assert "const resp = await fetch(replayTransformsUrl);" in html
    assert 'magic !== "EVAXFRM1"' in html
    assert "function replayApplyTransformFrame(frame)" in html
    assert "REPLAY_XF_LOADING" in html
    assert "if (REPLAY_XF_LOADING) return;" not in set_frame_body
    assert "let replayUrdfInFlight = false;" in html
    assert "let replayUrdfPendingFrame = null;" in html
    assert "if (replayUrdfInFlight) {" in set_frame_body
    assert "replayUrdfPendingFrame = i;" in set_frame_body
    assert "resetReplayUrdfRequests();" in xf_body
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
    assert load_body.index("mountReplayVideos();") < load_body.index('apiGet("/api/replay_series")')
    assert "seekReplay(0);" in load_body
    assert 'await waitForStageVideosReady("replay");' not in load_body
    assert 'await waitForStageVideosPainted("replay");' not in load_body
    assert "await transformsReady" not in load_body
    assert load_body.index("seekReplay(0);") < load_body.rindex("LIVE.replayLoading = false;")
    assert "loadReplayTransforms(loadSeq)" not in load_body
    assert "transformsReady.then" not in load_body
    assert "if (LIVE.replayLoading) return;" in html
    assert "const master = replayMasterVideo();" in html
    assert "targetTime = master.currentTime || 0;" in html
    assert "const framePos = Math.max(cursor, replayFrameAtTime(targetTime));" in html
    assert "setReplayCursorFrame(framePos, false);" in html
    assert "function syncRealReplayVisual(frame)" in html
    assert "function realReplayVisualFrame()" in html
    assert "REAL_REPLAY_MAX_EXTRAPOLATE_S" in html
    assert "if (reportedFrame > realReplayReportedFrame)" in html
    assert "realReplayAnchorFrame = Math.max(reportedFrame, cursor);" in html
    assert "const frame = Math.max(cursor, replayFrameAtTime(anchorTime + elapsed));" in html
    assert "setReplayCursorFrame(frame, false);" in html
    assert "syncReplayVideos(frame);" in html


def test_replay_load_does_not_prebuffer_video_before_user_playback():
    html = console_source()
    load_start = html.index("async function loadReplaySeries()")
    load_body = html[load_start : html.index("async function loadReplayTransforms", load_start)]
    play_start = html.index("function replayPlay()")
    play_body = html[play_start : html.index("function replayStop", play_start)]

    assert 'await waitForStageVideosReady("replay");' not in load_body
    assert 'await waitForStageVideosPainted("replay");' not in load_body
    assert "playStageVideos();" in play_body


def test_replay_load_does_not_compute_full_transforms_before_user_playback():
    html = console_source()
    load_start = html.index("async function loadReplaySeries()")
    load_body = html[load_start : html.index("async function loadReplayTransforms", load_start)]
    play_start = html.index("function replayPlay()")
    play_body = html[play_start : html.index("function replayStop", play_start)]

    assert "loadReplayTransforms(loadSeq)" not in load_body
    assert "loadReplayTransforms(replayLoadSeq)" in play_body


def test_replay_load_mounts_videos_from_load_response_without_status_poll():
    html = console_source()
    confirm_start = html.index("const confirm = async () => {")
    confirm_body = html[confirm_start : html.index("// QC deep-link", confirm_start)]

    assert 'import { maybeSyncReplayPlayer, loadMountedReplaySeries } from "./replay.js";' in html
    assert "const hasInspectedDir = replayInspectedDir === dir;" in confirm_body
    assert "const loadKeys = hasInspectedDir" in confirm_body
    assert "const loadVideoKeys = hasInspectedDir ? S.replayVideoKeys : {};" in confirm_body
    assert "inspectDataset(dir)" not in confirm_body
    assert 'const load = await apiPost("/api/load_replay_dataset"' in html
    assert "if (!load.ok) {" in html
    assert "await loadMountedReplaySeries(load);" in html
    assert "function loadMountedReplaySeries(info)" in html
    assert 'replayLoadedDatasetDir = info.dataset_dir || "";' in html
    assert "replayLoadedEpisodeId = Number(info.episode || 0);" in html
    assert "replayLoadedVideoKeys = { ...(info.video_keys || {}) };" in html
    assert "return loadReplaySeries();" in html


def test_replay_status_poll_does_not_remount_during_explicit_load():
    html = console_source()
    confirm_start = html.index("const confirm = async () => {")
    confirm_body = html[confirm_start : html.index("// QC deep-link", confirm_start)]
    maybe_start = html.index("function maybeSyncReplayPlayer(s)")
    maybe_body = html[maybe_start : html.index("function loadMountedReplaySeries", maybe_start)]

    assert "replayLoadPending: false" in html
    assert "S.replayLoadPending = true;" in confirm_body
    assert "S.replayLoadPending = false;" in confirm_body
    assert "if (S.replayLoadPending) return;" in maybe_body


def test_replay_tab_skips_live_frame_and_scene_polling_until_videos_load():
    html = console_source()
    frame_start = html.index("async function pollFrame()")
    frame_body = html[frame_start : html.index("let liveSeriesPolling", frame_start)]
    scene_start = html.index("async function pollScene()")
    scene_body = html[scene_start : html.index("function loop", scene_start)]

    assert 'if (S.ACTIVE_TAB === "replay") return;' in frame_body
    assert frame_body.index('if (S.ACTIVE_TAB === "replay") return;') < frame_body.index(
        "framePolling = true;"
    )
    assert 'if (S.ACTIVE_TAB === "replay" || LIVE.replayMode) return;' in scene_body
    assert scene_body.index('if (S.ACTIVE_TAB === "replay" || LIVE.replayMode) return;') < (
        scene_body.index("const now = performance.now();")
    )


def test_replay_page_uses_qc_style_frame_playback_not_chunks():
    html = console_source()

    assert (
        '<div class="panel-h"><span class="step-badge">2</span>CONFIG &amp; SETUP<span class="sel-chip" id="replay-mode-chip"></span></div>'
        in html
    )
    assert (
        '<div class="sub-h">MODE</div>\n            <div class="row-2" id="replay-mode-list"></div>'
        in html
    )
    assert 'id="replay-auto-setup-msg"' in html
    assert '<div class="panel-h"><span class="step-badge">4</span>CONTROL</div>' in html
    assert 'id="b-replay-step"' not in html
    assert 'id="b-replay-commit"' not in html
    assert 'id="b-replay-halt"' not in html
    assert 'id="replay-tune-exec-steps"' not in html
    assert "if (replayIsLocalMode()) {" in html
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
    assert (
        'const replaySetupRow = document.querySelector("#replay-panel-config .auto-setup-row");'
        in html
    )
    assert "if (replaySetupRow) {" in html
    assert 'if ($("b-replay-setup-pause")) $("b-replay-setup-pause").onclick = pauseSetup;' in html
    assert (
        'if ($("b-replay-setup-resume")) $("b-replay-setup-resume").onclick = resumeSetup;' in html
    )
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
    assert (
        'drawSeriesChart($("tp-achart-cv"), TP.series.action, TP.playTime, TP.dimsOnA, TP.i);'
        in html
    )
    assert (
        'drawSeriesChart($("tp-chart-cv"), TP.series.state, TP.playTime, TP.dimsOn, TP.i);' in html
    )
    assert "const dimsOn = {}; for (let d = 0; d < sd; d++) dimsOn[d] = true;" in html
    assert "const dimsOnA = {}; for (let d = 0; d < ad; d++) dimsOnA[d] = true;" in html
    result_body = html[
        html.index("function tpSetup(") : html.index(
            "function tpVideos()", html.index("function tpSetup(")
        )
    ]
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


def test_collection_qc_gates_on_selected_saved_episode_not_global_queue():
    html = console_source()
    start = html.index('const toggle = $("b-collect-toggle")')
    body = html[start : html.index('$("b-collect-note-save").disabled', start)]

    assert "const selectedEpisode = selectedCollectEpisodeItem();" in body
    assert "const selectedEpisodeSaved = savedEpisodeId(selectedEpisode) != null;" in body
    assert '$("b-collect-qc-pass").disabled = !enabled || !selectedEpisodeSaved;' in body
    assert '$("b-goto-qc").disabled = !enabled || !selectedEpisodeSaved;' in body
    assert "collecting || !selectedEpisodeSaved" not in body
    assert "queue.length > 0" not in body

    selected_start = html.index("function selectedCollectEpisodeItem()")
    selected_body = html[selected_start : html.index("function renderCollectTiles", selected_start)]
    assert "if (reviewTask !== collectTaskValue()) return null;" in selected_body
    assert "return items.find((item) => savedEpisodeId(item) === episode) || null;" in selected_body
    assert '{ status: "saved", episode_index: episode }' not in selected_body


def test_collection_qc_uses_collection_endpoint_without_changing_other_qc_paths():
    html = console_source()

    assert "function episodeQcEndpoint(kind)" in html
    assert 'return kind === "collect" ? "/api/collect_qc_mark" : "/api/qc_mark";' in html
    assert html.count("apiPost(episodeQcEndpoint(kind), {") == 2
    assert 'let reviewTask = "";' in html
    assert "reviewTask = collectTaskValue();" in html
    assert "task: reviewTask," in html
    assert 'apiPost("/api/qc_mark", {' in html


def test_failed_collect_qc_remains_visible_without_refreshing_review():
    html = console_source()
    start = html.index("async function submitEpisodeQc(kind, verdict)")
    body = html[start : html.index("async function submitEpisodeNote", start)]

    assert "if (!r.ok) {" in body
    assert 'status.textContent = `✗ ${r.error || "QC failed"}`;' in body
    assert body.index("if (!r.ok) {") < body.index("applyStatus(await apiGet")
    assert "return;" in body


def test_manual_qc_fail_marks_episode_tile_red():
    html = console_source()

    assert 'item.qc_verdict === "fail"' in html


def test_saved_collect_episode_is_gray_until_manual_qc():
    html = console_source()
    start = html.index("function collectTone(item)")
    body = html[start : html.index("function collectIssueText", start)]

    assert 'if (item.qc_verdict === "pass") return "cq-ok";' in body
    assert 'if (item.qc_verdict === "fail") return "cq-fail";' in body
    assert 'item.quality === "green"' not in body
    assert 'if (item.quality === "red") return "cq-fail";' in body
    assert "if (savedEpisodeId(item) != null" not in body
    assert body.rstrip().endswith('return "cq-queued";\n  }')
    assert "Green means saved" not in html
    assert "Pass green / fail red" in html


def test_selected_batch_qc_episode_is_highlighted():
    html = console_source()

    assert ".collect-tile.selected" in html
    assert 'if (episode === S.collectReplayEpisode) tile.classList.add("selected");' in html
    assert 'episode === S.collectReplayEpisode ? " selected" : ""' in html


def test_collect_queue_click_reviews_selected_episode_immediately():
    html = console_source()
    start = html.index("function selectCollectEpisode(item)")
    body = html[start : html.index("function selectRolloutSaveEpisode", start)]

    assert "S.collectReplayEpisode = episode;" in body
    assert 'reviewEpisode("collect", item);' in body
    assert "switchActiveReview" not in body


def test_rollout_saved_episode_list_stays_visible_for_debug_review_in_sim():
    html = console_source()
    start = html.index("function renderRolloutSave()")
    body = html[start : html.index("function renderCollect()", start)]

    assert "const hideRolloutSave =" in body
    assert 'items.length === 0 && S.reviewKind !== "rollout"' in body
    assert 'panel.style.display = hideRolloutSave ? "none" : "";' in body


def test_collect_review_ignores_stale_async_switches():
    html = console_source()
    start = html.index("async function reviewCollectEpisode(item)")
    body = html[start : html.index("async function submitEpisodeQc", start)]

    assert "let reviewRequestId = 0;" in html
    assert "const requestId = ++reviewRequestId;" in body
    assert "requestId !== reviewRequestId" in body
    assert body.index("exitReplayMode();") < body.index('apiPost("/api/review_episode"')


def test_collect_review_starts_from_selection_and_returns_to_live_explicitly():
    html = console_source()

    assert "function selectCollectEpisode(item)" in html
    assert 'id="b-collect-replay-toggle"' not in html
    assert 'id="review-return-live"' in html
    assert "function returnReviewToLive()" in html
    assert "function selectCollectEpisodePointer(event, item)" in html
    assert "tile.onpointerdown = (event) => selectCollectEpisodePointer(event, item);" in html
    assert "row.onpointerdown = (event) => selectCollectEpisodePointer(event, item);" in html
    assert "tile.onclick = () => selectCollectEpisode(item);" in html
    assert "row.onclick = () => selectCollectEpisode(item);" in html
    assert '$("review-return-live").onclick = returnReviewToLive;' in html
    assert "exitReplayMode();" in html
    assert "refreshCameraStreams();" in html
    assert "ondblclick" not in html
    assert "double-click to replay" not in html


def test_collect_fourth_stage_is_quality_check():
    html = console_source()

    assert '<div class="panel-h"><span class="step-badge">4</span>QUALITY CHECK</div>' in html
    assert (
        'const collectReplayActive = S.reviewKind === "collect" && LIVE.replayOwner === "collect";'
        in html
    )
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
        '"/api/rollout_intervention_abandon": "web:rollout_intervention_abandon"' in server_source
    )
    assert '"/api/rollout_stop": "web:rollout_stop"' in server_source
    assert "save_blocked_by_intervention" in html
    assert "accepted_intervention_segments" in html


def test_hil_control_mode_is_config_only_not_wired():
    html = console_source()
    server_source = (
        Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "server.py"
    ).read_text()

    assert 'id="hil-control-mode"' not in html
    assert 'apiPost("/api/rollout_intervention_mode"' not in html
    assert "hil_control_modes" not in html
    assert '"/api/rollout_intervention_mode"' not in server_source


def test_rollout_hil_enable_toggle_is_wired():
    html = console_source()
    server_source = (
        Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "server.py"
    ).read_text()

    assert 'id="hil-intervention-enable"' in html
    assert 'class="collect-arm-gate" title="Enable rollout HIL intervention"' in html
    assert 'apiPost("/api/rollout_intervention_enabled"' in html
    assert "rollout_intervention_enabled" in html
    assert '"rollout_intervention_enabled": r.rollout_intervention_enabled' in server_source
    assert '"hil_supported": hil_status.supported' in server_source
    assert '"hil_error": hil_status.error' in server_source
    assert '"HIL N/A"' in html
    assert "rollout_intervention_config_enabled" not in server_source
    assert '"/api/rollout_intervention_enabled"' in server_source


def test_operator_action_api_is_wired():
    server_source = (
        Path(__file__).resolve().parents[3] / "src" / "core" / "app" / "console" / "server.py"
    ).read_text()

    assert '"/api/operator_action": ConsoleRequestHandler._post_operator_action' in server_source
    assert 'self._enqueue_ok(f"web:operator_action:{intent}:ui")' in server_source


def test_collect_uses_operator_action_and_rollout_save_uses_rollout_save_route():
    html = console_source()

    assert 'apiPost("/api/operator_action", { intent: "start" })' in html
    assert 'apiPost("/api/operator_action", { intent: "accept" })' in html
    assert 'apiPost("/api/operator_action", { intent: "cancel" })' in html
    assert ('$("b-rollout-save").onclick = () => apiPost("/api/rollout_save");') in html
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
        'aria-label="Task"></select>' in html
    )
    assert (
        '<select class="choice-select task-select" id="collect-prompt-list" '
        'aria-label="Collection task"></select>' in html
    )
    assert (
        '<select class="choice-select task-select" id="eval-prompt-list" '
        'aria-label="Evaluation task"></select>' in html
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
    assert "opt.value = String(slot);" in html
    assert "list.onchange = () => {" in html
    assert "const slot = Number(list.value);" in html
    assert "if (evalBusy() || EVAL_SWITCHING)" in html
    assert "evalSwitchCkpt(slot);" in html
    assert 'if (list && list.tagName === "SELECT") list.disabled = true;' in html
    assert 'if (modelList && modelList.tagName === "SELECT") modelList.disabled = lockCkpt;' in html
    assert "-webkit-appearance: none" in html
    assert 'background-image: url("data:image/svg+xml' in html
    assert 'placeholder.textContent = "SELECT TASK";' in html
    assert 'apiPost("/api/select_task", { task });' in html
    assert 'apiPost("/api/select_collect_task", { task });' in html
    assert 'const opt = document.createElement("option");' in html
    assert "sl.onchange = () => {" in html
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
    assert 'collect_teleop_armed: tab === "collect" && S.collectArmEnabled' in html
    assert 'collect_teleop_armed: S.ACTIVE_TAB === "collect" && S.collectArmEnabled' in html
    assert 'apiPost("/api/tab_switch", {' in html
    assert 'S.collectArmEnabled = enabled && S.ACTIVE_TAB === "collect";' in html


def test_collect_task_selection_refreshes_record_gate_immediately():
    html = console_source()

    assert (
        "S.STATUS.selected_collect_task = task;\n"
        '        apiPost("/api/select_collect_task", { task });\n'
        '        mark("collect-prompt-list", "prompt", task);\n'
        "        renderCollect();"
    ) in html
    assert (
        "S.STATUS.selected_collect_task = p;\n"
        '        apiPost("/api/select_collect_task", { task: p });\n'
        '        mark("collect-prompt-list", "prompt", p);\n'
        "        renderCollect();"
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
    assert html.index('host.classList.toggle("has-value", !!label);') < html.index(
        "if (!chip) return;"
    )


def test_eval_unscored_trial_hint_is_english_and_not_shown_after_save_next():
    html = console_source()

    assert "运行该 trial 后才能打分" not in html
    assert "Run this trial before scoring" in html
    assert "（无备注）" not in html
    assert "(no note)" in html
    assert "EVAL_SHOW_UNSCORABLE_HINT && EVAL_PROMPT && !scorable" in html
    assert (
        "EVAL_SHOW_UNSCORABLE_HINT = false;\n"
        "        EVAL_TRIAL = saved < n ? saved + 1 : saved;" in html
    )


def test_eval_stop_latches_stopping_until_backend_settles():
    html = console_source()

    assert 'S.evalRunToggleBusy === "stop"' in html
    assert 'if (live) setEvalPending("stopping");' in html
    assert 'if (S.evalRunToggleBusy === "start")' in html
    assert 'apiPost("/api/eval_stop").catch' in html


def test_live_views_hide_charts_and_collect_review_reveals_them():
    html = console_source()

    assert '<div class="stage no-series" id="stage">' in html
    assert ".stage.no-series .stage-charts" in html
    assert ".stage.no-series .stage-scrub" in html
    assert 'const showSeries = tab === "replay" ||' in html
    assert 'tab === "collect" && LIVE.replayOwner === "collect"' in html
    assert 'stage.classList.toggle("no-series", !showSeries);' in html
    assert 'if (LIVE.replayMode || S.ACTIVE_TAB !== "replay" || liveSeriesPolling) return;' in html
    assert "loop(pollLiveSeries, 200);" not in html


def test_replay_charts_use_series_dimension_names():
    html = console_source()

    assert "LIVE.actionNames = series.action_names || [];" in html
    assert "LIVE.stateNames = series.state_names || [];" in html
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

    assert 's.replay_action_mode || ""' in html
    assert 's.replay_action_key || ""' in html
    assert "S.replaySeriesKey = null;" in html


def test_replay_video_src_uses_loaded_dataset_episode_and_video_key():
    html = console_source()

    assert 'let replayLoadedDatasetDir = "";' in html
    assert "dataset_dir: replayLoadedDatasetDir," in html
    assert "episode: String(replayLoadedEpisodeId)," in html
    assert 'if (videoKey) params.set("video_key", videoKey);' in html
    assert 'data-src="/api/replay_video?${params.toString()}"' in html
    assert 'src="/api/replay_video?cam=${encodeURIComponent(k)}"' not in html


def test_replay_and_review_defer_video_network_until_playback_needs_it():
    html = console_source()
    media_start = html.index("function mountEpisodeVideos({ datasetDir, episodeId, videoKeys })")
    media_body = html[media_start : html.index("function mountReplayVideos", media_start)]
    ready_start = html.index("function waitForVideoReady(v)")
    ready_body = html[ready_start : html.index("function waitForBrowserPaint", ready_start)]

    assert 'src="/api/replay_poster?${params.toString()}"' in media_body
    assert '`src="/api/replay_video?${params.toString()}"' not in media_body
    assert 'data-src="/api/replay_video?${params.toString()}"' in media_body
    assert "function ensureVideoSource(v)" in html
    assert "ensureVideoSource(v);" in ready_body
