// replay.js: replay playback engine + stage-video sync (replay);
// polling loop for frame/scene/camera refresh (poll).
import { $, LIVE, RUN_CONTROLS, S, apiGet } from "./core.js";
import { buildLiveDims, drawLiveCharts, resetLiveSeries, updateScrub } from "./charts.js";
import { pauseStageVideos, playStageVideos, setStageVideoLoading, waitForStageVideosPainted, waitForStageVideosReady, reviewActiveInCurrentTab } from "./collect.js";
import { applyRunControlStatus, renderManualTarget, uiMode } from "./run.js";

// ===== replay =====

let replayLoadedDatasetDir = "";

let replayLoadedEpisodeId = 0;

let replayLoadedVideoKeys = {};

let REPLAY_XF = null;

let REPLAY_XF_PARTS = null;

let REPLAY_XF_GEOMS = null;

let REPLAY_XF_NG = 0;

let _lastReplayChartDraw = 0;

const REPLAY_DEFAULT_FPS = 10;

const REAL_REPLAY_MAX_EXTRAPOLATE_S = 0.5;

let realReplayVizRaf = null;

let realReplayAnchorFrame = 0;

let realReplayAnchorWall = 0;

let realReplayLastVideoSync = 0;

function replayFallbackDt() {
    const fps = Number(S.STATUS && S.STATUS.replay_fps) || REPLAY_DEFAULT_FPS;
    return 1 / Math.max(1, fps);
  }

function buildReplayPlayTimeline(timestamps, nFrames) {
    const n = Math.max(0, Number(nFrames) || timestamps.length);
    if (!n) return [];
    const diffs = [];
    for (let i = 1; i < n; i++) {
      const dt = Number(timestamps[i]) - Number(timestamps[i - 1]);
      if (Number.isFinite(dt) && dt > 1e-6) diffs.push(dt);
    }
    diffs.sort((a, b) => a - b);
    const fallbackDt = diffs.length ? diffs[Math.floor(diffs.length / 2)] : replayFallbackDt();
    const out = new Array(n);
    out[0] = 0;
    for (let i = 1; i < n; i++) {
      const dt = Number(timestamps[i]) - Number(timestamps[i - 1]);
      out[i] = out[i - 1] + (Number.isFinite(dt) && dt > 1e-6 ? dt : fallbackDt);
    }
    return out;
  }

function replayTimeAtFrame(frame) {
    const timeline = LIVE.playTime.length === LIVE.n ? LIVE.playTime : LIVE.timestamp;
    if (!timeline.length) return 0;
    const clamped = Math.max(0, Math.min(Number(frame) || 0, timeline.length - 1));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(i0 + 1, timeline.length - 1);
    const a = clamped - i0;
    const t0 = Number(timeline[i0]) || 0;
    const t1 = Number(timeline[i1]) || t0;
    return t0 + (t1 - t0) * a;
  }

function replayFrameAtTime(timeSec) {
    const timeline = LIVE.playTime.length === LIVE.n ? LIVE.playTime : LIVE.timestamp;
    if (LIVE.n <= 1 || !timeline.length) return 0;
    const t = Math.max(0, Number(timeSec) || 0);
    if (t <= (timeline[0] || 0)) return 0;
    if (t >= (timeline[LIVE.n - 1] || 0)) return LIVE.n - 1;
    let lo = 0, hi = LIVE.n - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if ((timeline[mid] || 0) <= t) lo = mid;
      else hi = mid;
    }
    const t0 = Number(timeline[lo]) || 0;
    const t1 = Number(timeline[lo + 1]) || t0;
    if (t1 <= t0) return lo;
    return lo + (t - t0) / (t1 - t0);
  }

function maybeSyncReplayPlayer(s) {
    const onReplayTab = S.ACTIVE_TAB === "replay" || S.ACTIVE_TAB === "eval";
    const loaded = s.replay_loaded && s.replay_total_frames > 0;
    if (!onReplayTab || !loaded) {
      if (S.replaySeriesKey !== null) { S.replaySeriesKey = null; exitReplayMode(); }
      return;
    }
    const key = [
      s.replay_dataset_dir || "",
      s.replay_episode_id || 0,
      s.replay_action_mode || "",
      s.replay_action_key || "",
    ].join("|");
    if (key !== S.replaySeriesKey) {
      S.replaySeriesKey = key;
      replayLoadedDatasetDir = s.replay_dataset_dir || "";
      replayLoadedEpisodeId = s.replay_episode_id || 0;
      replayLoadedVideoKeys = { ...S.replayVideoKeys };
      loadReplaySeries();
      return;
    }
    if (LIVE.replayMode && s.session_status === "running" && uiMode(s.cli_mode) === "real") {
      if (LIVE.playing) replayStop();
      const frame = s.replay_frame_index;
      if (typeof frame === "number") syncRealReplayVisual(frame);
      return;
    }
    if (realReplayVizRaf !== null) {
      stopRealReplayVisual(s.replay_frame_index);
    }
  }

async function loadReplaySeries() {
    let r;
    try { r = await apiGet("/api/replay_series"); } catch (e) { return false; }
    if (!r || !r.state || !r.state.length) return false;
    LIVE.replayLoading = true;
    LIVE.timestamp = r.timestamp || [];
    LIVE.action = r.action || [];
    LIVE.state = r.state || [];
    LIVE.actionNames = r.action_names || [];
    LIVE.stateNames = r.state_names || [];
    LIVE.n = LIVE.state.length;
    LIVE.playTime = buildReplayPlayTimeline(LIVE.timestamp, LIVE.n);
    LIVE.dimsOnA = {}; LIVE.dimsOnS = {}; LIVE.dimsBuilt = false;
    LIVE.replayMode = true; LIVE.following = false; LIVE.cursor = 0; LIVE.cursorFrac = null;
    buildLiveDims();
    mountReplayVideos();
    const transformsReady = loadReplayTransforms();
    await waitForStageVideosReady("replay");
    await transformsReady;
    seekReplay(0);
    await waitForStageVideosPainted("replay");
    LIVE.replayLoading = false;
    updateScrub();
    syncReplayRunButtons();
    return true;
  }

async function loadReplayTransforms() {
    REPLAY_XF = null; REPLAY_XF_PARTS = null; REPLAY_XF_GEOMS = null; REPLAY_XF_NG = 0;
    try {
      const resp = await fetch("/api/replay_transforms");
      const ctype = resp.headers.get("Content-Type") || "";
      if (!resp.ok || ctype.indexOf("octet-stream") < 0) return false;
      const buf = await resp.arrayBuffer();
      const v = new DataView(buf);
      const magic = Array.from({ length: 8 }, (_, i) => String.fromCharCode(v.getUint8(i))).join("");
      if (magic !== "EVAXFRM1") return false;
      const nFrames = v.getUint32(8, true), nGeoms = v.getUint32(12, true), hdrLen = v.getUint32(16, true);
      const keys = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, hdrLen)));
      // Float32Array(buf, byteOffset) requires byteOffset % 4 === 0; the JSON header is
      // variable-length so 20+hdrLen is usually misaligned and would throw. slice() copies
      // out a fresh 0-offset buffer that is always aligned.
      const floats = new Float32Array(buf.slice(20 + hdrLen));
      if (floats.length !== nFrames * nGeoms * 16) return false;
      const parts = [], geoms = [];
      keys.forEach((key) => {
        const slash = key.indexOf("/");
        parts.push(key.slice(0, slash));
        geoms.push(key.slice(slash + 1));
      });
      REPLAY_XF = floats;
      REPLAY_XF_PARTS = parts;
      REPLAY_XF_GEOMS = geoms;
      REPLAY_XF_NG = nGeoms;
      return true;
    } catch (e) { return false; }
  }

function replayApplyTransformFrame(frame) {
    const sceneReady = window.Scene3D && Scene3D.applyTransformFrame;
    if (!REPLAY_XF || !REPLAY_XF_PARTS || !REPLAY_XF_GEOMS || !sceneReady) return false;
    Scene3D.applyTransformFrame(REPLAY_XF_PARTS, REPLAY_XF_GEOMS, REPLAY_XF, REPLAY_XF_NG, frame);
    return true;
  }

function syncRealReplayVisual(frame) {
    if (!LIVE.n) return;
    realReplayAnchorFrame = Math.max(0, Math.min(Number(frame) || 0, LIVE.n - 1));
    realReplayAnchorWall = performance.now();
    if (realReplayVizRaf === null) {
      syncReplayVideos(realReplayAnchorFrame);
      playStageVideos();
      realReplayLastVideoSync = realReplayAnchorWall;
      realReplayVizRaf = requestAnimationFrame(realReplayVisualFrame);
    }
  }

function realReplayVisualFrame() {
    if (realReplayVizRaf === null) return;
    const elapsed = Math.min(
      Math.max(0, (performance.now() - realReplayAnchorWall) / 1000),
      REAL_REPLAY_MAX_EXTRAPOLATE_S,
    );
    const anchorTime = replayTimeAtFrame(realReplayAnchorFrame);
    const frame = replayFrameAtTime(anchorTime + elapsed);
    setReplayCursorFrame(frame, false);
    const now = performance.now();
    if (now - realReplayLastVideoSync > 250) {
      syncReplayVideos(frame);
      playStageVideos();
      realReplayLastVideoSync = now;
    }
    realReplayVizRaf = requestAnimationFrame(realReplayVisualFrame);
  }

function stopRealReplayVisual(frame) {
    cancelAnimationFrame(realReplayVizRaf);
    realReplayVizRaf = null;
    pauseStageVideos();
    if (typeof frame === "number" && LIVE.n) {
      seekReplay(Math.max(0, Math.min(frame, LIVE.n - 1)));
    }
  }

function mountReplayVideos() {
    const strip = $("cam-strip");
    if (!strip) return;
    const cams = (S.CFG && S.CFG.camera_keys) || [];
    strip.innerHTML = cams.map((k) => {
      const params = new URLSearchParams({
        cam: k,
        dataset_dir: replayLoadedDatasetDir,
        episode: String(replayLoadedEpisodeId),
      });
      const videoKey = replayLoadedVideoKeys[k];
      if (videoKey) params.set("video_key", videoKey);
      return `<div class="cam-cell"><div class="cam-lbl">${k}</div>` +
        `<video class="cam" data-key="${k}" muted playsinline preload="auto" ` +
        `src="/api/replay_video?${params.toString()}" ` +
        `onerror="this.closest('.cam-cell').style.display='none'"></video>` +
        `<div class="cam-loading"><span class="spinner"></span><span>loading video</span></div></div>`;
    }).join("") || '<div class="cam-empty">no camera video</div>';
  }

function replayVideos() {
    const strip = $("cam-strip");
    return strip ? Array.from(strip.querySelectorAll("video.cam")) : [];
  }

function replayMasterVideo() {
    return replayVideos().find((v) => {
      const cell = v.closest(".cam-cell");
      return !v.error && (!cell || cell.style.display !== "none");
    }) || null;
  }

function syncReplayVideos(frame) {
    const t = replayTimeAtFrame(frame);
    replayVideos().forEach((v) => {
      if (isFinite(t) && Math.abs((v.currentTime || 0) - t) > 0.12) {
        try { v.currentTime = t; } catch (e) {}
      }
    });
  }

let _replayUrdfSeq = 0;

async function replaySetUrdfFrame(frame) {
    if (replayApplyTransformFrame(frame)) {
      return;
    }
    const i = Math.max(0, Math.min(Math.round(Number(frame) || 0), LIVE.n - 1));
    const seq = ++_replayUrdfSeq;
    try {
      const r = await apiGet("/api/replay_scene_frame?frame=" + i);
      if (seq !== _replayUrdfSeq) return;
      if (r.available && window.Scene3D) Scene3D.applyTransforms({ arms: r.arms });
    } catch (e) { /* transient */ }
  }

function exitReplayMode() {
    replayStop();
    stopRealReplayVisual();
    LIVE.replayMode = false;
    LIVE.replayLoading = false;
    LIVE.cursorFrac = null;
    replayLoadedDatasetDir = "";
    replayLoadedEpisodeId = 0;
    replayLoadedVideoKeys = {};
    REPLAY_XF = null; REPLAY_XF_NG = 0;
    REPLAY_XF_PARTS = null; REPLAY_XF_GEOMS = null; _lastReplayChartDraw = 0;
    LIVE.playTime = [];
    // Drop the replay <video> elements so the live MJPEG strip rebuilds cleanly.
    const strip = $("cam-strip");
    if (strip) strip.innerHTML = '<div class="cam-empty">awaiting frame…</div>';
    resetLiveSeries();
  }

function seekReplay(i, syncVideos = true) {
    LIVE.cursor = Math.max(0, Math.min(i, LIVE.n - 1));
    LIVE.cursorFrac = null;
    replaySetUrdfFrame(LIVE.cursor);
    if (syncVideos) syncReplayVideos(LIVE.cursor);
    updateScrub();
    drawReplayCharts();
  }

function setReplayCursorFrame(frame, syncVideos = false) {
    if (!LIVE.n) return;
    const frac = Math.max(0, Math.min(Number(frame) || 0, LIVE.n - 1));
    LIVE.cursorFrac = frac;
    LIVE.cursor = Math.max(0, Math.min(Math.floor(frac), LIVE.n - 1));
    replaySetUrdfFrame(frac);
    if (syncVideos) syncReplayVideos(frac);
    updateScrub();
    drawReplayCharts();
  }

function replayToggle() { LIVE.playing ? replayStop() : replayPlay(); }

function syncReplayRunButtons() { if (S.STATUS) applyRunControlStatus(RUN_CONTROLS.replay, S.STATUS); }

function replayPlay() {
    if (!LIVE.replayMode || LIVE.n === 0) return;
    if (LIVE.replayLoading) return;
    if (LIVE.cursor >= LIVE.n - 1) seekReplay(0);
    LIVE.playing = true;
    const btn = $("scrub-play"); if (btn) btn.textContent = "⏸";
    syncReplayRunButtons();
    playStageVideos();
    const playT0 = replayTimeAtFrame(LIVE.cursor);
    const wall0 = performance.now();
    const frame = () => {
      if (!LIVE.playing) return;
      const master = replayMasterVideo();
      if (master && (master.readyState < 3 || (master.paused && !master.ended))) {
        setStageVideoLoading(true, "buffering video");
        LIVE.raf = requestAnimationFrame(frame);
        return;
      }
      let targetTime;
      if (master) {
        setStageVideoLoading(false, "");
        targetTime = master.currentTime || 0;
      } else {
        targetTime = playT0 + (performance.now() - wall0) / 1000;
      }
      const framePos = replayFrameAtTime(targetTime);
      setReplayCursorFrame(framePos, false);
      if (framePos >= LIVE.n - 1) { replayStop(); return; }
      LIVE.raf = requestAnimationFrame(frame);
    };
    LIVE.raf = requestAnimationFrame(frame);
  }

function replayStop() {
    LIVE.playing = false;
    if (LIVE.raf) { cancelAnimationFrame(LIVE.raf); LIVE.raf = null; }
    const btn = $("scrub-play"); if (btn) btn.textContent = "▶";
    syncReplayRunButtons();
    pauseStageVideos();
    setStageVideoLoading(false, "");
    drawReplayCharts(true);
  }

function drawReplayCharts(force = false) {
    const throttled = LIVE.playing || realReplayVizRaf !== null;
    if (force || !throttled) {
      _lastReplayChartDraw = performance.now();
      drawLiveCharts();
      return;
    }
    const now = performance.now();
    if (now - _lastReplayChartDraw < 100) return;
    _lastReplayChartDraw = now;
    drawLiveCharts();
  }

// ===== poll =====

let framePolling = false;

function refreshCameraStreams() {
    // Clear the strip outright (not just re-poke src): the <img> may be mid-stream on
    // the previous tab's source, painting a stale replay frame. Dropping the elements
    // makes that frame vanish immediately; pollFrame rebuilds fresh streams on the next
    // tick, by which time the backend's active_tab has settled to the new tab.
    const strip = $("cam-strip");
    if (!strip) return;
    strip.innerHTML = '<div class="cam-empty">awaiting frame…</div>';
  }

async function pollFrame() {
    if (framePolling) return;
    framePolling = true;
    try {
      const f = await apiGet("/api/frame");
      const strip = $("cam-strip");
      const keys = f.cameras || [];
      // REPLAY owns the cam strip with native <video> elements driven by the scrub
      // clock; never let the live MJPEG rebuild clobber them.
      if (strip && !LIVE.replayMode && !reviewActiveInCurrentTab()) {
        const existing = strip.querySelectorAll("img.cam");
        if (!keys.length) {
          // No live cameras (e.g. SIM DEBUG): drop any leftover replay <img> streams
          // so a previous tab's last frame can't stay frozen on screen.
          if (existing.length) strip.innerHTML = '<div class="cam-empty">awaiting frame…</div>';
        } else {
          const sameSet = existing.length === keys.length &&
            keys.every((k, i) => existing[i].dataset.key === k);
          if (!sameSet) {
            strip.innerHTML = keys.map((k) =>
              `<div class="cam-cell"><div class="cam-lbl">${k}</div><img class="cam" data-key="${k}" src="/api/camera/${encodeURIComponent(k)}"></div>`
            ).join("");
          }
        }
      }
      // manual mode: seed sliders (once) from command qpos. Fall back to live qpos
      // only to seed the very first build; once built, never feed the lagging
      // real-robot position back into the sliders or they snap backward mid-drag.
      if (S.manualActive && uiMode(S.STATUS.cli_mode) === "manual") {
        renderManualTarget(S.STATUS.manual_qpos || (S._manualSlidersBuilt ? null : f.qpos));
      } else if (S._manualSlidersBuilt) {
        S._manualSlidersBuilt = false;
        $("manual-sliders-m").innerHTML = "";
      }
    } catch (e) { /* transient */ }
    framePolling = false;
  }

let liveSeriesPolling = false;

async function pollLiveSeries() {
    // The only visible stage charts are REPLAY charts, and they are driven from the
    // one-shot loaded series + scrub clock instead of the live buffer.
    if (LIVE.replayMode || S.ACTIVE_TAB !== "replay" || liveSeriesPolling) return;
    liveSeriesPolling = true;
    try {
      const r = await apiGet("/api/live_series?since=" + LIVE.n);
      const total = r.n || 0;
      // A shrinking buffer means a new episode started — drop stale frames.
      if (total < LIVE.n) resetLiveSeries();
      if (r.timestamp && r.timestamp.length) {
        for (let i = 0; i < r.timestamp.length; i++) {
          LIVE.timestamp.push(r.timestamp[i]);
          LIVE.action.push(r.action[i]);
          LIVE.state.push(r.state[i]);
        }
        LIVE.n = LIVE.timestamp.length;
        if (!LIVE.dimsBuilt) buildLiveDims();
      }
      if (LIVE.following && LIVE.n) LIVE.cursor = LIVE.n - 1;
      updateScrub();
      drawLiveCharts();
    } catch (e) { /* transient */ }
    liveSeriesPolling = false;
  }

let scenePolling = false;

async function pollScene() {
    if (scenePolling) return;
    // REPLAY drives the URDF per-frame off the scrub clock (replaySetUrdfFrame); don't
    // let the live /api/scene poll fight it for the shared Scene3D canvas.
    if (LIVE.replayMode) return;
    scenePolling = true;
    try {
      // MANUAL streams the live real-robot pose + target ghost only when the real
      // link is up; in SIM debug it falls back to /api/scene (which renders the
      // manual command qpos), so the 3D preview tracks the sliders without hardware.
      const manualLive = S.ACTIVE_TAB === "manual" && S.realRequested && S.realConnected;
      const endpoint = manualLive ? "/api/manual_scene" : "/api/scene";
      const sc = await apiGet(endpoint);
      if (sc.available && window.Scene3D) window.Scene3D.applyTransforms(sc);
    } catch (e) { /* transient */ }
    scenePolling = false;
  }

function loop(fn, delay) {
    const tick = async () => { await fn(); setTimeout(tick, delay); };
    tick();
  }

export {
  exitReplayMode, maybeSyncReplayPlayer, replayPlay, replayStop, replayToggle,
  replayVideos, seekReplay,
  loop, pollFrame, pollScene, refreshCameraStreams,
};
