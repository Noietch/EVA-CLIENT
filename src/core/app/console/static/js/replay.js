// replay.js: replay playback engine + stage-video sync (replay);
// polling loop for frame/scene/camera refresh (poll).
import { $, LIVE, RUN_CONTROLS, S, apiGet, clientTrace, replaceCamStripContent } from "./core.js";
import { buildLiveDims, drawLiveCharts, resetLiveSeries, updateScrub } from "./charts.js";
import { applyRunControlStatus, renderManualTarget, uiMode } from "./run.js";

// ===== replay =====

let replayLoadedDatasetDir = "";

let replayLoadedEpisodeId = 0;

let replayLoadedVideoKeys = {};

let REPLAY_XF = null;

let REPLAY_XF_PARTS = null;

let REPLAY_XF_GEOMS = null;

let REPLAY_XF_NG = 0;

let REPLAY_XF_LOADING = false;

let replayTransformsPromise = null;

let _lastReplayChartDraw = 0;

const REPLAY_DEFAULT_FPS = 10;

let replayVideoFps = REPLAY_DEFAULT_FPS;

const REAL_REPLAY_MAX_EXTRAPOLATE_S = 0.5;

let realReplayVizRaf = null;

let realReplayAnchorFrame = 0;

let realReplayAnchorWall = 0;

let realReplayReportedFrame = 0;

let realReplayLastVideoSync = 0;

let replayLoadSeq = 0;

let replayTransformsUrl = "/api/replay_transforms";

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
    if (LIVE.replayOwner === "collect") return;
    if (S.replayLoadPending) return;
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
      replayVideoFps = Math.max(1, Number(s.replay_fps) || REPLAY_DEFAULT_FPS);
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

function loadMountedReplaySeries(info) {
    S.replaySeriesKey = [
      info.dataset_dir || "",
      Number(info.episode || 0),
      info.action_mode || "",
      info.action_key || "",
    ].join("|");
    replayLoadedDatasetDir = info.dataset_dir || "";
    replayLoadedEpisodeId = Number(info.episode || 0);
    replayLoadedVideoKeys = { ...(info.video_keys || {}) };
    replayVideoFps = Math.max(1, Number(info.fps) || REPLAY_DEFAULT_FPS);
    return loadReplaySeries();
  }

async function loadReplaySeries() {
    const loadSeq = ++replayLoadSeq;
    LIVE.replayOwner = "replay";
    LIVE.replayError = "";
    replayTransformsUrl = "/api/replay_transforms";
    LIVE.replayLoading = true;
    LIVE.replayMode = true; LIVE.following = false; LIVE.cursor = 0; LIVE.cursorFrac = null;
    mountReplayVideos();
    let r;
    try { r = await apiGet("/api/replay_series"); } catch (e) {
      if (loadSeq === replayLoadSeq) LIVE.replayLoading = false;
      return false;
    }
    if (loadSeq !== replayLoadSeq) return false;
    if (!r || !r.state || !r.state.length) {
      LIVE.replayLoading = false;
      return false;
    }
    installReplaySeries(r);
    REPLAY_XF = null; REPLAY_XF_PARTS = null; REPLAY_XF_GEOMS = null; REPLAY_XF_NG = 0;
    REPLAY_XF_LOADING = false;
    resetReplayUrdfRequests();
    seekReplay(0);
    if (loadSeq !== replayLoadSeq) return false;
    LIVE.replayLoading = false;
    updateScrub();
    syncReplayRunButtons();
    return true;
  }

function installReplaySeries(series) {
    const seriesFps = Number(series.fps);
    if (seriesFps > 0) replayVideoFps = seriesFps;
    LIVE.timestamp = series.timestamp || [];
    LIVE.action = series.action || [];
    LIVE.state = series.state || [];
    LIVE.actionNames = series.action_names || [];
    LIVE.stateNames = series.state_names || [];
    LIVE.controlSource = series.control_source || [];
    LIVE.intervention = series.intervention || [];
    LIVE.interventionSegmentIndex = series.intervention_segment_index || [];
    LIVE.n = LIVE.state.length;
    LIVE.playTime = buildReplayPlayTimeline(LIVE.timestamp, LIVE.n);
    LIVE.dimsOnA = {};
    LIVE.dimsOnS = {};
    LIVE.dimsBuilt = false;
    LIVE.replayMode = true;
    LIVE.following = false;
    LIVE.cursor = 0;
    LIVE.cursorFrac = null;
    buildLiveDims();
  }

async function loadReplayTransforms(loadSeq) {
    if (REPLAY_XF_LOADING && replayTransformsPromise) return replayTransformsPromise;
    const promise = loadReplayTransformsOperation(loadSeq);
    replayTransformsPromise = promise;
    try {
      return await promise;
    } finally {
      if (replayTransformsPromise === promise) replayTransformsPromise = null;
    }
  }

async function loadReplayTransformsOperation(loadSeq) {
    REPLAY_XF = null; REPLAY_XF_PARTS = null; REPLAY_XF_GEOMS = null; REPLAY_XF_NG = 0;
    REPLAY_XF_LOADING = true;
    clientTrace("review.transforms.begin", {
      url: replayTransformsUrl,
      episode: replayLoadedEpisodeId,
      load_seq: loadSeq,
    });
    try {
      const resp = await fetch(replayTransformsUrl);
      const ctype = resp.headers.get("Content-Type") || "";
      if (!resp.ok || ctype.indexOf("octet-stream") < 0) {
        clientTrace("review.transforms.end", {
          ok: false, status: resp.status, content_type: ctype, load_seq: loadSeq,
        });
        return false;
      }
      const buf = await resp.arrayBuffer();
      const v = new DataView(buf);
      const magic = Array.from({ length: 8 }, (_, i) => String.fromCharCode(v.getUint8(i))).join("");
      if (magic !== "EVAXFRM1") {
        clientTrace("review.transforms.end", { ok: false, reason: "bad_magic", magic, load_seq: loadSeq });
        return false;
      }
      const nFrames = v.getUint32(8, true), nGeoms = v.getUint32(12, true), hdrLen = v.getUint32(16, true);
      const keys = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, hdrLen)));
      // Float32Array(buf, byteOffset) requires byteOffset % 4 === 0; the JSON header is
      // variable-length so 20+hdrLen is usually misaligned and would throw. slice() copies
      // out a fresh 0-offset buffer that is always aligned.
      const floats = new Float32Array(buf.slice(20 + hdrLen));
      if (floats.length !== nFrames * nGeoms * 16) {
        clientTrace("review.transforms.end", {
          ok: false, reason: "bad_length", floats: floats.length, n_frames: nFrames,
          n_geoms: nGeoms, load_seq: loadSeq,
        });
        return false;
      }
      const parts = [], geoms = [];
      keys.forEach((key) => {
        const slash = key.indexOf("/");
        parts.push(key.slice(0, slash));
        geoms.push(key.slice(slash + 1));
      });
      if (loadSeq !== replayLoadSeq) return false;
      REPLAY_XF = floats;
      REPLAY_XF_PARTS = parts;
      REPLAY_XF_GEOMS = geoms;
      REPLAY_XF_NG = nGeoms;
      resetReplayUrdfRequests();
      clientTrace("review.transforms.end", {
        ok: true, n_frames: nFrames, n_geoms: nGeoms, bytes: buf.byteLength, load_seq: loadSeq,
      });
      return true;
    } catch (e) {
      clientTrace("review.transforms.error", { message: String(e), load_seq: loadSeq });
      return false;
    }
    finally {
      if (loadSeq === replayLoadSeq) REPLAY_XF_LOADING = false;
    }
  }

async function loadReviewPlayback(info, owner) {
    const loadSeq = ++replayLoadSeq;
    clientTrace("review.playback.begin", {
      episode: Number(info.episode || 0),
      dataset_dir: info.dataset_dir || "",
      frames: Number(info.frames || 0),
      fps: Number(info.fps || 0),
      load_seq: loadSeq,
    });
    LIVE.replayOwner = owner;
    LIVE.replayError = "";
    LIVE.replayMode = true;
    LIVE.replayLoading = true;
    LIVE.playing = false;
    replayLoadedDatasetDir = info.dataset_dir || "";
    replayLoadedEpisodeId = Number(info.episode || 0);
    replayLoadedVideoKeys = { ...(info.video_keys || {}) };
    installReplaySeries(info);
    REPLAY_XF = null;
    REPLAY_XF_PARTS = null;
    REPLAY_XF_GEOMS = null;
    REPLAY_XF_NG = 0;
    REPLAY_XF_LOADING = false;
    replayTransformsPromise = null;
    resetReplayUrdfRequests();
    mountEpisodeVideos({
      datasetDir: replayLoadedDatasetDir,
      episodeId: replayLoadedEpisodeId,
      videoKeys: replayLoadedVideoKeys,
    });
    const params = new URLSearchParams({
      dataset_dir: replayLoadedDatasetDir,
      episode: String(replayLoadedEpisodeId),
    });
    replayTransformsUrl = `/api/review_transforms?${params.toString()}`;
    const videosReady = waitForStageVideosReady();
    await videosReady;
    if (loadSeq !== replayLoadSeq) {
      clientTrace("review.playback.stale", { load_seq: loadSeq, current_seq: replayLoadSeq });
      return false;
    }
    seekReplay(0);
    await waitForStageVideosPainted();
    if (loadSeq !== replayLoadSeq) return false;
    LIVE.replayLoading = false;
    updateScrub();
    replayPlay();
    clientTrace("review.playback.end", {
      ok: true, episode: replayLoadedEpisodeId, frames: LIVE.n, load_seq: loadSeq,
    });
    const transformsReady = loadReplayTransforms(loadSeq);
    transformsReady.then((transformsOk) => {
      if (loadSeq !== replayLoadSeq) return;
      clientTrace("review.media.ready", {
        episode: replayLoadedEpisodeId,
        videos: replayVideos().length,
        video_errors: replayVideos().filter((v) => !!v.error).length,
        transforms_ok: !!transformsOk,
        load_seq: loadSeq,
      });
      if (!transformsOk) {
        LIVE.replayError = "historical 3D transforms unavailable";
        updateScrub();
        return;
      }
      replaySetUrdfFrame(LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor);
    });
    return true;
  }

function replayApplyTransformFrame(frame) {
    const sceneReady = window.Scene3D && Scene3D.applyTransformFrame;
    if (!REPLAY_XF || !REPLAY_XF_PARTS || !REPLAY_XF_GEOMS || !sceneReady) return false;
    Scene3D.applyTransformFrame(REPLAY_XF_PARTS, REPLAY_XF_GEOMS, REPLAY_XF, REPLAY_XF_NG, frame);
    return true;
  }

function syncRealReplayVisual(frame) {
    if (!LIVE.n) return;
    const reportedFrame = Math.max(0, Math.min(Number(frame) || 0, LIVE.n - 1));
    const now = performance.now();
    if (realReplayVizRaf === null) {
      realReplayReportedFrame = reportedFrame;
      realReplayAnchorFrame = reportedFrame;
      realReplayAnchorWall = now;
      syncReplayVideos(realReplayAnchorFrame);
      playStageVideos();
      realReplayLastVideoSync = realReplayAnchorWall;
      realReplayVizRaf = requestAnimationFrame(realReplayVisualFrame);
      return;
    }
    if (reportedFrame > realReplayReportedFrame) {
      const cursor = LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor;
      realReplayReportedFrame = reportedFrame;
      realReplayAnchorFrame = Math.max(reportedFrame, cursor);
      realReplayAnchorWall = now;
    }
  }

function realReplayVisualFrame() {
    if (realReplayVizRaf === null) return;
    const elapsed = Math.min(
      Math.max(0, (performance.now() - realReplayAnchorWall) / 1000),
      REAL_REPLAY_MAX_EXTRAPOLATE_S,
    );
    const anchorTime = replayTimeAtFrame(realReplayAnchorFrame);
    const cursor = LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor;
    const frame = Math.max(cursor, replayFrameAtTime(anchorTime + elapsed));
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

function mountEpisodeVideos({ datasetDir, episodeId, videoKeys }) {
    const cams = (S.CFG && S.CFG.camera_keys) || [];
    replaceCamStripContent(cams.map((k) => {
      const params = new URLSearchParams({
        cam: k,
        dataset_dir: datasetDir,
        episode: String(episodeId),
      });
      const videoKey = videoKeys[k];
      if (videoKey) params.set("video_key", videoKey);
      return `<div class="cam-cell loading"><div class="cam-lbl">${k}</div>` +
        `<img class="cam cam-poster" data-key="${k}" ` +
        `src="/api/replay_poster?${params.toString()}" ` +
        `onload="this.closest('.cam-cell').classList.remove('loading')" ` +
        `onerror="this.closest('.cam-cell').classList.add('failed')">` +
        `<video class="cam cam-video" data-key="${k}" muted playsinline preload="none" ` +
        `data-poster="/api/replay_poster?${params.toString()}" ` +
        `data-src="/api/replay_video?${params.toString()}" ` +
        `oncanplay="this.closest('.cam-cell').classList.add('video-ready')" ` +
        `onerror="this.closest('.cam-cell').classList.add('failed')"></video>` +
        `<div class="cam-loading"><span class="spinner"></span><span>loading video</span></div>` +
        `<div class="cam-error">recorded camera unavailable</div></div>`;
    }).join("") || '<div class="cam-empty">no camera video</div>');
    clientTrace("review.videos.mount", {
      dataset_dir: datasetDir,
      episode: episodeId,
      cameras: cams,
      video_keys: videoKeys,
    });
    const strip = $("cam-strip");
    if (strip) {
      strip.querySelectorAll("video.cam").forEach((video) => {
        video.addEventListener("error", () => clientTrace("review.video.error", {
          episode: episodeId,
          camera: video.dataset.key || "",
          code: video.error ? video.error.code : 0,
          src: video.dataset.src || "",
        }));
      });
      strip.querySelectorAll("img.cam-poster").forEach((poster) => {
        poster.addEventListener("error", () => clientTrace("review.poster.error", {
          episode: episodeId,
          camera: poster.dataset.key || "",
          src: poster.getAttribute("src") || "",
        }));
      });
    }
  }

function mountReplayVideos() {
    mountEpisodeVideos({
      datasetDir: replayLoadedDatasetDir,
      episodeId: replayLoadedEpisodeId,
      videoKeys: replayLoadedVideoKeys,
    });
  }

function replayVideos() {
    const strip = $("cam-strip");
    return strip ? Array.from(strip.querySelectorAll("video.cam")) : [];
  }

function setVideosLoading(videos, on, text) {
    videos.forEach((v) => {
      const cell = v.closest(".cam-cell");
      if (!cell) return;
      const poster = cell.querySelector(".cam-poster");
      const posterReady = poster && poster.complete && poster.naturalWidth > 0;
      const videoPainted = v.readyState >= 2 && cell.classList.contains("video-ready");
      cell.classList.toggle("loading", on && !posterReady && !videoPainted);
      const label = cell.querySelector(".cam-loading span:last-child");
      if (label && text) label.textContent = text;
    });
  }

function setStageVideoLoading(on, text) {
    setVideosLoading(replayVideos(), on, text);
  }

function videoReady(v) {
    return v.error || v.readyState >= 3;
  }

function ensureVideoSource(v) {
    const src = v.dataset.src || "";
    if (!src || v.getAttribute("src") === src) return;
    const poster = v.dataset.poster || "";
    if (poster) v.setAttribute("poster", poster);
    v.setAttribute("src", src);
  }

function waitForVideoReady(v) {
    if (videoReady(v)) return Promise.resolve();
    return new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        v.removeEventListener("canplay", finish);
        v.removeEventListener("canplaythrough", finish);
        v.removeEventListener("error", finish);
        resolve();
      };
      v.addEventListener("canplay", finish);
      v.addEventListener("canplaythrough", finish);
      v.addEventListener("error", finish);
      ensureVideoSource(v);
      try { v.load(); } catch (e) { finish(); }
      if (videoReady(v)) finish();
    });
  }

function waitForBrowserPaint() {
    return new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  }

async function waitForVideoPainted(v) {
    if (v.error) return;
    await waitForVideoReady(v);
    await new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        requestAnimationFrame(resolve);
      };
      if (!v.paused && typeof v.requestVideoFrameCallback === "function") {
        v.requestVideoFrameCallback(finish);
        return;
      }
      requestAnimationFrame(() => requestAnimationFrame(finish));
    });
  }

async function waitForStageVideosReady() {
    const videos = replayVideos();
    if (!videos.length) {
      clientTrace("review.videos.ready", { count: 0, errors: 0 });
      return;
    }
    setVideosLoading(videos, true, "loading video");
    await Promise.all(videos.map((v) => waitForVideoReady(v)));
    clientTrace("review.videos.ready", {
      count: videos.length,
      errors: videos.filter((v) => !!v.error).length,
      states: videos.map((v) => ({ camera: v.dataset.key || "", ready_state: v.readyState })),
    });
  }

async function waitForStageVideosPainted() {
    const videos = replayVideos();
    if (!videos.length) return;
    setVideosLoading(videos, true, "rendering video");
    await Promise.all(videos.map((v) => waitForVideoPainted(v)));
    await waitForBrowserPaint();
    setVideosLoading(videos, false, "");
  }

function playStageVideos() {
    replayVideos().forEach((v) => {
      ensureVideoSource(v);
      if (!v.error) v.play().catch(() => {});
    });
  }

function pauseStageVideos() {
    replayVideos().forEach((v) => v.pause());
  }

function replayMasterVideo() {
    return replayVideos().find((v) => {
      const cell = v.closest(".cam-cell");
      return !v.error && (!cell || !cell.classList.contains("failed"));
    }) || null;
  }

function syncReplayVideos(frame, master = null, force = false) {
    const t = replayTimeAtFrame(frame);
    if (!Number.isFinite(t)) return;
    const tolerance = 0.5 / replayVideoFps;
    replayVideos().forEach((v) => {
      if (v === master) return;
      if (v.error) return;
      if (force || Math.abs((v.currentTime || 0) - t) > tolerance) {
        v.currentTime = t;
      }
    });
  }

let _replayUrdfSeq = 0;

let replayUrdfInFlight = false;

let replayUrdfPendingFrame = null;

let replayUrdfAppliedFrame = null;

function resetReplayUrdfRequests() {
    _replayUrdfSeq += 1;
    replayUrdfInFlight = false;
    replayUrdfPendingFrame = null;
    replayUrdfAppliedFrame = null;
  }

function replaySetUrdfFrame(frame) {
    if (replayApplyTransformFrame(frame)) {
      return;
    }
    // Stateless collection review computes a whole-episode transform blob in the
    // background. Do not issue unrelated mounted-replay frame requests while waiting;
    // the completion callback applies the current local cursor directly.
    if (["collect", "rollout"].includes(LIVE.replayOwner) && REPLAY_XF_LOADING) return;
    const i = Math.max(0, Math.min(Math.round(Number(frame) || 0), LIVE.n - 1));
    if (i === replayUrdfAppliedFrame && !replayUrdfInFlight) return;
    if (replayUrdfInFlight) {
      replayUrdfPendingFrame = i;
      return;
    }
    requestReplayUrdfFrame(i);
  }

async function requestReplayUrdfFrame(i) {
    replayUrdfInFlight = true;
    const seq = _replayUrdfSeq;
    try {
      const r = await apiGet("/api/replay_scene_frame?frame=" + i);
      if (seq !== _replayUrdfSeq) return;
      if (r.available && window.Scene3D) {
        Scene3D.applyTransforms({ arms: r.arms });
        replayUrdfAppliedFrame = i;
      }
    } catch (e) { /* transient */ }
    finally {
      if (seq !== _replayUrdfSeq) return;
      replayUrdfInFlight = false;
      const pending = replayUrdfPendingFrame;
      replayUrdfPendingFrame = null;
      if (pending !== null && pending !== replayUrdfAppliedFrame && LIVE.replayMode) {
        replaySetUrdfFrame(pending);
      }
    }
  }

function exitReplayMode() {
    replayLoadSeq += 1;
    replayStop();
    stopRealReplayVisual();
    LIVE.replayMode = false;
    LIVE.replayLoading = false;
    LIVE.replayOwner = "";
    LIVE.replayError = "";
    LIVE.cursorFrac = null;
    replayTransformsUrl = "/api/replay_transforms";
    replayLoadedDatasetDir = "";
    replayLoadedEpisodeId = 0;
    replayLoadedVideoKeys = {};
    REPLAY_XF = null; REPLAY_XF_NG = 0;
    REPLAY_XF_LOADING = false;
    replayTransformsPromise = null;
    REPLAY_XF_PARTS = null; REPLAY_XF_GEOMS = null; _lastReplayChartDraw = 0;
    resetReplayUrdfRequests();
    LIVE.playTime = [];
    // Drop the replay <video> elements so the live MJPEG strip rebuilds cleanly.
    replaceCamStripContent('<div class="cam-empty">awaiting frame…</div>');
    resetLiveSeries();
  }

function seekReplay(i, syncVideos = true) {
    LIVE.cursor = Math.max(0, Math.min(i, LIVE.n - 1));
    LIVE.cursorFrac = null;
    replaySetUrdfFrame(LIVE.cursor);
    if (syncVideos) syncReplayVideos(LIVE.cursor, null, true);
    updateScrub();
    drawReplayCharts();
  }

function setReplayCursorFrame(frame, syncVideos = false) {
    if (!LIVE.n) return;
    const frac = Math.max(0, Math.min(Number(frame) || 0, LIVE.n - 1));
    LIVE.cursorFrac = frac;
    LIVE.cursor = Math.max(0, Math.min(Math.floor(frac), LIVE.n - 1));
    replaySetUrdfFrame(frac);
    if (syncVideos) syncReplayVideos(frac, null, true);
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
    clientTrace("review.play", { episode: replayLoadedEpisodeId, cursor: LIVE.cursor, frames: LIVE.n });
    const btn = $("scrub-play"); if (btn) btn.textContent = "⏸";
    syncReplayRunButtons();
    playStageVideos();
    if (!REPLAY_XF && !REPLAY_XF_LOADING) {
      loadReplayTransforms(replayLoadSeq).then(() => {
        if (!LIVE.replayMode || !LIVE.playing) return;
        replaySetUrdfFrame(LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor);
      });
    }
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
      const cursor = LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor;
      const framePos = Math.max(cursor, replayFrameAtTime(targetTime));
      setReplayCursorFrame(framePos, false);
      syncReplayVideos(framePos, master);
      if (framePos >= LIVE.n - 1) { replayStop(); return; }
      LIVE.raf = requestAnimationFrame(frame);
    };
    LIVE.raf = requestAnimationFrame(frame);
  }

function replayStop() {
    const wasPlaying = LIVE.playing;
    LIVE.playing = false;
    if (LIVE.raf) { cancelAnimationFrame(LIVE.raf); LIVE.raf = null; }
    const btn = $("scrub-play"); if (btn) btn.textContent = "▶";
    syncReplayRunButtons();
    pauseStageVideos();
    setStageVideoLoading(false, "");
    drawReplayCharts(true);
    if (wasPlaying) {
      clientTrace("review.pause", { episode: replayLoadedEpisodeId, cursor: LIVE.cursor, frames: LIVE.n });
    }
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
    replaceCamStripContent('<div class="cam-empty">awaiting frame…</div>');
  }

async function pollFrame() {
    if (framePolling) return;
    if (S.ACTIVE_TAB === "replay") return;
    framePolling = true;
    try {
      const f = await apiGet("/api/frame");
      const strip = $("cam-strip");
      const keys = f.cameras || [];
      // REPLAY owns the cam strip with native <video> elements driven by the scrub
      // clock; never let the live MJPEG rebuild clobber them.
      if (strip && !LIVE.replayMode) {
        const existing = strip.querySelectorAll("img.cam");
        if (!keys.length) {
          // No live cameras (e.g. SIM DEBUG): drop any leftover replay <img> streams
          // so a previous tab's last frame can't stay frozen on screen.
          if (existing.length) replaceCamStripContent('<div class="cam-empty">awaiting frame…</div>');
        } else {
          const sameSet = existing.length === keys.length &&
            keys.every((k, i) => existing[i].dataset.key === k);
          if (!sameSet) {
            replaceCamStripContent(keys.map((k) =>
              `<div class="cam-cell"><div class="cam-lbl">${k}</div><img class="cam" data-key="${k}" src="/api/camera/${encodeURIComponent(k)}"></div>`
            ).join(""));
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
let lastScenePollAt = 0;
const COLLECT_SCENE_POLL_MS = 80;

function scenePollMinIntervalMs() {
    return S.STATUS && S.STATUS.collect && S.STATUS.collect.collecting
      ? COLLECT_SCENE_POLL_MS
      : 0;
  }

async function pollScene() {
    if (scenePolling) return;
    // REPLAY drives the URDF per-frame off the scrub clock (replaySetUrdfFrame); don't
    // let the live /api/scene poll fight it for the shared Scene3D canvas.
    if (S.ACTIVE_TAB === "replay" || LIVE.replayMode) return;
    const now = performance.now();
    const minInterval = scenePollMinIntervalMs();
    if (minInterval && now - lastScenePollAt < minInterval) return;
    lastScenePollAt = now;
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
  exitReplayMode, loadMountedReplaySeries, loadReviewPlayback, maybeSyncReplayPlayer,
  mountEpisodeVideos, playStageVideos, replayPlay, replayStop, replayToggle,
  replayVideos, seekReplay, waitForStageVideosPainted,
  waitForStageVideosReady,
  loop, pollFrame, pollScene, refreshCameraStreams,
};
