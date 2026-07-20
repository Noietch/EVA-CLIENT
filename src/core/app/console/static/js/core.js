// core.js: shared store + DOM helper + JSON fetch helpers.

export const $ = (id) => document.getElementById(id);

function replaceCamStripContent(html) {
  const strip = $("cam-strip");
  if (!strip) return;
  strip.querySelectorAll("video.cam").forEach((v) => {
    v.pause();
    v.removeAttribute("src");
    try { v.load(); } catch (e) {}
  });
  strip.innerHTML = html;
}

export const RUN_CONTROLS = {
  debug: {
    runGroup: "control-run", stepGroup: "control-step", stepState: "step-state",
    run: "b-run", halt: "b-halt", step: "b-step", commit: "b-commit",
    stepHalt: "b-step-halt", stepReset: "b-step-reset",
  },
  replay: {
    runGroup: "replay-control-run", stepGroup: "replay-control-step", stepState: "replay-step-state",
    run: "b-replay-run", halt: "b-replay-halt", step: "b-replay-step", commit: "b-replay-commit",
    stepHalt: "b-replay-step-halt", stepReset: "b-replay-step-reset",
  },
};

export const LIVE = {
  timestamp: [], playTime: [], action: [], state: [], n: 0,
  criticTimestamp: [], criticValue: [],
  controlSource: [], intervention: [], interventionSegmentIndex: [],
  actionNames: [], stateNames: [],
  dimsOnA: {}, dimsOnS: {}, dimsBuilt: false,
  following: true, cursor: 0,
  replayMode: false, replayLoading: false, playing: false, raf: null, cursorFrac: null,
  replayOwner: "", replayError: "",
  replaySync: {
    samples: 0, videoSamples: 0, maxReadyVideos: 0, expectedVideos: 0,
    maxVideoSkewSec: 0, maxUrdfFrameSkew: 0,
    maxFrameGapMs: 0, lastFrameGapMs: 0, lastVideoSkewSec: 0, lastUrdfFrameSkew: 0,
  },
};

export const RT_COLORS = ["#FF4D00", "#1F7A4D", "#2563EB", "#B0A14F", "#9B59B6", "#16A085", "#D08C60", "#5C8AC6", "#C0341E", "#6AA84F", "#A36B3E", "#7E8CE0", "#C9A227", "#46998A"];

// Cross-module mutable state. Modules read/write via S.<name> so the binding is shared.
export const S = {
  CFG: null,
  STATUS: {},
  ACTIVE_TAB: "debug",
  collectReplayEpisode: null,
  reviewKind: "",
  replayVideoKeys: {},
  qcEpisode: null,
  pendingQcLoad: null,
  qcMode: false,
  manualActive: false,
  realRequested: false,
  realConnected: false,
  manualDispatching: false,
  manualDispatchStopPending: false,
  _manualSlidersBuilt: false,
  _setupFired: false,
  _replayModePushed: false,
  _setupPaused: false,
  replayLoadPending: false,
  replaySeriesKey: null,
  chartModalWhich: null,
  EVAL_MODEL_NAME: "",
  EVAL_EPISODES_DIR: "",
  collectQueueExpanded: false,
  collectQueueEnabled: false,
  collectArmEnabled: false,
  collectToggleBusy: null,
  rolloutSaveQueueExpanded: false,
  rolloutSaveEpisode: null,
  runToggleBusy: null,
  evalRunToggleBusy: null,
  rlTask: "",
  rlPolicy: "",
  rlCritic: "",
  rlSaveExpanded: false,
};

const CLIENT_TRACE_ID = (() => {
  try {
    const existing = sessionStorage.getItem("eva-client-trace-id");
    if (existing) return existing;
    const id = globalThis.crypto && crypto.randomUUID
      ? crypto.randomUUID()
      : `browser-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem("eva-client-trace-id", id);
    return id;
  } catch (e) {
    return `browser-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }
})();

let clientTraceSeq = 0;

function clientTrace(event, details = {}, traceId = null) {
  const payload = {
    client_id: CLIENT_TRACE_ID,
    seq: ++clientTraceSeq,
    event,
    details: { tab: S.ACTIVE_TAB, review_kind: S.reviewKind, ...details },
  };
  console.info("[EVA_TRACE]", payload);
  fetch("/api/client_trace", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(traceId ? { "X-EVA-Trace-ID": traceId } : {}),
    },
    body: JSON.stringify(payload),
    keepalive: true,
  }).catch(() => {});
}

window.addEventListener("error", (event) => {
  clientTrace("window.error", {
    message: event.message || "unknown error",
    source: event.filename || "",
    line: event.lineno || 0,
    column: event.colno || 0,
  });
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  clientTrace("window.unhandledrejection", {
    message: reason && reason.message ? reason.message : String(reason || "unknown rejection"),
    stack: reason && reason.stack ? String(reason.stack).slice(0, 1200) : "",
  });
});

async function apiGet(path) {
  try {
    const r = await fetch(path);
    if (!r.ok) clientTrace("api.get.error", { path, status: r.status });
    return await r.json();
  } catch (error) {
    clientTrace("api.get.exception", { path, message: String(error) });
    throw error;
  }
}

let postQueue = Promise.resolve();

async function apiPost(path, body) {
  const request = async () => {
    const traceId = `${CLIENT_TRACE_ID}:${clientTraceSeq + 1}`;
    const started = performance.now();
    clientTrace("api.post.begin", { path }, traceId);
    try {
      const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-EVA-Trace-ID": traceId },
        body: JSON.stringify(body || {}),
      });
      const payload = await r.json();
      clientTrace("api.post.end", {
        path,
        status: r.status,
        ok: !!(r.ok && payload && payload.ok !== false),
        error: payload && payload.error ? String(payload.error).slice(0, 500) : "",
        elapsed_ms: Math.round((performance.now() - started) * 10) / 10,
      }, traceId);
      return payload;
    } catch (error) {
      clientTrace("api.post.error", {
        path,
        message: String(error),
        elapsed_ms: Math.round((performance.now() - started) * 10) / 10,
      }, traceId);
      throw error;
    }
  };
  const result = postQueue.catch(() => {}).then(request);
  postQueue = result.then(() => undefined, () => undefined);
  return result;
}
export { apiGet, apiPost, clientTrace, replaceCamStripContent };
