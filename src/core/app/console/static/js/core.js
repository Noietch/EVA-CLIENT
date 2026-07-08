// core.js: shared store + DOM helper + JSON fetch helpers.

export const $ = (id) => document.getElementById(id);

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
  actionNames: [], stateNames: [],
  dimsOnA: {}, dimsOnS: {}, dimsBuilt: false,
  following: true, cursor: 0,
  replayMode: false, replayLoading: false, playing: false, raf: null, cursorFrac: null,
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
};

async function apiGet(path) { const r = await fetch(path); return r.json(); }

let postQueue = Promise.resolve();

async function apiPost(path, body) {
  const request = async () => {
    const r = await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return r.json();
  };
  const result = postQueue.catch(() => {}).then(request);
  postQueue = result.then(() => undefined, () => undefined);
  return result;
}
export { apiGet, apiPost };
