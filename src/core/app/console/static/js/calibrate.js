// Shared CAMERAS panel (DEBUG left column) + MANUAL-tab calibration workspace.
// PR#1 surface: read-only /api/calibration -> per-camera rows with a
// Calibrated/Missing badge, plus Scene3D camera-frustum push. Extended by PR#2
// with the MANUAL-tab BOARD/CAMERAS/POSES/SOLVE/SAVE workspace at file bottom.
import { $, apiGet, apiPost } from "./core.js";

let _last = null;

function _statusBadge(entry) {
  if (entry && entry.calibrated) return '<span class="cam-badge ok">CALIBRATED</span>';
  return '<span class="cam-badge warn">MISSING</span>';
}

function _fmtK(K) {
  if (!Array.isArray(K)) return "";
  const fx = Number(K[0][0]).toFixed(1), fy = Number(K[1][1]).toFixed(1);
  const cx = Number(K[0][2]).toFixed(1), cy = Number(K[1][2]).toFixed(1);
  return `fx ${fx} · fy ${fy} · cx ${cx} · cy ${cy}`;
}

function _fmtDist(dist) {
  if (!Array.isArray(dist) || !dist.length) return "dist —";
  return "dist [" + dist.slice(0, 5).map((x) => Number(x).toFixed(3)).join(", ")
    + (dist.length > 5 ? ", …" : "") + "]";
}

export function paintCamerasPanel(cams) {
  const host = $("cameras-list");
  if (!host) return;
  const names = Object.keys(cams || {});
  if (!names.length) {
    host.innerHTML = '<div class="lock-note">no calibration loaded</div>';
    return;
  }
  host.innerHTML = names.map((name) => {
    const c = cams[name] || {};
    const meta = c.calibrated
      ? `<div class="cam-meta">${_fmtK(c.K)}</div>`
        + `<div class="cam-meta">${_fmtDist(c.dist)}</div>`
        + `<div class="cam-meta">attach: ${c.attach_link || "(world)"}</div>`
      : '<div class="cam-meta">no K/dist loaded</div>';
    return `<div class="cam-row"><div><div class="cam-name">${name}</div>${meta}</div>${_statusBadge(c)}</div>`;
  }).join("");
}

function _pushToScene(cams) {
  if (window.Scene3D && typeof window.Scene3D.setCameraFrusta === "function") {
    window.Scene3D.setCameraFrusta(cams);
  }
}

export async function refreshCameras() {
  try {
    _last = await apiGet("/api/calibration");
  } catch (e) {
    _last = {};
  }
  paintCamerasPanel(_last);
  _pushToScene(_last);
}

export async function reloadCameras() {
  try {
    const resp = await apiPost("/api/calibration/reload", {});
    _last = (resp && resp.cameras) || _last || {};
  } catch (e) {
    return refreshCameras();
  }
  paintCamerasPanel(_last);
  _pushToScene(_last);
}

// ---------------------------------------------------------------------------
// CALIBRATE tab (position 07): full guided flow.
// ---------------------------------------------------------------------------

const DEFAULT_DICTS = [
  "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
  "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
  "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250",
  "DICT_7X7_250",
];

let _cal = null;
let _pollTimer = null;
let _editingIndex = -1;     // pose being edited; -1 = none
let _editingQpos = null;    // draft qpos while editing (streams to Scene3D)

function _fmtRms(r) { return r == null ? "—" : Number(r).toFixed(2); }

function _setPanelState(id, state) {
  const el = $(id);
  if (el) el.setAttribute("data-st", state);
}

function _setBtnEnabled(id, on) {
  const el = $(id);
  if (!el) return;
  if (on) el.removeAttribute("disabled");
  else el.setAttribute("disabled", "");
}

// -- BOARD ------------------------------------------------------------------

function _paintBoardPanel() {
  const host = $("cal-board-dict");
  if (host && host.options.length === 0) {
    for (const name of DEFAULT_DICTS) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      host.appendChild(opt);
    }
    host.value = "DICT_5X5_100";
  }
  if (_cal && _cal.board) {
    if (host) host.value = _cal.board.dict;
    const cols = $("cal-board-cols"); if (cols) cols.value = _cal.board.cols;
    const rows = $("cal-board-rows"); if (rows) rows.value = _cal.board.rows;
    const sq = $("cal-board-square"); if (sq) sq.value = _cal.board.square;
    const mk = $("cal-board-marker"); if (mk) mk.value = _cal.board.marker;
  }
  _setPanelState("calib-panel-board", _cal && _cal.active ? "done" : "active");
}

// -- MODE / guidance --------------------------------------------------------

function _paintModePanel() {
  const sim = $("b-cal-mode-sim"), real = $("b-cal-mode-real");
  const mode = _cal && _cal.mode ? _cal.mode : "sim";
  if (sim) { sim.classList.toggle("primary", mode === "sim"); }
  if (real) { real.classList.toggle("primary", mode === "real"); }
  const guide = $("cal-guidance");
  if (!guide) return;
  if (!_cal || !_cal.active) {
    guide.textContent = "Step 1 · print the ChArUco board, then click START to begin.";
    return;
  }
  const n = _cal.n_poses || 0;
  const captured = _cal.n_captured || 0;
  const idx = _cal.current_pose_index;
  if (mode === "sim") {
    guide.innerHTML = `SIM · previewing pose ${idx + 1}/${n}. Move &amp; edit poses freely. Switch to REAL to CAPTURE.`;
  } else if (n === 0) {
    guide.textContent = "REAL · no poses yet. Use + ADD FROM CURRENT to seed the list.";
  } else if (captured < n) {
    guide.innerHTML = `REAL · pose ${idx + 1}/${n} · <b>${captured}/${n}</b> captured. MOVE → CAPTURE → next.`;
  } else {
    guide.textContent = "REAL · all poses captured. Now SOLVE and SAVE.";
  }
}

// -- CAMERAS ----------------------------------------------------------------

function _paintCamerasPanel() {
  const host = $("cal-cam-list");
  const activeSel = $("cal-active-cam");
  if (!host) return;
  if (!_cal || !_cal.active) {
    host.innerHTML = '<div class="lock-note">START session to enumerate cameras</div>';
    if (activeSel) { activeSel.innerHTML = ""; activeSel.disabled = true; }
    _setPanelState("calib-panel-cameras", "disabled");
    return;
  }
  const keys = _cal.camera_keys || [];
  host.innerHTML = keys.map((key) => {
    const c = _cal.per_camera[key] || {};
    const src = c.intrinsic_source || "sdk_or_solve";
    const attach = c.attach_link || "";
    const radios = ["sdk", "solve", "sdk_or_solve"].map((v) =>
      `<label class="${src === v ? "on" : ""}" data-key="${key}" data-src="${v}">
        <input type="radio" name="src-${key}" value="${v}" ${src === v ? "checked" : ""}>${v}
      </label>`
    ).join("");
    return `<div class="cal-cam-item">
      <div>
        <div class="cal-cam-title">${key}</div>
        <div class="cal-cam-attach">attach: ${attach || "(world)"}</div>
      </div>
      <div class="cal-source-radios">${radios}</div>
    </div>`;
  }).join("");
  host.querySelectorAll(".cal-source-radios label").forEach((lbl) => {
    lbl.addEventListener("click", (e) => {
      e.preventDefault();
      setIntrinsicSource(lbl.dataset.key, lbl.dataset.src);
    });
  });
  if (activeSel) {
    activeSel.disabled = false;
    const prev = activeSel.value;
    activeSel.innerHTML = keys.map((k) => `<option value="${k}">${k}</option>`).join("");
    if (keys.includes(prev)) activeSel.value = prev;
  }
  _setPanelState("calib-panel-cameras", "done");
}

// -- POSES ------------------------------------------------------------------

function _paintPosesPanel() {
  const badge = $("cal-poses-badge");
  const listHost = $("cal-pose-list");
  const status = $("cal-capture-status");
  const thumbs = $("cal-thumbs");
  const canCapture = !!(_cal && _cal.active && _cal.mode === "real");
  _setBtnEnabled("b-cal-capture", canCapture);
  _setBtnEnabled("b-cal-add-current", !!(_cal && _cal.active));

  if (!_cal || !_cal.active) {
    if (badge) badge.textContent = "";
    if (listHost) listHost.innerHTML = "";
    _setPanelState("calib-panel-poses", "disabled");
    if (thumbs) thumbs.innerHTML = "";
    return;
  }
  const poses = _cal.poses || [];
  const capState = _cal.pose_capture_state || [];
  const cur = _cal.current_pose_index;
  const n = poses.length;
  const captured = _cal.n_captured || 0;
  if (badge) badge.textContent = `${captured}/${n}`;
  if (listHost) {
    listHost.innerHTML = poses.map((_, i) => {
      const done = capState[i] ? '<span class="cal-pose-state done">DONE</span>'
                               : '<span class="cal-pose-state todo">TODO</span>';
      const selected = i === cur ? "selected" : "";
      return `<div class="cal-pose-row ${selected}">
        <span class="cal-pose-idx">${i + 1}</span>
        ${done}
        <span></span>
        <span class="cal-pose-actions">
          <button class="btn" data-cal-pose="select" data-i="${i}">SELECT</button>
          <button class="btn" data-cal-pose="edit" data-i="${i}">EDIT</button>
          <button class="btn primary" data-cal-pose="move" data-i="${i}">MOVE</button>
          <button class="btn" data-cal-pose="delete" data-i="${i}">×</button>
        </span>
      </div>`;
    }).join("");
    listHost.querySelectorAll("button[data-cal-pose]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const i = parseInt(btn.dataset.i, 10);
        const op = btn.dataset.calPose;
        if (op === "select") posesOp("select", i);
        else if (op === "edit") beginEditPose(i);
        else if (op === "move") moveToPose(i);
        else if (op === "delete") posesOp("delete", i);
      });
    });
  }
  const thumbHTML = [];
  for (const key of (_cal.camera_keys || [])) {
    const c = _cal.per_camera[key] || {};
    if (c.last_thumb) thumbHTML.push(`<div class="cal-thumb" title="${key}"><img src="${c.last_thumb}"></div>`);
  }
  if (thumbs) thumbs.innerHTML = thumbHTML.join("");
  if (status) {
    if (_cal.last_error) { status.textContent = _cal.last_error; status.className = "cal-status err"; }
    else if (_cal.mode === "sim") { status.textContent = "SIM — CAPTURE disabled. Switch to REAL to record."; status.className = "cal-status"; }
    else if (captured === 0) { status.textContent = "aim the board at the target camera, then click CAPTURE"; status.className = "cal-status"; }
    else { status.textContent = `${captured}/${n} captured`; status.className = "cal-status ok"; }
  }
  _setPanelState("calib-panel-poses", captured === n && n > 0 ? "done" : "active");
}

function _paintPoseEditor() {
  const editor = $("cal-pose-editor");
  if (!editor) return;
  if (_editingIndex < 0 || !_cal || !_cal.poses || _editingIndex >= _cal.poses.length) {
    editor.style.display = "none";
    return;
  }
  editor.style.display = "";
  const idxLabel = $("cal-edit-idx");
  if (idxLabel) idxLabel.textContent = _editingIndex + 1;
  const host = $("cal-joint-sliders");
  if (!host) return;
  if (_editingQpos === null) _editingQpos = _cal.poses[_editingIndex].slice();
  // Only rebuild slider DOM when joint count changes.
  if (host.dataset.n !== String(_editingQpos.length)) {
    host.innerHTML = _editingQpos.map((v, j) =>
      `<div class="cal-slider-row">
        <span>j${j}</span>
        <input type="range" min="-3.14" max="3.14" step="0.01" value="${v}" data-j="${j}">
        <span class="cal-slider-val" data-j="${j}">${Number(v).toFixed(2)}</span>
      </div>`
    ).join("");
    host.dataset.n = String(_editingQpos.length);
    host.querySelectorAll("input[type=range]").forEach((inp) => {
      inp.addEventListener("input", (e) => {
        const j = parseInt(inp.dataset.j, 10);
        _editingQpos[j] = parseFloat(inp.value);
        const val = host.querySelector(`.cal-slider-val[data-j="${j}"]`);
        if (val) val.textContent = Number(_editingQpos[j]).toFixed(2);
        _pushGhostToScene(_editingQpos);
      });
    });
  } else {
    // Re-sync slider values on external state change.
    host.querySelectorAll("input[type=range]").forEach((inp) => {
      const j = parseInt(inp.dataset.j, 10);
      if (Math.abs(parseFloat(inp.value) - _editingQpos[j]) > 1e-4) {
        inp.value = _editingQpos[j];
      }
    });
  }
}

// -- SOLVE ------------------------------------------------------------------

function _paintSolvePanel() {
  const host = $("cal-outputs");
  const canSolve = !!(_cal && _cal.active && (_cal.n_captured || 0) >= 4);
  _setBtnEnabled("b-cal-solve", canSolve);
  const methodSel = $("cal-hand-eye-method");
  if (methodSel) methodSel.disabled = !canSolve;
  if (!host || !_cal || !_cal.active) {
    if (host) host.innerHTML = "";
    _setPanelState("calib-panel-solve", "disabled");
    return;
  }
  const rows = [];
  let anySolved = false;
  for (const key of (_cal.camera_keys || [])) {
    const c = _cal.per_camera[key] || {};
    const intrinsic = c.intrinsic;
    const he = c.hand_eye;
    const intrinsicLine = intrinsic
      ? `${_fmtK(intrinsic.K)} · rms ${_fmtRms(intrinsic.rms)}px (${intrinsic.method})`
      : "no intrinsic yet";
    const heLine = he
      ? `hand-eye ${he.method} · ${_fmtRms(he.translation_rms_m)} m · ${_fmtRms(he.rotation_rms_deg)}°`
      : "";
    if (intrinsic) anySolved = true;
    rows.push(`<div class="cal-outrow" id="cal-out-${key}">
      <div class="cal-out-cam">${key}</div>
      <div class="cal-out-line">${intrinsicLine}</div>
      ${heLine ? `<div class="cal-out-line">${heLine}</div>` : ""}
    </div>`);
  }
  host.innerHTML = rows.join("");
  _setPanelState("calib-panel-solve", anySolved ? "done" : (canSolve ? "active" : "disabled"));
}

// -- SAVE -------------------------------------------------------------------

function _paintSavePanel() {
  const path = $("cal-save-path");
  if (path) path.textContent = _cal && _cal.saved_path ? `last saved: ${_cal.saved_path}` : "";
  const canSave = !!(_cal && _cal.active && _cal.ready_to_save);
  _setBtnEnabled("b-cal-save", canSave);
  _setPanelState("calib-panel-save", canSave ? "active" : "disabled");
}

// -- Scene3D ghost preview --------------------------------------------------

function _pushGhostToScene(qpos) {
  if (!qpos) return;
  if (!window.Scene3D || typeof window.Scene3D.setPreviewQpos !== "function") return;
  window.Scene3D.setPreviewQpos(qpos);
}

// -- Poll loop --------------------------------------------------------------

function _paint() {
  _paintBoardPanel();
  _paintModePanel();
  _paintCamerasPanel();
  _paintPosesPanel();
  _paintPoseEditor();
  _paintSolvePanel();
  _paintSavePanel();
}

async function _pull() {
  try {
    _cal = await apiGet("/api/calibrate/session");
  } catch (e) {
    _cal = null;
  }
  _paint();
}

function _startPolling() {
  if (_pollTimer !== null) return;
  _pollTimer = setInterval(async () => {
    if (document.querySelector('.tab.active')?.dataset.tab !== "calibrate") {
      _stopPolling();
      return;
    }
    if (_cal && _cal.active) await _pull();
  }, 700);
}

function _stopPolling() {
  if (_pollTimer !== null) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

export async function renderCalibratePanels() {
  await _pull();
  _paint();
  _startPolling();
}

// -- Exported button handlers ----------------------------------------------

export async function startCalibrate() {
  try {
    _cal = await apiPost("/api/calibrate/start", {});
    _paint();
    _startPolling();
  } catch (e) {
    const status = $("cal-capture-status");
    if (status) { status.textContent = "start failed"; status.className = "cal-status err"; }
  }
}

export async function resetCalibrate() {
  await apiPost("/api/calibrate/reset", {});
  _cal = null;
  _editingIndex = -1;
  _editingQpos = null;
  _paint();
  _stopPolling();
}

export async function applyBoard() {
  const body = {
    dict: $("cal-board-dict").value,
    cols: parseInt($("cal-board-cols").value, 10),
    rows: parseInt($("cal-board-rows").value, 10),
    square: parseFloat($("cal-board-square").value),
    marker: parseFloat($("cal-board-marker").value),
  };
  try {
    _cal = await apiPost("/api/calibrate/board", body);
  } catch (e) {}
  _paint();
}

export async function setMode(mode) {
  try {
    _cal = await apiPost("/api/calibrate/mode", { mode });
  } catch (e) {}
  _paint();
}

export async function setIntrinsicSource(camera_key, source) {
  try {
    _cal = await apiPost("/api/calibrate/intrinsic_source", { camera_key, source });
  } catch (e) {}
  _paint();
}

export function currentCalibCam() {
  return $("cal-active-cam")?.value || "";
}

export async function captureSample(camera_key) {
  if (!camera_key) return;
  const status = $("cal-capture-status");
  try {
    _cal = await apiPost("/api/calibrate/capture", { camera_key });
    _paint();
    if (status) {
      const ok = _cal && _cal.capture_ok;
      const msg = (_cal && _cal.capture_msg) || "";
      status.textContent = ok ? `capture ok — ${msg}` : `capture failed — ${msg || 'no detection'}`;
      status.className = ok ? "cal-status ok" : "cal-status err";
    }
  } catch (e) {
    if (status) { status.textContent = "capture failed"; status.className = "cal-status err"; }
  }
}

export async function solveAll(method) {
  const btn = $("b-cal-solve");
  if (btn) btn.disabled = true;
  try {
    _cal = await apiPost("/api/calibrate/solve", { method });
  } catch (e) {}
  _paint();
}

export async function saveCalibration() {
  const status = $("cal-save-status");
  try {
    const resp = await apiPost("/api/calibrate/save", {});
    _cal = resp;
    _paint();
    if (resp && resp.cameras) {
      _last = resp.cameras;
      paintCamerasPanel(_last);
      _pushToScene(_last);
    }
    if (status) {
      const ok = resp && resp.ok;
      status.textContent = ok ? "saved" : ((resp && resp.error) || "save failed");
      status.className = ok ? "cal-status ok" : "cal-status err";
    }
    _setPanelState("calib-panel-save", "done");
  } catch (e) {
    if (status) { status.textContent = "save failed"; status.className = "cal-status err"; }
  }
}

export async function posesOp(op, index, qpos) {
  const body = { op };
  if (index !== undefined) body.index = index;
  if (qpos !== undefined) body.qpos = qpos;
  try {
    _cal = await apiPost("/api/calibrate/poses", body);
  } catch (e) {}
  _paint();
}

export async function moveToPose(index) {
  const mode = (_cal && _cal.mode) || "sim";
  if (mode === "real") {
    const ok = window.confirm(`Move real robot to pose ${index + 1}?`);
    if (!ok) return;
  }
  try {
    _cal = await apiPost("/api/calibrate/move_to", { index });
  } catch (e) {}
  // In SIM: also push the target qpos as a ghost.
  if (_cal && _cal.poses && index >= 0 && index < _cal.poses.length) {
    _pushGhostToScene(_cal.poses[index]);
  }
  _paint();
}

export function beginEditPose(index) {
  if (!_cal || !_cal.poses || index < 0 || index >= _cal.poses.length) return;
  _editingIndex = index;
  _editingQpos = _cal.poses[index].slice();
  _paint();
  _pushGhostToScene(_editingQpos);
}

export async function commitEditPose() {
  if (_editingIndex < 0 || _editingQpos === null) return;
  await posesOp("replace", _editingIndex, _editingQpos);
  _editingIndex = -1;
  _editingQpos = null;
  _paint();
}

export function cancelEditPose() {
  _editingIndex = -1;
  _editingQpos = null;
  _paint();
}

export async function addPoseFromCurrent() {
  await posesOp("add");
}

export async function estop() {
  try { await apiPost("/api/halt", {}); } catch (e) {}
}
