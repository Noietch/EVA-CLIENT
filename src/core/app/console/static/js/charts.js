// charts.js: live-telemetry & series chart rendering, chart modal, scrubber.
import { $, LIVE, RT_COLORS, S } from "./core.js";
import { replayStop, seekReplay } from "./replay.js";

function buildLiveDims() {
    const da = LIVE.action.length ? LIVE.action[0].length : 0;
    const ds = LIVE.state.length ? LIVE.state[0].length : 0;
    if (!da && !ds) return;
    for (let d = 0; d < da; d++) if (!(d in LIVE.dimsOnA)) LIVE.dimsOnA[d] = true;
    for (let d = 0; d < ds; d++) if (!(d in LIVE.dimsOnS)) LIVE.dimsOnS[d] = true;
    LIVE.dimsBuilt = true;
    renderLiveDims("chart-action-dims", da, LIVE.dimsOnA, "a");
    renderLiveDims("chart-state-dims", ds, LIVE.dimsOnS, "s");
  }

function renderLiveDims(hostId, nd, dimsOn, tag) {
    const host = $(hostId); if (!host) return;
    host.innerHTML = "";
    const names = tag === "a" ? LIVE.actionNames : LIVE.stateNames;
    for (let d = 0; d < nd; d++) {
      const el = document.createElement("span");
      el.className = "dim" + (dimsOn[d] ? "" : " off");
      el.style.borderLeft = "8px solid " + RT_COLORS[d % RT_COLORS.length];
      el.textContent = names[d] || ((tag === "a" ? "a" : "q") + d);
      el.title = el.textContent;
      el.onclick = () => { dimsOn[d] = !dimsOn[d]; renderLiveDims(hostId, nd, dimsOn, tag); drawLiveCharts(); };
      host.appendChild(el);
    }
  }

function liveDimsAll(which) {
    const dimsOn = which === "a" ? LIVE.dimsOnA : LIVE.dimsOnS;
    const ks = Object.keys(dimsOn);
    if (!ks.length) return;
    const anyOff = ks.some((d) => !dimsOn[d]);   // ALL turns every dim on; if all on, turn off
    ks.forEach((d) => { dimsOn[d] = anyOff; });
    if (which === "a") renderLiveDims("chart-action-dims", ks.length, LIVE.dimsOnA, "a");
    else renderLiveDims("chart-state-dims", ks.length, LIVE.dimsOnS, "s");
    if (S.chartModalWhich === which) renderLiveDims("chart-modal-dims", ks.length, dimsOn, which === "a" ? "a" : "s");
    drawLiveCharts();
  }

function drawLiveCharts() {
    const cursor = LIVE.replayMode && LIVE.cursorFrac != null
      ? LIVE.cursorFrac
      : ((!LIVE.replayMode && LIVE.following) ? (LIVE.n - 1) : LIVE.cursor);
    const ts = LIVE.replayMode && LIVE.playTime.length === LIVE.n ? LIVE.playTime : LIVE.timestamp;
    drawSeriesChart($("chart-action-cv"), LIVE.action, ts, LIVE.dimsOnA, cursor);
    drawSeriesChart($("chart-state-cv"), LIVE.state, ts, LIVE.dimsOnS, cursor);
    if (S.chartModalWhich) {
      const mat = S.chartModalWhich === "a" ? LIVE.action : LIVE.state;
      const dimsOn = S.chartModalWhich === "a" ? LIVE.dimsOnA : LIVE.dimsOnS;
      drawSeriesChart($("chart-modal-cv"), mat, ts, dimsOn, cursor);
    }
  }

function openChartModal(which) {
    S.chartModalWhich = which;
    const nd = which === "a"
      ? (LIVE.action.length ? LIVE.action[0].length : 0)
      : (LIVE.state.length ? LIVE.state[0].length : 0);
    const dimsOn = which === "a" ? LIVE.dimsOnA : LIVE.dimsOnS;
    $("chart-modal-title").textContent = which === "a" ? "ACTION" : "STATE";
    renderLiveDims("chart-modal-dims", nd, dimsOn, which === "a" ? "a" : "s");
    $("chart-modal").classList.add("on");
    requestAnimationFrame(drawLiveCharts);
  }

function closeChartModal() {
    S.chartModalWhich = null;
    $("chart-modal").classList.remove("on");
  }

function setScrubValue(frac) {
    const range = $("scrub-range");
    if (!range) return;
    range.value = String(frac);
    const max = Math.max(LIVE.n - 1, 0);
    const pct = max > 0 ? (Math.max(0, Math.min(frac, max)) / max) * 100 : 0;
    range.style.setProperty("--pct", pct.toFixed(2) + "%");
  }

function updateScrubText() {
    const idx = LIVE.replayMode && LIVE.cursorFrac != null
      ? LIVE.cursorFrac
      : ((!LIVE.replayMode && LIVE.following) ? (LIVE.n - 1) : LIVE.cursor);
    const shown = LIVE.n ? Math.floor(Math.max(0, Math.min(idx, LIVE.n - 1))) + 1 : 0;
    $("scrub-pos").textContent = shown + " / " + LIVE.n;
    const ts = LIVE.replayMode && LIVE.playTime.length === LIVE.n ? LIVE.playTime : LIVE.timestamp;
    const t = LIVE.n ? timeAtIndex(ts, idx) - (ts[0] || 0) : 0;
    $("scrub-time").textContent = t.toFixed(1) + "s";
  }

function updateScrub() {
    const range = $("scrub-range"), stateEl = $("scrub-state"), bar = $("stage-scrub");
    if (!range) return;
    range.max = String(Math.max(LIVE.n - 1, 0));
    range.step = "0.01";   // fine step so the thumb can glide between frames
    // The local play button only makes sense in REPLAY mode; a REAL run owns the cursor
    // (chase-the-hardware), so hide it there to keep playback authority with the robot.
    const playBtn = $("scrub-play");
    if (playBtn) {
      const realRun = S.STATUS && S.STATUS.session_status === "running" && S.STATUS.cli_mode === "real";
      playBtn.style.display = (LIVE.replayMode && !LIVE.replayLoading && !realRun) ? "" : "none";
    }
    if (LIVE.replayMode) {
      bar.classList.remove("live");
      stateEl.textContent = "REPLAY";
      range.disabled = LIVE.n === 0;
      setScrubValue(LIVE.cursorFrac != null ? LIVE.cursorFrac : LIVE.cursor);
    } else if (LIVE.following) {
      bar.classList.add("live");
      stateEl.textContent = "LIVE";
      range.disabled = true;
      setScrubValue(Math.max(LIVE.n - 1, 0));
      LIVE.cursor = LIVE.n - 1;
    } else {
      bar.classList.remove("live");
      stateEl.textContent = "REVIEW";
      range.disabled = LIVE.n === 0;
      setScrubValue(LIVE.cursor);
    }
    updateScrubText();
  }

function onScrubInput(v) {
    const i = Math.max(0, Math.min(Math.round(parseFloat(v) || 0), LIVE.n - 1));
    if (LIVE.replayMode) {
      if (LIVE.playing) replayStop();   // manual scrub takes over from the play clock
      LIVE.cursorFrac = null;
      seekReplay(i);
      return;
    }
    LIVE.cursor = i;
    updateScrub();
    drawLiveCharts();
  }

function resetLiveSeries() {
    LIVE.timestamp = []; LIVE.playTime = []; LIVE.action = []; LIVE.state = []; LIVE.n = 0;
    LIVE.actionNames = []; LIVE.stateNames = [];
    LIVE.dimsOnA = {}; LIVE.dimsOnS = {}; LIVE.dimsBuilt = false;
    LIVE.following = true; LIVE.cursor = 0;
    renderLiveDims("chart-action-dims", 0, LIVE.dimsOnA, "a");
    renderLiveDims("chart-state-dims", 0, LIVE.dimsOnS, "s");
    updateScrub(); drawLiveCharts();
  }

function timeAtIndex(ts, idx) {
    if (!ts || !ts.length) return 0;
    const clamped = Math.max(0, Math.min(Number(idx) || 0, ts.length - 1));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(i0 + 1, ts.length - 1);
    const a = clamped - i0;
    const t0 = Number(ts[i0]) || 0;
    const t1 = Number(ts[i1]) || t0;
    return t0 + (t1 - t0) * a;
  }

function drawSeriesChart(cv, mat, ts, dimsOn, cursorIdx) {
    const rect = cv.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    const dpr = Math.min(window.devicePixelRatio, 2);
    cv.width = rect.width * dpr; cv.height = rect.height * dpr;
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height, pad = 26;
    ctx.clearRect(0, 0, W, H);
    const n = mat ? mat.length : 0;
    const sd = n ? mat[0].length : 0;
    const dims = []; for (let d = 0; d < sd; d++) if (dimsOn[d]) dims.push(d);
    if (!n || !dims.length) {
      ctx.fillStyle = "rgba(0,0,0,0.3)"; ctx.font = "10px monospace";
      ctx.fillText(n ? "no dims" : "awaiting data…", pad + 6, H / 2);
      return;
    }
    let lo = Infinity, hi = -Infinity;
    for (const d of dims) for (let i = 0; i < n; i++) { const v = mat[i][d]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) { lo = -1; hi = 1; }
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const t0 = ts[0], t1 = ts[n - 1] || (t0 + 1);
    const X = (t) => pad + (W - 2 * pad) * (t1 > t0 ? (t - t0) / (t1 - t0) : 0);
    const Y = (v) => H - pad - (H - 2 * pad) * (v - lo) / (hi - lo);
    ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.font = "9px monospace";
    ctx.fillText(hi.toFixed(2), 2, pad + 4); ctx.fillText(lo.toFixed(2), 2, H - pad);
    ctx.save();
    ctx.globalAlpha = 0.6;           // soften the trace lines; legend chips stay vivid
    ctx.lineJoin = "round";
    for (const d of dims) {
      ctx.strokeStyle = RT_COLORS[d % RT_COLORS.length]; ctx.lineWidth = 1.4; ctx.beginPath();
      for (let i = 0; i < n; i++) { const x = X(ts[i]), y = Y(mat[i][d]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    }
    ctx.restore();
    if (cursorIdx != null && cursorIdx >= 0 && cursorIdx < n) {
      const xc = X(timeAtIndex(ts, cursorIdx));
      ctx.strokeStyle = "rgba(0,0,0,0.5)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xc, pad); ctx.lineTo(xc, H - pad); ctx.stroke();
    }
  }
export { buildLiveDims, closeChartModal, drawLiveCharts, drawSeriesChart, liveDimsAll, onScrubInput, openChartModal, resetLiveSeries, updateScrub };
