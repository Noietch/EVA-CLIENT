// eval.js: evaluation runs/selectors/scoring (eval); results browser (results);
// trial-detail popup (trialpop).
import { $, RT_COLORS, S, apiGet, apiPost, setCommandMetadata } from "./core.js";
import { drawSeriesChart } from "./charts.js";
import { renderEvalGripper, setPanel, syncChip } from "./run.js";
import { renderSceneGrid } from "./scene_map.js";

// ===== eval =====

let EVAL_TASK = null;

let EVAL_TRIAL = null;

let EVAL_CLIP = null;

let EVAL_RECORDS = {};

let EVAL_LAST_PHASE = "idle";

let EVAL_PENDING = null;

let EVAL_PENDING_TICKS = 0;

let EVAL_ACTIVE_SLOT = null;

let EVAL_SWITCHING = false;

let EVAL_ELAPSED_MS = 0;

let EVAL_SETUP_STAGE = "";

let EVAL_SHOW_UNSCORABLE_HINT = false;

let EVAL_SETUP_FIRED = false;

let EVAL_PREV_SETUP_DONE = false;

let EVAL_RESULTS_REQUEST = 0;

let EVAL_RESULTS_LOADING = false;

let EVAL_SCORE_PENDING = false;

let EVAL_OBSERVED_CLIP = null;

let EVAL_RESTORE = null;

const EVAL_SCENE_MAP = {
  taskSet: null, plan: null, sceneId: "", loading: false, error: "", requestId: 0,
};

const EVAL_TASK_CATALOG = {plan: null, authored: null, taskSet: null, tasks: [], byKey: new Map()};

const EVAL_PHASE_LABEL = { idle: "not ready", ready: "ready", starting: "starting", running: "running",
    stopping: "stopping", resetting: "resetting", warming: "warming up" };

function evalCfg() { return (S.CFG && S.CFG.eval) || {}; }

function evalEnabled() { return !!evalCfg().tasks; }

function evalCurPhase() { return EVAL_PENDING || EVAL_LAST_PHASE || "idle"; }

function evalBusy() { return ["starting","running","stopping","resetting","warming"].includes(evalCurPhase()); }

function evalTasks() {
    const authored = evalCfg().tasks || [];
    const plan = EVAL_SCENE_MAP.plan;
    const catalog = EVAL_TASK_CATALOG;
    if (catalog.plan === plan && catalog.authored === authored
        && catalog.taskSet === EVAL_SCENE_MAP.taskSet) return catalog.tasks;
    const tasks = (plan?.tasks || []).map((task) => {
      const configured = authored.find((entry) => (
        (!entry.task_set || entry.task_set === EVAL_SCENE_MAP.taskSet)
        && (entry.task_id ? entry.task_id === task.task_id : entry.prompt_en === task.prompt_en)
      ));
      return {...task, ...configured, task_id: task.task_id, task_set: EVAL_SCENE_MAP.taskSet,
        milestones: configured?.milestones || [{id: "success", label: "Task completed"}]};
    });
    const selected = plan ? tasks.concat(authored.filter((entry) => !entry.task_set
      && !tasks.some((task) => entry.task_id ? task.task_id === entry.task_id
        : task.prompt_en === entry.prompt_en))) : authored;
    Object.assign(catalog, {plan, authored, taskSet: EVAL_SCENE_MAP.taskSet, tasks: selected,
      byKey: new Map(selected.map((task) => [evalTaskKey(task), task]))});
    return selected;
  }

function evalTaskKey(task) { return JSON.stringify([task.task_set || "", task.task_id || task.prompt_en]); }

function evalTask(key = EVAL_TASK) { evalTasks(); return EVAL_TASK_CATALOG.byKey.get(key); }

function evalCellKey(task, sceneId, trial) {
    return JSON.stringify([task.task_set || "", task.task_id || task.prompt_en || task.prompt,
      sceneId || "", trial]);
  }

function evalSceneTaskSet() {
    return EVAL_SCENE_MAP.taskSet ?? evalTask()?.task_set ?? evalCfg().scene_plan_sets?.[0] ?? "";
  }

function selectEvalTask(key, sceneId = "") {
    EVAL_TASK = key;
    EVAL_SCENE_MAP.sceneId = sceneId;
    renderEvalSceneMap();
    EVAL_TRIAL = firstUntestedTrial(key);
    EVAL_CLIP = null;
    EVAL_SETUP_FIRED = false;
    const task = evalTask();
    if (task && !evalBusy()) apiPost("/api/select_task", {task: task.prompt_en});
  }

function restoreEvalAttempt() {
    if (!EVAL_RESTORE) return false;
    const {cell, clip} = EVAL_RESTORE;
    if (cell.scene_id && !EVAL_SCENE_MAP.plan) return false;
    const key = evalTaskKey({task_set: cell.task_set || "", task_id: cell.task_id || "", prompt_en: cell.prompt});
    if (!evalTask(key)) return false;
    EVAL_TASK = key;
    EVAL_TRIAL = cell.trial;
    EVAL_CLIP = clip;
    EVAL_SCENE_MAP.sceneId = cell.scene_id || "";
    EVAL_SETUP_FIRED = true;
    EVAL_RESTORE = null;
    renderEvalSceneMap();
    return true;
  }

async function loadEvalScenePlan(force = false) {
    if (force && evalBusy() && !EVAL_RESTORE) return;
    const taskSet = evalSceneTaskSet();
    if (!force && EVAL_SCENE_MAP.taskSet === taskSet) {
      renderEvalSceneMap();
      return;
    }
    const requestId = ++EVAL_SCENE_MAP.requestId;
    Object.assign(EVAL_SCENE_MAP, {taskSet, loading: true, error: ""});
    renderEvalSceneMap();
    try {
      const query = new URLSearchParams({scope: "eval"});
      if (taskSet) query.set("set", taskSet);
      const plan = await apiGet(`/api/scene_plan?${query}`);
      if (requestId !== EVAL_SCENE_MAP.requestId) return;
      EVAL_SCENE_MAP.plan = plan;
      const tasks = evalTasks();
      if (EVAL_RESTORE) {
        if (!restoreEvalAttempt()) EVAL_SCENE_MAP.error = "Active evaluation task is not in this task set";
      } else if (!evalTask()) selectEvalTask(tasks.length ? evalTaskKey(tasks[0]) : null);
    } catch (error) {
      if (requestId !== EVAL_SCENE_MAP.requestId) return;
      EVAL_SCENE_MAP.error = error.message;
    }
    if (requestId !== EVAL_SCENE_MAP.requestId) return;
    EVAL_SCENE_MAP.loading = false;
    renderEvalSceneMap();
    renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
  }

function renderEvalSceneMap() {
    const host = $("eval-scene-grid");
    if (!host) return;
    const state = EVAL_SCENE_MAP;
    const selectedTask = evalTask();
    const task = (state.plan?.tasks || []).find((entry) => selectedTask?.task_id
      ? entry.task_id === selectedTask.task_id
      : (entry.runtime_prompt || entry.prompt_en) === selectedTask?.prompt_en);
    const scenes = (task?.scene_ids || []).map((id) => (
      (state.plan?.scenes || []).find((scene) => scene.scene_id === id)
    )).filter(Boolean);
    if (!scenes.some((scene) => scene.scene_id === state.sceneId)) {
      state.sceneId = scenes[0]?.scene_id || "";
    }
    const selector = $("eval-scene-list");
    const sceneKey = scenes.map((scene) => scene.scene_id).join("|");
    if (selector.dataset.scenes !== sceneKey) {
      selector.dataset.scenes = sceneKey;
      selector.replaceChildren(...scenes.map((scene) => {
        const option = document.createElement("option");
        option.value = scene.scene_id;
        option.textContent = scene.scene_id;
        return option;
      }));
    }
    selector.value = state.sceneId;
    selector.disabled = evalBusy() || scenes.length < 2;
    selector.onchange = () => {
      if (evalBusy()) { selector.value = state.sceneId; return; }
      state.sceneId = selector.value;
      EVAL_CLIP = null;
      EVAL_TRIAL = firstUntestedTrial(EVAL_TASK);
      renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
    };
    const sets = evalCfg().scene_plan_sets || [];
    const setSelector = $("eval-scene-set-list");
    const setsKey = JSON.stringify(sets);
    if (setSelector.dataset.sets !== setsKey) {
      setSelector.dataset.sets = setsKey;
      setSelector.replaceChildren(...sets.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        return option;
      }));
    }
    setSelector.hidden = sets.length < 2;
    setSelector.value = state.taskSet || "";
    setSelector.disabled = evalBusy();
    setSelector.onchange = () => {
      if (evalBusy()) { setSelector.value = state.taskSet || ""; return; }
      state.taskSet = setSelector.value;
      state.plan = null;
      state.sceneId = "";
      EVAL_TASK = null;
      EVAL_CLIP = null;
      loadEvalScenePlan(true);
    };
    const scene = scenes.find((entry) => entry.scene_id === state.sceneId);
    $("eval-scene-task").textContent = task?.prompt_zh || selectedTask?.prompt_en || "—";
    $("eval-scene-status").textContent = state.loading ? "Loading scene map…"
      : state.error ? state.error
      : !task ? "No scene map linked to this evaluation task"
      : !scene ? "No scene layout found for this task" : "";
    $("eval-scene-refresh").disabled = state.loading || evalBusy();
    $("eval-scene-refresh").onclick = () => loadEvalScenePlan(true);
    host.hidden = !scene;
    if (scene) renderSceneGrid(host, state.plan, scene, task);
    else {
      host.replaceChildren();
      delete host.dataset.layoutKey;
    }
  }

function evalHasTrial() { return !!evalTask() && EVAL_TRIAL != null
    && !EVAL_SCENE_MAP.loading && !EVAL_RESTORE; }

function evalSetupReady() {
    return evalHasTrial() && evalCurPhase() === "ready";
  }

function evalSetupErrored() {
    const s = S.STATUS || {};
    return evalHasTrial() && !s.is_setup_done && !!s.last_error && !evalBusy();
  }

function renderEvalSetupState() {
    const msg = $("eval-auto-setup-msg");
    const setup = $("be-setup");
    const done = !!((S.STATUS || {}).is_setup_done) && evalCurPhase() !== "warming";
    const errored = evalSetupErrored();
    const stage = EVAL_SETUP_STAGE || (evalCurPhase() === "warming" ? "preparing robot" : "");
    if (msg) {
      if (!evalHasTrial()) msg.textContent = "AWAITING TRIAL…";
      else if (errored) msg.textContent = "SETUP FAILED";
      else if (stage && !done) msg.textContent = "AUTO · " + stage.toUpperCase();
      else if (done) msg.textContent = "ROBOT READY";
      else msg.textContent = "AUTO · PREPARING ROBOT…";
    }
    if (setup) {
      setup.classList.toggle("busy", evalCurPhase() === "warming");
      setup.textContent = evalCurPhase() === "warming"
        ? (EVAL_SETUP_STAGE ? EVAL_SETUP_STAGE.toUpperCase() : "SETTING UP…")
        : (done ? "SETUP AGAIN ⚙" : "SETUP ⚙");
      setup.disabled = !evalSetupReady();
    }
  }

function maybeAutoEvalSetup() {
    const done = !!((S.STATUS || {}).is_setup_done) && evalCurPhase() !== "warming";
    if (EVAL_PREV_SETUP_DONE && !done) EVAL_SETUP_FIRED = false;
    EVAL_PREV_SETUP_DONE = done;
    renderEvalSetupState();
    if (done || EVAL_SETUP_FIRED || evalSetupErrored() || !evalSetupReady()) return;
    EVAL_SETUP_FIRED = true;
    evalSetup(false);
  }

function renderEvalSelectors() {
    EVAL_SETUP_FIRED = false;
    $("eval-run-id").textContent = "run " + ((S.CFG && S.CFG.run_id) || "—");
    renderEvalModelList();
    // default to the first prompt's first untested trial
    const prompts = evalTasks();
    if (prompts.length && !evalBusy() && !EVAL_RESTORE && (EVAL_TASK == null || !evalTask())) {
      selectEvalTask(evalTaskKey(prompts[0]));
    }
    renderEvalGripper();
    renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
  }

// MODEL picker: one option per checkpoint, showing the real ckpt name. The first model is
// default-selected on entry (the eval bootstrap already connects it, so just highlight).
function renderEvalModelList() {
    const cks = evalCfg().checkpoints || [];
    const panel = $("eval-panel-model"), list = $("eval-model-list");
    if (!cks.length) { panel.style.display = "none"; return; }
    panel.style.display = "";
    if (EVAL_ACTIVE_SLOT == null) EVAL_ACTIVE_SLOT = Number(cks[0].slot);
    list.innerHTML = "";
    if (list.tagName === "SELECT") {
      setCommandMetadata(list, "web:switch_ckpt:{slot}", true);
      cks.forEach((c) => {
        const slot = Number(c.slot);
        const opt = document.createElement("option");
        opt.value = String(slot);
        opt.dataset.label = c.name || c.label;
        opt.textContent = String.fromCharCode(65 + slot) + " " + (c.name || c.label);
        list.appendChild(opt);
      });
      list.onchange = () => {
        const slot = Number(list.value);
        if (evalBusy() || EVAL_SWITCHING || EVAL_SCORE_PENDING) {
          list.value = String(EVAL_ACTIVE_SLOT ?? Number(cks[0].slot));
          syncChip("eval-model-list");
          return;
        }
        evalSwitchCkpt(slot);
      };
      list.value = String(EVAL_ACTIVE_SLOT);
      syncChip("eval-model-list");
      return;
    }
    cks.forEach((c) => {
      const slot = Number(c.slot);
      const seg = document.createElement("button");
      seg.className = "seg" + (slot === EVAL_ACTIVE_SLOT ? " active" : "");
      seg.dataset.slot = String(slot);
      setCommandMetadata(seg, "web:switch_ckpt:{slot}", true);
      seg.innerHTML = '<span class="mk">' + String.fromCharCode(65 + slot) + '</span><span>'
        + (c.name || c.label) + '</span>';
      seg.onclick = () => evalSwitchCkpt(slot);
      list.appendChild(seg);
    });
  }

function evalSwitchCkpt(slot) {
    slot = Number(slot);
    if (slot === EVAL_ACTIVE_SLOT || EVAL_SWITCHING || EVAL_SCORE_PENDING) return;
    if (["running","starting","stopping"].includes(evalCurPhase())) return;
    EVAL_SWITCHING = true;
    EVAL_RESULTS_REQUEST++;
    EVAL_RESULTS_LOADING = false;
    EVAL_CLIP = null;
    EVAL_SETUP_FIRED = false;
    setEvalActivePill(slot);
    const list = $("eval-model-list");
    if (list && list.tagName === "SELECT") list.disabled = true;
    else document.querySelectorAll("#eval-model-list .seg").forEach((b) => { b.classList.add("locked"); });
    // Clear the current model's records immediately for instant feedback. The ckpt swap +
    // logger rebuild are async on the backend, so the NEW model's results are reloaded by
    // applyEvalStatus once the status poll confirms the active slot actually changed —
    // reloading here would race and read the old model's dataset.
    EVAL_RECORDS = {};
    renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
    apiPost("/api/select_ckpt", { slot })
      .catch(() => { EVAL_SWITCHING = false; setEvalActivePill(EVAL_ACTIVE_SLOT ?? 0); });
  }

function setEvalActivePill(slot) {
    const list = $("eval-model-list");
    if (list && list.tagName === "SELECT") {
      list.value = String(slot);
      syncChip("eval-model-list");
      return;
    }
    document.querySelectorAll("#eval-model-list .seg").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.slot) === slot);
    });
    syncChip("eval-model-list");
  }

function activateEvalCell(taskKey, trial) {
    if (evalBusy()) return;
    const rec = EVAL_RECORDS[evalRecKey(taskKey, trial)];
    if (trial > firstUntestedTrial(taskKey) && !(rec && rec.score != null)) return;
    if (taskKey !== EVAL_TASK) selectEvalTask(taskKey);
    if (EVAL_TRIAL !== trial) EVAL_CLIP = null;
    EVAL_TRIAL = trial;
    EVAL_SHOW_UNSCORABLE_HINT = true;
    if (trial === firstUntestedTrial(taskKey)) EVAL_SETUP_FIRED = false;
    renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
  }

function evalRecKey(taskKey, trial) {
    const task = evalTask(taskKey);
    if (!task) return "";
    const sceneId = taskKey === EVAL_TASK ? EVAL_SCENE_MAP.sceneId : task.scene_ids?.[0] || "";
    return evalCellKey(task, sceneId, trial);
  }

function promptMilestones(taskKey) {
    const p = evalTask(taskKey);
    return (p && p.milestones) || [];
  }

function firstUntestedTrial(taskKey) {
    const n = evalCfg().trials_per_prompt || 5;
    for (let t = 1; t <= n; t++) { const r = EVAL_RECORDS[evalRecKey(taskKey, t)]; if (!r || r.score == null) return t; }
    return 1;
  }

// First task (in config order) that still has any unscored trial; falls back to the first
// task when every trial is scored. Drives resume so re-entering EVAL lands on real work.
function firstUntestedTaskPrompt() {
    const tasks = evalTasks();
    const n = evalCfg().trials_per_prompt || 5;
    for (const task of tasks) {
      for (const sceneId of task.scene_ids?.length ? task.scene_ids : [""]) {
        for (let trial = 1; trial <= n; trial++) {
          const record = EVAL_RECORDS[evalCellKey(task, sceneId, trial)];
          if (!record || record.score == null) return {key: evalTaskKey(task), sceneId};
        }
      }
    }
    return tasks.length ? {key: evalTaskKey(tasks[0]), sceneId: tasks[0].scene_ids?.[0] || ""} : null;
  }

// Re-entering EVAL resumes at the first untested task+trial, using the on-disk scores that
// loadEvalResults just populated. Never yanks the selection mid-run.
function evalResumeOnEnter() {
    if (evalBusy()) return;
    const p = firstUntestedTaskPrompt();
    if (!p) return;
    selectEvalTask(p.key, p.sceneId);
    EVAL_SHOW_UNSCORABLE_HINT = false;
    EVAL_SETUP_FIRED = false;
    renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
  }

function renderEvalPromptList() {
    const list = $("eval-prompt-list"); if (!list) return;
    const prompts = evalTasks();
    const n = evalCfg().trials_per_prompt || 5;
    list.innerHTML = "";
    if (list.tagName === "SELECT") {
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.disabled = true;
      placeholder.textContent = "SELECT TASK";
      list.appendChild(placeholder);
      prompts.forEach((p, pi) => {
        let sc = 0, mx = 0, te = 0;
        for (let t = 1; t <= n; t++) { const r = EVAL_RECORDS[evalRecKey(evalTaskKey(p), t)]; if (r && r.score != null) { sc += r.score; mx += r.max_score; te++; } }
        const rate = te ? sc + "/" + mx : "—";
        const label = (p.task_id ? p.task_id + " · " : "") + (p.prompt_zh || p.prompt_en);
        const opt = document.createElement("option");
        opt.value = evalTaskKey(p);
        opt.dataset.label = label;
        opt.textContent = String(pi + 1).padStart(2, "0") + " " + label + " · " + rate;
        list.appendChild(opt);
      });
      list.onchange = () => {
        if (evalBusy()) { list.value = EVAL_TASK || ""; syncChip("eval-prompt-list"); return; }
        const taskKey = list.value;
        if (!taskKey) return;
        activateEvalCell(taskKey, firstUntestedTrial(taskKey));
      };
      list.value = EVAL_TASK || "";
      syncChip("eval-prompt-list");
      return;
    }
    prompts.forEach((p, pi) => {
      let sc = 0, mx = 0, te = 0;
      for (let t = 1; t <= n; t++) { const r = EVAL_RECORDS[evalRecKey(evalTaskKey(p), t)]; if (r && r.score != null) { sc += r.score; mx += r.max_score; te++; } }
      const rate = te ? sc + "/" + mx : "—";
      const seg = document.createElement("div");
      seg.className = "seg" + (evalTaskKey(p) === EVAL_TASK ? " active" : "");
      seg.innerHTML = '<span class="mk">' + String(pi + 1).padStart(2, "0") + '</span><span>'
        + (p.task_id ? p.task_id + ' · ' : '') + (p.prompt_zh || p.prompt_en) + '</span><span class="epb-rate">' + rate + '</span>';
      seg.onclick = () => activateEvalCell(evalTaskKey(p), firstUntestedTrial(evalTaskKey(p)));
      list.appendChild(seg);
    });
    syncChip("eval-prompt-list");
  }

function renderEvalTrialRow() {
    const row = $("eval-trial-row"); if (!row) return;
    const n = evalCfg().trials_per_prompt || 5;
    row.innerHTML = "";
    if (!EVAL_TASK) return;
    const ms = promptMilestones(EVAL_TASK);
    const firstOpen = firstUntestedTrial(EVAL_TASK);
    for (let t = 1; t <= n; t++) {
      const rec = EVAL_RECORDS[evalRecKey(EVAL_TASK, t)];
      const scored = rec && rec.score != null;
      const locked = !scored && t > firstOpen;
      const cell = document.createElement("div");
      cell.className = "eval-trial" + (scored ? " scored" : "") + (EVAL_TRIAL === t ? " active" : "") + (locked ? " locked" : "");
      if (scored) { const m2 = rec.max_score || ms.length || 1; if (rec.score === 0) cell.classList.add("zero"); else cell.style.setProperty("--score-pct", (100 * rec.score / m2) + "%"); }
      cell.innerHTML = "<span>" + (scored ? rec.score + "/" + (rec.max_score || ms.length) : t) + "</span>";
      if (!locked) cell.onclick = () => activateEvalCell(EVAL_TASK, t);
      row.appendChild(cell);
    }
  }

function renderEvalConsole() {
    loadEvalScenePlan();
    if (!EVAL_TASK) {
      $("ctx-trial").textContent = "—/—";
      $("ms-nodes").innerHTML = ""; updateMsFill(); $("eval-note").value = "";
      applyEvalPhase(); return;
    }
    const n = evalCfg().trials_per_prompt || 5;
    $("ctx-trial").textContent = (EVAL_TRIAL || 1) + "/" + n;
    const rec = EVAL_RECORDS[evalRecKey(EVAL_TASK, EVAL_TRIAL)] || {};
    const checks = rec.milestones || {};
    const ms = promptMilestones(EVAL_TASK);
    let level = 0; for (let i = 0; i < ms.length; i++) { if (checks[ms[i].id]) level = i + 1; else break; }
    const nodes = $("ms-nodes"); nodes.innerHTML = "";
    ms.forEach((m, i) => {
      const on = i < level;
      const node = document.createElement("div");
      node.className = "ms-node" + (on ? " done" : "");
      node.dataset.idx = i;
      node.innerHTML = '<div class="ms-dot">' + (on ? "✓" : (i + 1)) + '</div><div class="ms-label">' + m.label + '</div>';
      node.onclick = () => setMilestoneLevel(i);
      nodes.appendChild(node);
    });
    updateMsFill();
    $("eval-note").value = rec.note || "";
    $("eval-saved").textContent = rec.score != null ? "scored: " + rec.score : "";
    applyEvalPhase();
  }

function setMilestoneLevel(idx) {
    if (!evalTrialScorable(EVAL_TRIAL)) return;
    // Scoring is allowed during resetting (the trial already ended + its meta row is on
    // disk; the arm is just returning home). Only block while a run is actively in flight.
    if (["warming","starting","running","stopping"].includes(evalCurPhase())) return;
    const nodes = [...document.querySelectorAll("#ms-nodes .ms-node")];
    const cur = nodes.filter((nn) => nn.classList.contains("done")).length;
    const nl = (cur === idx + 1) ? idx : idx + 1;
    nodes.forEach((nn, i) => { const on = i < nl; nn.classList.toggle("done", on); nn.querySelector(".ms-dot").textContent = on ? "✓" : (i + 1); });
    updateMsFill();
  }

function updateMsFill() {
    const nodes = [...document.querySelectorAll("#ms-nodes .ms-node")];
    const total = nodes.length, done = nodes.filter((nn) => nn.classList.contains("done")).length;
    const fill = $("ms-rail-fill"), score = $("ms-score");
    if (!total) { fill.style.height = "0%"; score.textContent = "0/0"; return; }
    fill.style.height = (done === 0 ? 0 : total === 1 ? 100 : ((done - 1) / (total - 1)) * 100) + "%";
    score.textContent = done + "/" + total;
  }

function readMilestoneLevel() { return document.querySelectorAll("#ms-nodes .ms-node.done").length; }

function evalTrialScorable(t) {
    const rec = EVAL_RECORDS[evalRecKey(EVAL_TASK, t)];
    return !!(rec && rec.episode_index != null && (!EVAL_CLIP || rec.clip_id === EVAL_CLIP));
  }

function applyEvalPhase() {
    renderEvalSceneMap();
    const phase = evalCurPhase();
    const run = $("be-run"), reset = $("be-reset"), save = $("be-submit");
    if (!run) return;
    const running = phase === "running";
    // Single RUN/STOP toggle (mirrors DEBUG): clear the optimistic busy flag once the
    // polled phase reaches the click's target, so a stale poll can't double-fire.
    if ((S.evalRunToggleBusy === "start" && running)
        || (S.evalRunToggleBusy === "stop" && ["stopping","resetting","ready","idle"].includes(phase))) {
      S.evalRunToggleBusy = null;
    }
    const busy = S.evalRunToggleBusy !== null;
    const label = run.querySelector(".rec-label");
    run.classList.remove("busy");
    run.classList.toggle("recording", running);
    run.classList.toggle("primary", !running);
    reset.classList.remove("attention"); reset.textContent = "RESET ⟲"; reset.disabled = true;
    const hasPrompt = !!EVAL_TASK;
    const hasTrial = evalHasTrial();
    const setupDone = !!((S.STATUS || {}).is_setup_done) && phase !== "warming";
    const setupErrored = evalSetupErrored();
    if (running) label.textContent = "STOP ■";
    else if (phase === "starting") { label.textContent = "STARTING…"; run.classList.add("busy"); }
    else if (phase === "stopping") label.textContent = "STOPPING…";
    else if (phase === "resetting") { label.textContent = "RUN ▶"; reset.classList.add("attention"); reset.textContent = "RESETTING…"; }
    else { label.textContent = "RUN ▶"; reset.disabled = !(setupDone && phase === "ready"); }
    const clickable = running || (phase === "ready" && hasTrial && setupDone
      && !EVAL_SWITCHING && !EVAL_SCORE_PENDING);
    run.disabled = busy || !clickable;
    renderEvalSetupState();
    const lockCkpt = EVAL_SWITCHING || EVAL_SCORE_PENDING || ["warming","starting","running","stopping","resetting"].includes(phase);
    const modelList = $("eval-model-list");
    if (modelList && modelList.tagName === "SELECT") modelList.disabled = lockCkpt;
    else document.querySelectorAll("#eval-model-list .seg").forEach((b) => { b.classList.toggle("locked", lockCkpt); });
    const scorable = evalTrialScorable(EVAL_TRIAL);
    const scoreLocked = EVAL_SCORE_PENDING || EVAL_SWITCHING || !scorable || ["warming","starting","running","stopping"].includes(phase);
    save.disabled = scoreLocked;
    $("eval-note").disabled = scoreLocked;
    document.querySelectorAll("#ms-nodes .ms-node").forEach((nn) => nn.classList.toggle("locked", scoreLocked));
    const modelDone = !((evalCfg().checkpoints || []).length) || (EVAL_ACTIVE_SLOT != null && !EVAL_SWITCHING);
    setPanel("eval-panel-model", modelDone ? "done" : "active");
    setPanel("eval-panel-task", modelDone ? (hasPrompt ? "done" : "active") : "pending");
    setPanel("eval-panel-trial", hasPrompt ? (hasTrial ? "done" : "active") : "pending");
    setPanel("eval-panel-setup", setupDone ? "done" : (setupErrored ? "error" : (hasTrial ? "active" : "pending")));
    setPanel("eval-panel-run", setupDone && hasTrial ? (scorable ? "done" : "active") : "pending");
    setPanel("eval-panel-gripper", setupDone && hasTrial ? "active" : "pending");
    setPanel("eval-panel-score", scorable ? "active" : "pending");
    maybeAutoEvalSetup();
    const hint = $("eval-gate-hint");
    if (hint) hint.textContent = (EVAL_SHOW_UNSCORABLE_HINT && EVAL_TASK && !scorable) ? "Run this trial before scoring" : "";
  }

function setEvalPending(p) { EVAL_PENDING = p; EVAL_PENDING_TICKS = 0; applyEvalPhase(); }

function evalRunToggle() {
    if (S.evalRunToggleBusy !== null) return;
    const live = evalCurPhase() === "running";
    S.evalRunToggleBusy = live ? "stop" : "start";
    if (live) setEvalPending("stopping");
    else applyEvalPhase();
    setTimeout(() => { if (S.evalRunToggleBusy === "start") { S.evalRunToggleBusy = null; applyEvalPhase(); } }, 1500);
    return live
      ? apiPost("/api/eval_stop").catch(() => { EVAL_PENDING = null; S.evalRunToggleBusy = null; applyEvalPhase(); })
      : evalRun();
  }

function evalSetup(force) {
    if (!evalHasTrial()) { $("be-err").textContent = "select a trial first"; return; }
    if (evalCurPhase() !== "ready") return;
    if (!force && ((S.STATUS || {}).is_setup_done)) return;
    EVAL_SETUP_FIRED = true;
    setEvalPending("warming");
    apiPost("/api/warmup").catch(() => { EVAL_PENDING = null; applyEvalPhase(); });
  }

function evalRun() {
    if (!evalHasTrial()) { $("be-err").textContent = "select a trial first"; return; }
    if (evalCurPhase() !== "ready" || EVAL_SWITCHING || EVAL_SCORE_PENDING) return;
    if (!((S.STATUS || {}).is_setup_done)) { evalSetup(true); return; }
    const t = EVAL_TRIAL || firstUntestedTrial(EVAL_TASK);
    EVAL_TRIAL = t;
    EVAL_SHOW_UNSCORABLE_HINT = false;
    EVAL_CLIP = "eval-" + Date.now() + "-" + Math.floor(performance.now());
    EVAL_OBSERVED_CLIP = EVAL_CLIP;
    setEvalPending("starting");
    const task = evalTask();
    apiPost("/api/eval_start", {clip_id: EVAL_CLIP, prompt: task.prompt_en, trial: t,
      task_set: task.task_set || "", task_id: task.task_id || "", scene_id: EVAL_SCENE_MAP.sceneId})
      .catch(() => { EVAL_PENDING = null; EVAL_CLIP = null; applyEvalPhase(); });
    renderEvalTrialRow();
  }

function evalReset() {
    if (!["ready","idle","resetting"].includes(evalCurPhase())) return;
    setEvalPending("resetting");
    apiPost("/api/eval_reset").catch(() => { EVAL_PENDING = null; applyEvalPhase(); });
  }

async function submitEvalScore() {
    if (EVAL_SCORE_PENDING || EVAL_SWITCHING || !evalHasTrial()
        || ["warming", "starting", "running", "stopping"].includes(evalCurPhase())) return;
    if (!evalTrialScorable(EVAL_TRIAL)) {
      EVAL_SHOW_UNSCORABLE_HINT = true;
      applyEvalPhase();
      return;
    }
    const task = evalTask();
    const trial = EVAL_TRIAL;
    const key = EVAL_TASK;
    const sceneId = EVAL_SCENE_MAP.sceneId;
    const rec = EVAL_RECORDS[evalRecKey(key, trial)];
    const model = S.EVAL_MODEL_NAME;
    const ms = task.milestones || [];
    const level = readMilestoneLevel();
    const milestones = {};
    ms.forEach((milestone, index) => { milestones[milestone.id] = index < level; });
    EVAL_SCORE_PENDING = true;
    applyEvalPhase();
    try {
      const response = await apiPost("/api/score", {
        clip_id: rec.clip_id, prompt: task.prompt_en, trial, model_name: model,
        task_set: task.task_set || "", task_id: task.task_id || "", scene_id: sceneId,
        score: level, max_score: ms.length, milestones, note: $("eval-note").value,
      });
      if (response?.ok === false) { $("eval-saved").textContent = response.error || "score rejected"; return; }
      await loadEvalResults();
      if (EVAL_TASK !== key || EVAL_TRIAL !== trial || EVAL_SCENE_MAP.sceneId !== sceneId
          || S.EVAL_MODEL_NAME !== model || EVAL_SWITCHING
          || ["warming", "starting", "running", "stopping"].includes(evalCurPhase())
          || EVAL_RECORDS[evalRecKey(key, trial)]?.clip_id !== rec.clip_id) return;
      EVAL_SHOW_UNSCORABLE_HINT = false;
      const n = evalCfg().trials_per_prompt || 5;
      EVAL_TRIAL = Math.min(trial + 1, n);
      EVAL_CLIP = null;
      renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole();
      if (trial < n) evalSetup(true);
    } catch (error) {
      $("eval-saved").textContent = error.message;
    } finally {
      EVAL_SCORE_PENDING = false;
      applyEvalPhase();
    }
  }

async function loadEvalResults() {
    const requestId = ++EVAL_RESULTS_REQUEST;
    const slot = EVAL_ACTIVE_SLOT;
    EVAL_RESULTS_LOADING = true;
    try {
      const data = await apiGet("/api/results");
      if (requestId !== EVAL_RESULTS_REQUEST || slot !== EVAL_ACTIVE_SLOT || EVAL_SWITCHING) return;
      S.EVAL_MODEL_NAME = data.model_name || "";
      EVAL_RECORDS = {};
      for (const record of data.records || []) {
        if (record.prompt == null || record.trial == null) continue;
        EVAL_RECORDS[evalCellKey(record, record.scene_id, record.trial)] = record;
      }
      if (S.ACTIVE_TAB === "eval") { renderEvalPromptList(); renderEvalTrialRow(); renderEvalConsole(); }
    } catch (error) {
      $("be-err").textContent = error.message;
    } finally {
      if (requestId === EVAL_RESULTS_REQUEST) EVAL_RESULTS_LOADING = false;
    }
  }

function applyEvalStatus(s) {
    if (!evalEnabled()) return;
    EVAL_SETUP_STAGE = s.setup_stage || "";
    let phase = s.web_phase || "idle";
    // Source of truth is the real session state (the telemetry bar's STATE = session_status).
    // web_phase can latch at "running" after the backend already left RUNNING (auto-stop /
    // client watchdog / a start that never truly began) — that would show a phantom run with
    // a ticking timer, no stepping, and locked scoring. If it isn't really running, collapse
    // the phase to ready so the UI matches the top status bar.
    if (phase === "running" && s.session_status !== "running") phase = "ready";
    if ($("be-phase")) $("be-phase").textContent = phase;
    // reconcile the optimistic pending phase
    if (EVAL_PENDING && (phase === EVAL_PENDING
        || (EVAL_PENDING === "starting" && phase === "running")
        || (EVAL_PENDING === "warming" && phase === "ready")
        || (EVAL_PENDING === "stopping" && ["resetting","ready","idle"].includes(phase))
        || (EVAL_PENDING === "resetting" && ["ready","idle"].includes(phase)))) {
      EVAL_PENDING = null; EVAL_PENDING_TICKS = 0;
    } else if (EVAL_PENDING === "warming" && phase === "starting") {
      // web:warmup drives the backend phase to "starting" while it homes + validates;
      // keep the optimistic "warming" latched (don't tick it out) until it settles to ready.
      EVAL_PENDING_TICKS = 0;
    } else if (EVAL_PENDING) { if (++EVAL_PENDING_TICKS >= 3) { EVAL_PENDING = null; EVAL_PENDING_TICKS = 0; } }
    // model pill sync: when the backend confirms the active slot actually changed (the
    // ckpt swap + logger rebuild are async, completing a few polls after select_ckpt),
    // reload this model's results so the trial scores refresh without a manual reload.
    if (s.ckpt_slot != null && s.ckpt_slot !== EVAL_ACTIVE_SLOT) {
      EVAL_RESULTS_REQUEST++; EVAL_RESULTS_LOADING = false; EVAL_CLIP = null;
      EVAL_ACTIVE_SLOT = s.ckpt_slot; EVAL_SWITCHING = false; setEvalActivePill(EVAL_ACTIVE_SLOT);
      EVAL_SETUP_FIRED = false;
      EVAL_RECORDS = {};
      loadEvalResults();
    }
    // elapsed timer (hosted in the top guide bar so the run time shows on the STEP rail)
    EVAL_ELAPSED_MS = s.run_elapsed_ms || 0;
    const timer = $("be-elapsed");
    timer.style.display = S.ACTIVE_TAB === "eval" ? "" : "none";
    if (phase === "running") { timer.classList.add("running"); timer.textContent = "⏱ " + (EVAL_ELAPSED_MS / 1000).toFixed(1) + "s"; }
    else { timer.classList.remove("running"); const rec = EVAL_RECORDS[evalRecKey(EVAL_TASK, EVAL_TRIAL)]; timer.textContent = "⏱ " + (((rec && rec.duration_ms) || 0) / 1000).toFixed(1) + "s"; }
    $("be-err").textContent = s.last_error || "";
    // trial finished -> reload so it becomes scorable
    const settled = phase === "ready" || phase === "idle";
    const record = EVAL_RECORDS[evalRecKey(EVAL_TASK, EVAL_TRIAL)];
    if (settled && EVAL_CLIP && record?.clip_id !== EVAL_CLIP && !EVAL_RESULTS_LOADING) loadEvalResults();
    EVAL_LAST_PHASE = phase;
    if (s.clip_id && s.cell && s.clip_id !== EVAL_OBSERVED_CLIP
        && ["starting", "running", "stopping", "resetting"].includes(phase)) {
      EVAL_OBSERVED_CLIP = s.clip_id;
      EVAL_RESTORE = {clip: s.clip_id, cell: s.cell};
      EVAL_TASK = null;
      const taskSet = s.cell.task_set || "";
      const sameSet = EVAL_SCENE_MAP.taskSet === taskSet;
      if (!sameSet) EVAL_SCENE_MAP.plan = null;
      EVAL_SCENE_MAP.taskSet = taskSet;
      if (!restoreEvalAttempt() && (!sameSet || !EVAL_SCENE_MAP.loading)) loadEvalScenePlan(true);
    }
    applyEvalPhase();
  }

// ===== results =====
// RESULT is a master-detail browser: a 3-level tree on the left (model → task → trial),
// each model/task row carrying ring gauges for score% and success%. Clicking a recorded
// trial drives the shared #trial-pop replay (REPLAY-styled) docked in the right column.

let RV = { tasks: [], models: [], trials: 5, run_id: "", results_dir: "" };

let rvOpenModel = null;

let rvOpenPrompt = null;

let rvActiveKey = null;

function rvTaskKey(task) {
    return JSON.stringify([task.task_set || "", task.task_id || task.prompt_en || task.prompt,
      task.scene_id || ""]);
  }

function rvTasks() {
    const tasks = new Map();
    for (const model of RV.models) {
      for (const record of model.records) {
        const key = rvTaskKey(record);
        const authored = RV.tasks.find((task) => task.task_id
          ? task.task_id === record.task_id && (!task.task_set || task.task_set === record.task_set)
          : task.prompt_en === record.prompt);
        const snapshotTask = record.scene_map?.tasks?.find((task) => task.task_id === record.task_id);
        tasks.set(key, {...snapshotTask, ...authored, ...record, prompt_en: record.prompt,
          milestones: authored?.milestones || [{id: "success", label: "Task completed"}]});
      }
    }
    for (const task of RV.tasks) {
      if (![...tasks.values()].some((entry) => task.task_id
        ? entry.task_id === task.task_id && (!task.task_set || task.task_set === entry.task_set)
        : entry.prompt_en === task.prompt_en)) tasks.set(rvTaskKey(task), task);
    }
    return [...tasks.values()];
  }

function rvTrialMs(record) {
    if (!record) return [];
    const task = rvTasks().find((entry) => rvTaskKey(entry) === rvTaskKey(record));
    return task?.milestones || [];
  }

function rvModel(name) { return RV.models.find((model) => model.model_name === name); }

function rvBuildMatrix(records) {
    const matrix = {};
    for (const task of rvTasks()) matrix[rvTaskKey(task)] = Array(RV.trials).fill(null);
    for (const record of records) {
      const trials = matrix[rvTaskKey(record)];
      const index = Number(record.trial) - 1;
      if (trials && index >= 0 && index < RV.trials) trials[index] = record;
    }
    return matrix;
  }

function rvStats(records) {
    let score = 0, max = 0, tested = 0, succ = 0;
    for (const r of records) {
      if (r.score == null) continue;
      score += r.score || 0; max += r.max_score || 0; tested++;
      if (r.max_score > 0 && r.score === r.max_score) succ++;
    }
    return { score, max, tested, succ };
  }

async function loadResultsAll() {
    try {
      RV = Object.assign(RV, await apiGet("/api/results_all"));
      if (RV.trials_per_prompt != null) RV.trials = RV.trials_per_prompt;
      // also refresh which model is active (for popup score editability)
      await loadEvalResults();
      renderResultTree();
    } catch (e) {
      $("rv-tree").innerHTML = '<div class="rv-empty">load failed: ' + e.message + '</div>';
    }
  }

// SVG ring gauge: a value/whole arc with the pct label centered. `kind` colors the arc
// (score = ink, success = green) and a zero value reads danger-red.
function rvRing(pct, label, kind) {
    const r = 15, c = 2 * Math.PI * r;
    const has = pct != null && pct >= 0;
    const frac = has ? Math.max(0, Math.min(1, pct / 100)) : 0;
    const off = c * (1 - frac);
    const col = !has ? "var(--rule-strong)" : pct === 0 ? "var(--danger)"
      : kind === "succ" ? "var(--ok)" : "var(--ink)";
    const txt = has ? Math.round(pct) + "%" : "—";
    return '<div class="ring"><svg viewBox="0 0 40 40">'
      + '<circle class="ring-bg" cx="20" cy="20" r="' + r + '"></circle>'
      + '<circle class="ring-fg" cx="20" cy="20" r="' + r + '" stroke="' + col + '"'
      + ' stroke-dasharray="' + c.toFixed(2) + '" stroke-dashoffset="' + off.toFixed(2) + '"></circle>'
      + '<text x="20" y="20" class="ring-t">' + txt + '</text></svg>'
      + '<span class="ring-l">' + label + '</span></div>';
  }

function rvTaskStats(arr) {
    let sc = 0, mx = 0, te = 0, su = 0;
    for (const rec of arr) {
      if (!rec || rec.score == null) continue;
      sc += rec.score; mx += rec.max_score; te++;
      if (rec.max_score > 0 && rec.score === rec.max_score) su++;
    }
    return { sc, mx, te, su, scorePct: mx ? 100 * sc / mx : -1, succPct: te ? 100 * su / te : -1 };
  }

// L1 model → L2 task → L3 trial cells. Only one model and one task expand at a time;
// expanding a model collapses any open task. Rebuilt wholesale on every toggle (cheap).
function renderResultTree() {
    const host = $("rv-tree"); if (!host) return;
    if (!RV.models.length) { host.innerHTML = '<div class="rv-empty">no recorded models yet</div>'; return; }
    host.innerHTML = "";
    RV.models.forEach((m) => {
      const matrix = rvBuildMatrix(m.records);
      const s = rvStats(Object.values(matrix).flat().filter(Boolean));
      const mOpen = rvOpenModel === m.model_name;
      const node = document.createElement("div");
      node.className = "rv-node rv-model" + (mOpen ? " open" : "");
      const head = document.createElement("div");
      head.className = "rv-node-head";
      head.innerHTML = '<span class="caret">▸</span><span class="rv-name">' + m.model_name + '</span>'
        + '<div class="rings">'
        + rvRing(s.max ? 100 * s.score / s.max : -1, "score", "score")
        + rvRing(s.tested ? 100 * s.succ / s.tested : -1, "succ", "succ")
        + '</div>';
      head.onclick = () => { rvOpenModel = mOpen ? null : m.model_name; rvOpenPrompt = null; renderResultTree(); };
      node.appendChild(head);
      if (mOpen) {
        const tasks = document.createElement("div"); tasks.className = "rv-tasks";
        rvTasks().forEach((p, pi) => {
          const arr = matrix[rvTaskKey(p)];
          const st = rvTaskStats(arr);
          const key = JSON.stringify([m.model_name, rvTaskKey(p)]);
          const tOpen = rvOpenPrompt === key;
          const t = document.createElement("div");
          t.className = "rv-node rv-task" + (tOpen ? " open" : "");
          const th = document.createElement("div");
          th.className = "rv-node-head";
          th.innerHTML = '<span class="caret">▸</span><span class="pn">P' + (pi + 1) + '</span>'
            + '<span class="rv-name pt">' + (p.task_id ? p.task_id + ' · ' : '') + (p.prompt_zh || p.prompt_en)
            + (p.scene_id ? ' · ' + p.task_set + ' / ' + p.scene_id : '') + '</span>'
            + '<div class="rings">'
            + rvRing(st.scorePct, "score", "score")
            + rvRing(st.succPct, "succ", "succ")
            + '</div>';
          th.onclick = () => { rvOpenPrompt = tOpen ? null : key; renderResultTree(); };
          t.appendChild(th);
          if (tOpen) t.appendChild(rvTrialGrid(m.model_name, p, arr));
          tasks.appendChild(t);
        });
        node.appendChild(tasks);
      }
      host.appendChild(node);
    });
  }

function rvTrialGrid(modelName, p, arr) {
    const ms = (p.milestones) || [];
    const grid = document.createElement("div"); grid.className = "rv-trials eval-trial-row";
    for (let i = 0; i < RV.trials; i++) {
      const rec = arr[i];
      const key = JSON.stringify([modelName, rvTaskKey(p), i]);
      const cell = document.createElement("div"); cell.className = "eval-trial";
      if (rec && rec.score != null) {
        cell.classList.add("scored");
        const m2 = rec.max_score || ms.length || 1;
        if (rec.score === 0) cell.classList.add("zero");
        else cell.style.setProperty("--score-pct", (100 * rec.score / m2) + "%");
        cell.innerHTML = "<span>" + rec.score + "/" + m2 + "</span>";
        if (rec.episode_index != null) {
          if (rvActiveKey === key) cell.classList.add("active");
          cell.onclick = () => { rvActiveKey = key; renderResultTree(); trialPopOpen(rec, modelName); };
        }
      } else {
        cell.innerHTML = "<span>trial " + (i + 1) + "</span>";
        cell.style.opacity = "0.5";
        if (rec?.episode_index != null) cell.onclick = () => {
          rvActiveKey = key; renderResultTree(); trialPopOpen(rec, modelName);
        };
      }
      grid.appendChild(cell);
    }
    return grid;
  }
// ===== trialpop =====

let TP = null;

let TP_REC = null;

let TP_MODEL = "";

let TP_LOAD_SEQ = 0;

function tpBuildPlayTimeline(timestamps, nFrames) {
    const n = Math.max(0, Number(nFrames) || timestamps.length);
    if (!n) return [];
    const diffs = [];
    for (let i = 1; i < n; i++) {
      const dt = Number(timestamps[i]) - Number(timestamps[i - 1]);
      if (Number.isFinite(dt) && dt > 1e-6) diffs.push(dt);
    }
    diffs.sort((a, b) => a - b);
    const fallbackDt = diffs.length ? diffs[Math.floor(diffs.length / 2)] : 1 / 30;
    const out = new Array(n);
    out[0] = 0;
    for (let i = 1; i < n; i++) {
      const dt = Number(timestamps[i]) - Number(timestamps[i - 1]);
      out[i] = out[i - 1] + (Number.isFinite(dt) && dt > 1e-6 ? dt : fallbackDt);
    }
    return out;
  }

function tpTimeAtFrame(frame) {
    if (!TP || !TP.playTime.length) return 0;
    const clamped = Math.max(0, Math.min(Number(frame) || 0, TP.playTime.length - 1));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(i0 + 1, TP.playTime.length - 1);
    const a = clamped - i0;
    const t0 = Number(TP.playTime[i0]) || 0;
    const t1 = Number(TP.playTime[i1]) || t0;
    return t0 + (t1 - t0) * a;
  }

function tpFrameAtTime(timeSec) {
    if (!TP || TP.n <= 1 || !TP.playTime.length) return 0;
    const t = Math.max(0, Number(timeSec) || 0);
    if (t <= (TP.playTime[0] || 0)) return 0;
    if (t >= (TP.playTime[TP.n - 1] || 0)) return TP.n - 1;
    let lo = 0, hi = TP.n - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if ((TP.playTime[mid] || 0) <= t) lo = mid;
      else hi = mid;
    }
    const t0 = Number(TP.playTime[lo]) || 0;
    const t1 = Number(TP.playTime[lo + 1]) || t0;
    if (t1 <= t0) return lo;
    return lo + (t - t0) / (t1 - t0);
  }

// #trial-pop docks into #rv-detail-replay (RESULT detail). When detached it floats over
// .workspace as a fallback popup.
function tpResultDock() { const c = document.getElementById("rv-detail-replay"); return !!(c && c.contains($("trial-pop"))); }

function trialPopOpen(rec, modelName) {
    TP_REC = rec; TP_MODEL = modelName || "";
    tpRenderSteps();
    const grid = $("tp-scene-grid");
    const scene = rec.scene_map?.scenes?.find((entry) => entry.scene_id === rec.scene_id);
    $("tp-scene-label").textContent = rec.scene_id ? `${rec.task_set} / ${rec.task_id} / ${rec.scene_id}` : "";
    grid.hidden = !scene;
    if (scene) renderSceneGrid(grid, rec.scene_map, scene,
      rec.scene_map.tasks.find((task) => task.task_id === rec.task_id));
    else { grid.replaceChildren(); delete grid.dataset.layoutKey; }
    const note = $("tp-note");
    if (note) { note.textContent = rec.note || "(no note)"; note.classList.toggle("empty", !rec.note); }
    if (tpResultDock()) { const e = $("rv-replay-empty"); if (e) e.style.display = "none"; }
    tpSetup(rec.episode_index, rec.prompt, TP_MODEL);
    $("trial-pop").classList.add("open");
    if (window.ReplayScene) ReplayScene.resizeSoon();
  }

function trialPopClose() {
    TP_LOAD_SEQ += 1;
    tpStop();
    $("trial-pop").classList.remove("open");
    $("tp-cam-strip").innerHTML = "";
    const e = $("rv-replay-empty"); if (e) e.style.display = tpResultDock() ? "flex" : "";
  }

function tpRenderSteps() {
    const ms = rvTrialMs(TP_REC);
    const host = $("tp-steps"); if (!host) return;
    const score = (TP_REC && TP_REC.score != null)
      ? TP_REC.score
      : ms.filter((m) => ((TP_REC && TP_REC.milestones) || {})[m.id]).length;
    host.innerHTML = "";
    ms.forEach((m, i) => {
      const el = document.createElement("div");
      el.className = "tp-step" + (score >= i + 1 ? " done" : "");
      el.innerHTML = `<span class="si">${i + 1}</span><span class="sl">${m.label}</span>`;
      host.appendChild(el);
    });
  }

function tpSetup(epi, prompt, model) {
    const loadSeq = ++TP_LOAD_SEQ;
    tpStop(); TP = null;
    $("tp-cam-strip").innerHTML = "";
    const mq = model ? "&model=" + encodeURIComponent(model) : "";
    const camerasReady = fetch("/api/episode_cams?episode_index=" + epi + mq)
      .then((r) => r.ok ? r.json() : { cams: [] })
      .then((d) => {
        if (loadSeq !== TP_LOAD_SEQ) return false;
        const cams = d.cams || [];
        const strip = $("tp-cam-strip");
        if (!cams.length) {
          strip.innerHTML = '<div class="miss">no camera video</div>';
          return false;
        }
        strip.innerHTML = cams.map((c) => {
          const src = "/api/episode_video?episode_index=" + epi + "&cam=" + encodeURIComponent(c) + mq;
          return '<div class="tp-cam-cell"><div class="tp-cam-lbl">' + c.split(".").pop() + '</div>'
            + '<video class="tp-cam" muted playsinline preload="auto" src="' + src
            + '" onerror="this.closest(\'.tp-cam-cell\').style.display=\'none\'"></video></div>';
        }).join("");
        return true;
      })
      .catch(() => false);
    fetch("/api/episode_series?episode_index=" + epi + mq).then((r) => r.ok ? r.json() : null).then((series) => {
      if (loadSeq !== TP_LOAD_SEQ) return;
      if (!series || !series.state || !series.state.length) return;
      const sd = series.state[0].length;
      const ad = (series.action && series.action.length) ? series.action[0].length : 0;
      const dimsOn = {}; for (let d = 0; d < sd; d++) dimsOn[d] = true;
      const dimsOnA = {}; for (let d = 0; d < ad; d++) dimsOnA[d] = true;
      const playTime = tpBuildPlayTimeline(series.timestamp || [], series.state.length);
      TP = { epi, model, series, n: series.state.length, fps: Number(series.fps) || 10, i: 0, playing: false, raf: null, dimsOn, sd, dimsOnA, ad, playTime, lastChartDraw: 0, lastVideoSync: 0, camerasReady };
      $("tp-seek").max = String(TP.n - 1); $("tp-seek").step = "0.001"; $("tp-seek").value = "0";
      const loaded = window.ReplayScene ? ReplayScene.loadEpisode(epi, model) : Promise.resolve(false);
      loaded.finally(() => { tpRenderDims(); tpApplyFrame(0, true); });
    }).catch(() => {});
  }

function tpVideos() { return Array.from($("tp-cam-strip").querySelectorAll("video.tp-cam")); }

function tpMasterVideo() {
    return tpVideos().find((v) => {
      const cell = v.closest(".tp-cam-cell");
      return !v.error && (!cell || cell.style.display !== "none");
    }) || null;
  }

function tpSeekVideos(frame = null, force = false) {
    if (!TP) return;
    const t = tpTimeAtFrame(frame == null ? TP.i : frame);
    tpVideos().forEach((v) => {
      const tolerance = Math.min(0.5 / Math.max(1, Number(TP.fps) || 10), 0.03);
      if (v && isFinite(t) && (force || Math.abs((v.currentTime || 0) - t) > tolerance)) {
        try { v.currentTime = t; } catch (e) {}
      }
    });
  }

function tpApplyFrame(frame, syncVideos = false) {
    if (!TP) return;
    const clamped = Math.max(0, Math.min(Number(frame) || 0, TP.n - 1));
    TP.i = clamped; $("tp-seek").value = String(clamped);
    $("tp-time").textContent = tpTimeAtFrame(clamped).toFixed(1) + "s";
    if (window.ReplayScene) ReplayScene.setFrame(TP.epi, clamped, TP_MODEL);
    tpDrawChart();
    if (syncVideos) tpSeekVideos(clamped);
  }

function tpSeek(v) { tpStop(); tpApplyFrame(parseFloat(v) || 0, true); }

function tpToggle() { TP && (TP.playing ? tpStop() : tpPlay()); }

function tpVideoReady(v) {
    return v.error || v.readyState >= 3;
  }

function waitTpVideoReady(v) {
    if (tpVideoReady(v)) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        v.removeEventListener("canplay", done);
        v.removeEventListener("canplaythrough", done);
        v.removeEventListener("error", done);
        resolve();
      };
      v.addEventListener("canplay", done, { once: true });
      v.addEventListener("canplaythrough", done, { once: true });
      v.addEventListener("error", done, { once: true });
    });
  }

function waitTpVideoPlaying(v) {
    if (v.error || !v.paused) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        v.removeEventListener("playing", done);
        v.removeEventListener("error", done);
        resolve();
      };
      v.addEventListener("playing", done, { once: true });
      v.addEventListener("error", done, { once: true });
    });
  }

function alignTpVideos(frame) {
    if (!TP) return Promise.resolve();
    const target = tpTimeAtFrame(frame);
    return Promise.all(tpVideos().map((video) => {
      if (video.error || !Number.isFinite(target)) return Promise.resolve();
      if (Math.abs((video.currentTime || 0) - target) < 0.001) return Promise.resolve();
      return new Promise((resolve) => {
        const done = () => {
          video.removeEventListener("seeked", done);
          video.removeEventListener("error", done);
          resolve();
        };
        video.addEventListener("seeked", done, { once: true });
        video.addEventListener("error", done, { once: true });
        video.currentTime = target;
      });
    }));
  }

async function tpPlay() {
    if (!TP) return;
    if (TP.i >= TP.n - 1) tpApplyFrame(0, true);
    const seq = TP_LOAD_SEQ;
    TP.playing = true; $("tp-play").textContent = "⏸";
    TP.playFrame0 = TP.i;
    const camerasReady = TP.camerasReady;
    if (camerasReady) await camerasReady;
    if (!TP || seq !== TP_LOAD_SEQ || !TP.playing) return;
    const videos = tpVideos();
    await Promise.all(videos.map(waitTpVideoReady));
    if (!TP || seq !== TP_LOAD_SEQ || !TP.playing) return;
    await Promise.all(videos.map((v) => v.play().catch(() => null)));
    await Promise.all(videos.map(waitTpVideoPlaying));
    if (!TP || seq !== TP_LOAD_SEQ || !TP.playing) return;
    videos.forEach((video) => video.pause());
    await alignTpVideos(TP.i);
    await Promise.all(videos.map((v) => v.play().catch(() => null)));
    await new Promise((resolve) => requestAnimationFrame(resolve));
    TP.playWall0 = performance.now();
    TP.lastVideoSync = TP.playWall0;
    TP.raf = requestAnimationFrame(tpPlayFrame);
  }

function tpPlayFrame() {
    if (!TP || !TP.playing) return;
    const targetTime = tpTimeAtFrame(TP.playFrame0)
      + (performance.now() - TP.playWall0) / 1000;
    const frame = tpFrameAtTime(targetTime);
    tpApplyFrame(frame, false);
    const now = performance.now();
    if (now - TP.lastVideoSync >= 200) {
      tpSeekVideos(frame);
      TP.lastVideoSync = now;
    }
    if (frame >= TP.n - 1) { tpStop(); return; }
    TP.raf = requestAnimationFrame(tpPlayFrame);
  }

function tpStop() {
    if (!TP) return;
    TP.playing = false; if (TP.raf) cancelAnimationFrame(TP.raf); TP.raf = null;
    $("tp-play").textContent = "▶";
    tpVideos().forEach((v) => v.pause());
  }

function tpRenderDimRow(hostId, nd, dimsOn, tag) {
    const host = $(hostId); if (!host) return;
    host.innerHTML = "";
    for (let d = 0; d < nd; d++) {
      const el = document.createElement("span");
      el.className = "dim" + (dimsOn[d] ? "" : " off");
      el.style.borderLeft = "8px solid " + RT_COLORS[d % RT_COLORS.length];
      el.textContent = tag + d;
      el.onclick = () => { dimsOn[d] = !dimsOn[d]; tpRenderDims(); };
      host.appendChild(el);
    }
  }

function tpRenderDims() {
    if (!TP) return;
    tpRenderDimRow("tp-achart-dims", TP.ad, TP.dimsOnA, "a");
    tpRenderDimRow("tp-chart-dims", TP.sd, TP.dimsOn, "q");
    tpDrawChart(true);
  }

function tpDrawChart(force = false) {
    if (!TP) return;
    if (!force && TP.playing) {
      const now = performance.now();
      if (now - TP.lastChartDraw < 100) return;
      TP.lastChartDraw = now;
    } else {
      TP.lastChartDraw = performance.now();
    }
    if (TP.ad) drawSeriesChart($("tp-achart-cv"), TP.series.action, TP.playTime, TP.dimsOnA, TP.i);
    drawSeriesChart($("tp-chart-cv"), TP.series.state, TP.playTime, TP.dimsOn, TP.i);
  }

export {
  applyEvalStatus, evalCfg, evalEnabled, evalReset, evalSetup, evalRunToggle, evalResumeOnEnter, loadEvalResults,
  renderEvalSelectors, submitEvalScore,
  loadResultsAll, rvTrialMs,
  tpSeek, tpToggle, trialPopClose, trialPopOpen,
};
