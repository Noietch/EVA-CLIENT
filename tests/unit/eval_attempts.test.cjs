const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const STATIC = path.resolve(__dirname, "../../src/core/app/console/static/js");
const SOURCE = fs.readFileSync(path.join(STATIC, "eval.js"), "utf8")
  .replace(/^import .*\n/gm, "").replace(/export \{[\s\S]*$/, "");
const GRID = fs.readFileSync(path.join(STATIC, "scene_map.js"), "utf8")
  .replace("export function renderSceneGrid", "function renderSceneGrid");

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {setProperty(name, value) { this[name] = value; }};
    this.classes = new Set();
    this.classList = {
      add: (name) => this.classes.add(name), remove: (name) => this.classes.delete(name),
      contains: (name) => this.classes.has(name),
      toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name),
    };
    this.value = "";
  }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  querySelector(tag) {
    return this.children.find((child) => child.tagName === tag) || new Element(tag);
  }
  querySelectorAll() { return this.children; }
  getAttribute(name) { return this[name]; }
  removeAttribute(name) { delete this[name]; }
  set innerHTML(value) { this.children = []; this.html = value; }
  get innerHTML() { return this.html || ""; }
}

function plan(taskSet = "set-a") {
  return {task_set: taskSet, positions: [{position_id: "P1", x: 0, y: 0}], tasks: [
    {task_id: "pick", prompt_en: "Pick cup", scene_ids: ["left", "right"]},
    {task_id: "place", prompt_en: "Place cup", scene_ids: ["right"]},
  ], scenes: [
    {scene_id: "left", placements: [{position_id: "P1", object_id: "cup", name_en: "Cup"}]},
    {scene_id: "right", placements: []},
  ]};
}

function harness({get, post} = {}) {
  const nodes = new Map();
  const $ = (id) => {
    if (!nodes.has(id)) nodes.set(id, new Element(
      ["eval-prompt-list", "eval-model-list"].includes(id) ? "SELECT" : "div"));
    return nodes.get(id);
  };
  const state = {ACTIVE_TAB: "eval", evalRunToggleBusy: null, EVAL_MODEL_NAME: "model-a",
    STATUS: {is_setup_done: true}, CFG: {eval: {scene_plan_sets: ["set-a", "set-b"],
      trials_per_prompt: 5, checkpoints: [{slot: 0, name: "model-a"}, {slot: 1, name: "model-b"}],
      tasks: [{task_set: "set-a", task_id: "pick", prompt_en: "Pick cup",
        milestones: [{id: "lift", label: "Lift cup"}], init_pose: [1, 2]}]}}};
  const calls = [];
  const context = vm.createContext({S: state, $, URLSearchParams,
    document: {createElement: (tag) => new Element(tag), getElementById: (id) => nodes.get(id),
      querySelectorAll: (selector) => selector === "#ms-nodes .ms-node"
        ? $("ms-nodes").children : selector === "#ms-nodes .ms-node.done"
          ? $("ms-nodes").children.filter((node) => node.classList.contains("done")) : []},
    window: {}, fetch: async () => ({ok: false}),
    apiGet: async (url) => get ? get(url) : url.startsWith("/api/scene_plan")
      ? plan(new URL(url, "http://test").searchParams.get("set")) : {model_name: "model-a", records: []},
    apiPost: async (url, body) => { calls.push({url, body}); return post ? post(url, body) : {ok: true}; },
    renderEvalGripper() {}, setPanel() {}, syncChip() {}, setCommandMetadata() {},
    performance: {now: () => 1}, setTimeout() {},
  });
  vm.runInContext(GRID + "\n" + SOURCE + `\nEVAL_LAST_PHASE = "ready"; EVAL_ACTIVE_SLOT = 0;
    this.api = {load: loadEvalScenePlan, results: loadEvalResults, run: evalRun, score: submitEvalScore,
      status: applyEvalStatus, switchModel: evalSwitchCkpt, render: renderEvalSelectors,
      resume: evalResumeOnEnter, tasks: evalTasks, taskKey: evalTaskKey,
      fill: updateMsFill, replay: trialPopOpen,
      select(key, scene = "") { selectEvalTask(key, scene); renderEvalConsole(); },
      current() { return {task: evalTask(), trial: EVAL_TRIAL, clip: EVAL_CLIP,
        scene: EVAL_SCENE_MAP.sceneId, records: EVAL_RECORDS}; },
      resultMatrix(records) { RV = {tasks: [], models: [{model_name: "model-a", records}], trials: 5};
        return rvBuildMatrix(records); },
      resultStats(records) { return rvStats(Object.values(rvBuildMatrix(records)).flat().filter(Boolean)); },
    };`, context);
  return {api: context.api, nodes, calls, state};
}

function record(scene, clip, episode, score = null, task = "pick") {
  return {task_set: "set-a", task_id: task, scene_id: scene, prompt: "Pick cup",
    trial: 1, clip_id: clip, episode_index: episode, score, max_score: 1};
}

const settle = () => new Promise(setImmediate);

test("task-set tasks become runnable selections and preserve authored scoring", async () => {
  const {api, calls} = harness();
  await api.load();
  const tasks = api.tasks();
  assert.equal(tasks.length, 2);
  assert.equal(tasks[0].milestones[0].id, "lift");
  assert.deepEqual(Array.from(tasks[0].init_pose), [1, 2]);
  assert.equal(tasks[1].milestones[0].id, "success");
  api.select(api.taskKey(tasks[1]), "right");
  api.run();
  const start = calls.find((call) => call.url === "/api/eval_start").body;
  assert.equal(start.prompt, "Place cup");
  assert.equal(start.task_set, "set-a");
  assert.equal(start.task_id, "place");
  assert.equal(start.scene_id, "right");
  assert.equal(start.trial, 1);
});

test("reload preserves ready scene and cannot change a running selection", async () => {
  const {api, nodes} = harness();
  await api.load();
  api.select(api.taskKey(api.tasks()[0]), "right");
  await nodes.get("eval-scene-refresh").onclick();
  assert.equal(api.current().scene, "right");
  api.run();
  assert.equal(nodes.get("eval-scene-refresh").disabled, true);
  await nodes.get("eval-scene-refresh").onclick();
  assert.equal(api.current().scene, "right");
});

test("latest attempt wins independently for each scene and RESULT group", async () => {
  const records = [record("left", "old", 0, 1), record("right", "other", 1, 1),
    record("left", "new", 2)];
  const {api} = harness({get: async (url) => url.startsWith("/api/scene_plan")
    ? plan() : {model_name: "model-a", records}});
  await api.load();
  await api.results();
  assert.equal(Object.values(api.current().records).length, 2);
  assert.equal(Object.values(api.current().records).find((r) => r.scene_id === "left").clip_id, "new");
  const matrix = api.resultMatrix(records);
  assert.equal(Object.keys(matrix).length, 2);
  assert.equal(Object.values(matrix).find((trials) => trials[0]?.scene_id === "left")[0].clip_id, "new");
  const stats = api.resultStats(records);
  assert.equal(stats.tested, 1);
  assert.equal(stats.score, 1);
  assert.equal(stats.max, 1);
  assert.equal(stats.succ, 1);
  api.resume();
  assert.equal(api.current().scene, "left");
  assert.equal(api.current().trial, 1);
});

test("rerun cannot score the old clip while fresh results are still loading", async () => {
  let pending;
  let reads = 0;
  const {api, nodes, calls} = harness({get: async (url) => {
    if (url.startsWith("/api/scene_plan")) return plan();
    if (++reads === 1) return {model_name: "model-a", records: [record("left", "old", 0, 1)]};
    return new Promise((resolve) => { pending = resolve; });
  }});
  await api.load();
  await api.results();
  api.run();
  const clip = api.current().clip;
  api.status({web_phase: "running", session_status: "running", is_setup_done: true});
  api.status({web_phase: "ready", session_status: "ready", is_setup_done: true});
  assert.equal(nodes.get("be-submit").disabled, true);
  await api.score();
  assert.equal(calls.some((call) => call.url === "/api/score"), false);
  pending({model_name: "model-a", records: [record("left", "old", 0, 1)]});
  await settle();
  assert.equal(nodes.get("be-submit").disabled, true);
  const firstRefresh = pending;
  api.status({web_phase: "ready", session_status: "ready", is_setup_done: true});
  assert.notEqual(pending, firstRefresh);
  pending({model_name: "model-a", records: [record("left", clip, 1)]});
  await settle();
  assert.equal(nodes.get("be-submit").disabled, false);
});

test("score completion preserves a newly selected scene and locks model changes", async () => {
  let completeScore;
  const {api, calls, nodes} = harness({get: async (url) => url.startsWith("/api/scene_plan")
    ? plan() : {model_name: "model-a", records: [record("left", "old", 0)]},
  post: async (url) => url === "/api/score" ? new Promise((resolve) => { completeScore = resolve; }) : {ok: true}});
  await api.load();
  await api.results();
  const scoring = api.score();
  assert.equal(nodes.get("eval-model-list").disabled, true);
  assert.equal(nodes.get("be-run").disabled, true);
  api.run();
  assert.equal(calls.some((call) => call.url === "/api/eval_start"), false);
  api.switchModel(1);
  assert.equal(calls.some((call) => call.url === "/api/select_ckpt"), false);
  const request = calls.find((call) => call.url === "/api/score").body;
  assert.equal(request.model_name, "model-a");
  assert.equal(request.scene_id, "left");
  api.select(api.taskKey(api.tasks()[0]), "right");
  completeScore({ok: true});
  await scoring;
  assert.equal(api.current().scene, "right");
  assert.equal(api.current().trial, 1);
});

test("score completion advances the trial while the robot is resetting", async () => {
  const {api, nodes, calls} = harness({get: async (url) => url.startsWith("/api/scene_plan")
    ? plan() : {model_name: "model-a", records: [record("left", "saved", 0)]}});
  await api.load();
  await api.results();
  api.status({web_phase: "resetting", session_status: "ready", is_setup_done: false});
  await settle();
  assert.equal(nodes.get("be-submit").disabled, false);
  await api.score();
  assert.equal(calls.find((call) => call.url === "/api/score").body.clip_id, "saved");
  assert.equal(api.current().trial, 2);
  assert.equal(api.current().scene, "left");
  assert.equal(calls.some((call) => call.url === "/api/warmup"), false);
});

test("late old model results cannot replace the newly selected model", async () => {
  let oldResponse;
  let reads = 0;
  const {api, state} = harness({get: async (url) => {
    if (url.startsWith("/api/scene_plan")) return plan();
    if (++reads === 1) return new Promise((resolve) => { oldResponse = resolve; });
    return {model_name: "model-b", records: [record("left", "model-b-clip", 2)]};
  }});
  await api.load();
  const old = api.results();
  api.switchModel(1);
  api.status({web_phase: "ready", session_status: "ready", is_setup_done: true, ckpt_slot: 1});
  await settle();
  oldResponse({model_name: "model-a", records: [record("left", "model-a-clip", 1)]});
  await old;
  assert.equal(state.EVAL_MODEL_NAME, "model-b");
  assert.equal(Object.values(api.current().records)[0].clip_id, "model-b-clip");
});


test("single success milestone fills the scoring rail without NaN", async () => {
  const {api, nodes} = harness();
  await api.load();
  api.select(api.taskKey(api.tasks()[1]), "right");
  nodes.get("ms-nodes").children[0].classList.add("done");
  api.fill();
  assert.equal(nodes.get("ms-rail-fill").style.height, "100%");
  assert.equal(nodes.get("ms-score").textContent, "1/1");
});

test("record replay renders the saved scene snapshot and clears it for a manual record", async () => {
  const {api, nodes} = harness();
  const snapshot = plan();
  const saved = {...record("left", "saved", 0), scene_map: snapshot};
  api.resultMatrix([saved]);
  api.replay(saved, "model-a");
  assert.equal(nodes.get("tp-scene-grid").hidden, false);
  assert.equal(nodes.get("tp-scene-grid").children[0].querySelector("b").textContent, "Cup");
  assert.equal(nodes.get("tp-scene-label").textContent, "set-a / pick / left");
  api.replay({prompt: "manual", trial: 1, clip_id: "manual", episode_index: 1}, "model-a");
  assert.equal(nodes.get("tp-scene-grid").hidden, true);
  assert.equal(nodes.get("tp-scene-grid").children.length, 0);
});


test("tasks with the same prompt retain distinct task IDs in selection and runs", async () => {
  const catalog = plan();
  catalog.tasks[1].prompt_en = "Pick cup";
  const {api, nodes, calls} = harness({get: async () => catalog});
  await api.load();
  const tasks = api.tasks();
  assert.notEqual(api.taskKey(tasks[0]), api.taskKey(tasks[1]));
  assert.match(nodes.get("eval-prompt-list").children[1].textContent, /pick/);
  assert.match(nodes.get("eval-prompt-list").children[2].textContent, /place/);
  api.select(api.taskKey(tasks[1]), "right");
  api.run();
  assert.equal(calls.find((call) => call.url === "/api/eval_start").body.task_id, "place");
});


test("running browser recovery waits for the authoritative task set without selecting another task", async () => {
  let finishPlan;
  const {api, calls} = harness({get: async (url) => url.startsWith("/api/scene_plan")
    ? new Promise((resolve) => { finishPlan = resolve; }) : {model_name: "model-a", records: []}});
  api.status({web_phase: "running", session_status: "running", is_setup_done: true,
    clip_id: "external-running", cell: {task_set: "set-b", task_id: "place", scene_id: "right",
      prompt: "Place cup", trial: 3}});
  api.render();
  assert.equal(calls.some((call) => call.url === "/api/select_task"), false);
  finishPlan(plan("set-b"));
  await settle();
  assert.equal(api.current().task.task_set, "set-b");
  assert.equal(api.current().task.task_id, "place");
  assert.equal(api.current().scene, "right");
  assert.equal(api.current().trial, 3);
  assert.equal(api.current().clip, "external-running");
  assert.equal(calls.some((call) => call.url === "/api/select_task"), false);
});


test("an unlinked manual task still runs without a scene identity", async () => {
  const {api, state, calls} = harness();
  state.CFG.eval.tasks.push({prompt_en: "Manual operation", milestones: [{id: "done", label: "Done"}]});
  await api.load();
  const manual = api.tasks().find((task) => task.prompt_en === "Manual operation");
  api.select(api.taskKey(manual));
  api.run();
  const request = calls.find((call) => call.url === "/api/eval_start").body;
  assert.equal(request.prompt, "Manual operation");
  assert.equal(request.task_set, "");
  assert.equal(request.task_id, "");
  assert.equal(request.scene_id, "");
});
