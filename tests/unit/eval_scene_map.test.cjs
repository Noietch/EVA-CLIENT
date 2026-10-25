const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const STATIC_JS = path.join(__dirname, "../../src/core/app/console/static/js");
const EVAL_SOURCE = fs.readFileSync(path.join(STATIC_JS, "eval.js"), "utf8");
const GRID_SOURCE = fs.readFileSync(path.join(STATIC_JS, "scene_map.js"), "utf8");

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {setProperty(name, value) { this[name] = value; }};
    this.attributes = {};
    this.hidden = false;
    this.textContent = "";
    this.classes = new Set();
    this.classList = {
      toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name),
      contains: (name) => this.classes.has(name),
    };
  }

  append(...children) { this.children.push(...children); }

  replaceChildren(...children) { this.children = children; }

  querySelector(tagName) { return this.children.find((child) => child.tagName === tagName); }

  querySelectorAll() { return this.children; }

  getAttribute(name) { return name === "src" ? this.src : this.attributes[name]; }

  removeAttribute(name) { delete this[name]; delete this.attributes[name]; }
}

function makePlan() {
  const columns = [0.42, 0.28, 0.14, 0, -0.14, -0.28, -0.42];
  const rows = [0.56, 0.42, 0.28];
  const positions = Array.from({length: 21}, (_, index) => ({
    position_id: `P${index + 1}`, x: rows[Math.floor(index / 7)], y: columns[index % 7],
  }));
  return {
    positions,
    display_grid: {column_axis: "y", row_axis: "x", columns, rows,
      position_ids: positions.map((position) => position.position_id)},
    tasks: [
      {task_id: "wrong", prompt_en: "Display prompt", scene_ids: ["wrong-scene"]},
      {task_id: "pick", prompt_en: "Canonical pick", prompt_zh: "拿杯子",
        scene_ids: ["left", "right"], operation_object_ids: ["cup"]},
      {task_id: "place", prompt_en: "Place prompt", scene_ids: ["right"],
        operation_object_ids: ["plate"]},
    ],
    scenes: [
      {scene_id: "left", placements: [{position_id: "P1", object_id: "cup",
        name_en: "Cup", photo_url: "/cup.jpg", pose_xyzw: [0, 0, 0, 1, 0, 0, 0]}]},
      {scene_id: "right", placements: [{position_id: "P2", object_id: "plate",
        name_en: "Plate", photo_url: "/plate.jpg"}]},
      {scene_id: "wrong-scene", placements: []},
    ],
  };
}

function loadSceneMap(apiGet = async () => makePlan()) {
  const nodes = new Map([
    "eval-scene-grid", "eval-scene-list", "eval-scene-set-list", "eval-scene-task",
    "eval-scene-status", "eval-scene-refresh",
  ].map((id) => [id, new Element()]));
  const state = {CFG: {eval: {
    scene_plan_sets: ["set-a", "set-b"],
    tasks: [
      {prompt_en: "Display prompt", task_id: "pick", task_set: "set-a"},
      {prompt_en: "Place prompt", task_id: "place", task_set: "set-a"},
      {prompt_en: "Other set", task_id: "pick", task_set: "set-b"},
    ],
  }}};
  const calls = [];
  const context = vm.createContext({
    URLSearchParams,
    S: state,
    $: (id) => nodes.get(id),
    document: {createElement: (tagName) => new Element(tagName)},
    apiGet: (url) => { calls.push(url); return apiGet(url); },
    apiPost: async () => ({ok: true}),
  });
  const evalChunk = EVAL_SOURCE.slice(
    EVAL_SOURCE.indexOf("let EVAL_TASK ="), EVAL_SOURCE.indexOf("// ===== results ====="),
  );
  vm.runInContext(GRID_SOURCE.replace("export function renderSceneGrid", "function renderSceneGrid")
    + "\n" + evalChunk + `\nrenderEvalPromptList = () => {}; renderEvalTrialRow = () => {};
    renderEvalConsole = () => renderEvalSceneMap();
    this.sceneMap = {
      state: EVAL_SCENE_MAP, load: loadEvalScenePlan, render: renderEvalSceneMap,
      setPrompt(value) {
        const task = (evalCfg().tasks || []).find((entry) => entry.prompt_en === value)
          || evalTasks().find((entry) => entry.prompt_en === value);
        EVAL_TASK = task ? evalTaskKey(task) : null;
        if (task?.task_set && task.task_set !== EVAL_SCENE_MAP.taskSet) {
          EVAL_SCENE_MAP.taskSet = null; EVAL_SCENE_MAP.plan = null;
        }
      },
      setPhase(value) { EVAL_LAST_PHASE = value; },
    };`, context);
  return {api: context.sceneMap, nodes, calls, state};
}

function occupiedCell(nodes, positionId) {
  return nodes.get("eval-scene-grid").children.find((cell) => (
    cell.dataset.positionIds.split("|").includes(positionId)
  ));
}

test("评估任务按 task_id 匹配，场景切换显示 21 格、物体图片与任务高亮", async () => {
  const {api, nodes, calls} = loadSceneMap();
  api.setPrompt("Display prompt");
  await api.load();
  const query = new URL(calls[0], "http://test");
  assert.equal(query.pathname, "/api/scene_plan");
  assert.equal(query.searchParams.get("scope"), "eval");
  assert.equal(query.searchParams.get("set"), "set-a");
  assert.equal(nodes.get("eval-scene-grid").children.length, 21);
  const selector = nodes.get("eval-scene-list");
  assert.deepEqual(selector.children.map((option) => option.value), ["left", "right"]);
  assert.equal(selector.value, "left");
  const cup = occupiedCell(nodes, "P1");
  assert.equal(cup.querySelector("b").textContent, "Cup");
  assert.equal(cup.querySelector("img").src, "/cup.jpg");
  assert.equal(cup.querySelector("img").hidden, false);
  assert.equal(cup.classList.contains("task-relevant"), true);

  selector.value = "right";
  selector.onchange();
  const plate = occupiedCell(nodes, "P2");
  assert.equal(plate.querySelector("b").textContent, "Plate");
  assert.equal(plate.querySelector("img").src, "/plate.jpg");
  assert.equal(plate.classList.contains("task-relevant"), false);
  api.setPrompt("Place prompt");
  await api.load();
  assert.equal(calls.length, 1);
  assert.equal(selector.value, "right");
  assert.equal(selector.disabled, true);
  assert.equal(plate.classList.contains("task-relevant"), true);
});

test("未匹配的任务清空场景后重新选择可恢复同一网格", async () => {
  const {api, nodes} = loadSceneMap();
  api.setPrompt("Display prompt");
  await api.load();
  api.setPrompt("Unlinked task");
  api.render();
  const grid = nodes.get("eval-scene-grid");
  assert.equal(grid.hidden, true);
  assert.equal(grid.children.length, 0);
  assert.match(nodes.get("eval-scene-status").textContent, /No scene map linked/);
  api.setPrompt("Display prompt");
  api.render();
  assert.equal(grid.hidden, false);
  assert.equal(grid.children.length, 21);
  assert.equal(occupiedCell(nodes, "P1").querySelector("b").textContent, "Cup");
});

test("评估执行过程中锁定场景选择，恢复就绪后可切换", async () => {
  const {api, nodes} = loadSceneMap();
  api.setPrompt("Display prompt");
  await api.load();
  const selector = nodes.get("eval-scene-list");
  for (const phase of ["starting", "running", "stopping", "resetting", "warming"]) {
    api.setPhase(phase);
    api.render();
    assert.equal(selector.disabled, true, phase);
    selector.value = "right";
    selector.onchange();
    assert.equal(selector.value, "left", phase);
    assert.equal(occupiedCell(nodes, "P1").querySelector("b").textContent, "Cup");
  }
  api.setPhase("ready");
  api.render();
  assert.equal(selector.disabled, false);
  selector.value = "right";
  selector.onchange();
  assert.equal(occupiedCell(nodes, "P2").querySelector("b").textContent, "Plate");
});

test("异步旧任务集响应不会覆盖新任务集场景", async () => {
  const pending = new Map();
  const {api, nodes} = loadSceneMap((url) => new Promise((resolve) => {
    pending.set(new URL(url, "http://test").searchParams.get("set"), resolve);
  }));
  api.setPrompt("Display prompt");
  const first = api.load();
  api.setPrompt("Other set");
  const second = api.load();
  const newerPlan = makePlan();
  newerPlan.scenes[0].placements[0].name_en = "New cup";
  pending.get("set-b")(newerPlan);
  await second;
  assert.equal(occupiedCell(nodes, "P1").querySelector("b").textContent, "New cup");
  pending.get("set-a")(makePlan());
  await first;
  assert.equal(occupiedCell(nodes, "P1").querySelector("b").textContent, "New cup");
  assert.equal(nodes.get("eval-scene-set-list").value, "set-b");
  assert.equal(nodes.get("eval-scene-status").textContent, "");
});


test("未配置评估任务集时保持空白，不请求默认场景计划", async () => {
  const {api, nodes, calls, state} = loadSceneMap();
  state.CFG.eval = {};
  await api.load();
  await api.load(true);
  assert.equal(calls.length, 0);
  assert.equal(api.state.plan, null);
  assert.equal(nodes.get("eval-scene-grid").hidden, true);
  assert.equal(nodes.get("eval-scene-status").textContent, "No evaluation task set configured");
});
