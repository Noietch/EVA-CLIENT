const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const STATIC_JS = path.join(__dirname, "../../src/core/app/console/static/js");

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {setProperty(name, value) { this[name] = value; }};
    this.textContent = "";
    this.classes = new Set();
    this.classList = {toggle: (name, enabled) => (
      enabled ? this.classes.add(name) : this.classes.delete(name)
    )};
  }

  append(...children) { this.children.push(...children); }

  replaceChildren(...children) { this.children = children; }

  querySelector(tagName) { return this.children.find((child) => child.tagName === tagName); }

  querySelectorAll() { return this.children; }

  getAttribute(name) { return this[name]; }

  removeAttribute(name) { delete this[name]; }
}

function makePlan(name = "Cup") {
  return {
    positions: [{position_id: "P1", x: 0, y: 0}, {position_id: "P2", x: 0, y: 1}],
    tasks: [
      {task_id: "pick", prompt_en: "Pick cup", operation_object_ids: ["cup"],
        scene_ids: ["left", "right"]},
      {task_id: "place", prompt_zh: "放盘子", operation_object_ids: ["plate"],
        scene_ids: ["right"]},
    ],
    scenes: [
      {scene_id: "left", placements: [{position_id: "P1", object_id: "cup",
        name_en: name, photo_url: "/cup.jpg"}]},
      {scene_id: "right", placements: [{position_id: "P2", object_id: "plate",
        name_en: "Plate", photo_url: "/plate.jpg"}]},
    ],
  };
}

function loadDashboardSceneMap(apiGet = async () => makePlan(), sets = {
  collect: ["collect-a", "collect-b"], eval: ["eval-a"],
}, dashboardApi = async () => ({available_range: {}, marker: "dashboard"})) {
  const nodes = new Map([
    "dashboard-scene-scope", "dashboard-scene-set", "dashboard-scene-task",
    "dashboard-scene-list", "dashboard-scene-grid", "dashboard-scene-status",
    "dashboard-scene-refresh", "dashboard-scene-all", "dashboard-scene-filter-label",
    "dashboard-error", "dash-date-from", "dash-date-to",
  ].map((id) => [id, new Element()]));
  const selectedPlan = {name: "active collection plan"};
  const state = {CFG: {scene_plan_sets: sets}, SCENE_PLAN: selectedPlan};
  const calls = [];
  const statsCalls = [];
  const renderedStats = [];
  const context = vm.createContext({
    URLSearchParams,
    S: state,
    $: (id) => nodes.get(id),
    document: {createElement: (tagName) => new Element(tagName)},
    loadDatasetManager: () => {},
    renderedStats,
    apiGet: (url) => {
      if (url.startsWith("/api/dashboard")) {
        statsCalls.push(url);
        return dashboardApi(url);
      }
      calls.push(url);
      return apiGet(url);
    },
  });
  const gridSource = fs.readFileSync(path.join(STATIC_JS, "scene_map.js"), "utf8")
    .replace("export function renderSceneGrid", "function renderSceneGrid");
  const dashboardSource = fs.readFileSync(path.join(STATIC_JS, "dashboard.js"), "utf8")
    .replace(/^import .*;\n/gm, "").replace(/^export /gm, "");
  vm.runInContext(gridSource + "\n" + dashboardSource + `\nthis.sceneMap = {
    init: initDashboardSceneMap, load: loadDashboardScenePlan, render: renderDashboardSceneMap,
    dashboard: loadDashboard,
    dates(from, to) {
      dateRangeInitialized = dateRangeApplied = true;
      $("dash-date-from").value = from;
      $("dash-date-to").value = to;
    },
  };
  renderDashboard = () => renderedStats.push(dashboardData);`, context);
  return {api: context.sceneMap, nodes, calls, statsCalls, renderedStats, state, selectedPlan};
}

function select(nodes, id, value) {
  const selector = nodes.get(id);
  selector.value = value;
  return selector.onchange({target: selector});
}

function cell(nodes, positionId) {
  return nodes.get("dashboard-scene-grid").children.find((item) => (
    item.dataset.positionIds.split("|").includes(positionId)
  ));
}

test("看板场景选择显示图片与高亮，任务切换不拉取，刷新保留合法选择", async () => {
  const {api, nodes, calls, state, selectedPlan} = loadDashboardSceneMap();
  await api.init();
  assert.equal(new URL(calls[0], "http://test").search, "?scope=collect&set=collect-a");
  assert.equal(cell(nodes, "P1").querySelector("b").textContent, "Cup");
  assert.equal(cell(nodes, "P1").querySelector("img").src, "/cup.jpg");
  assert.equal(cell(nodes, "P1").classes.has("task-relevant"), true);
  select(nodes, "dashboard-scene-list", "right");
  assert.equal(cell(nodes, "P2").querySelector("b").textContent, "Plate");
  assert.equal(cell(nodes, "P2").classes.has("task-relevant"), false);
  select(nodes, "dashboard-scene-task", "place");
  assert.equal(cell(nodes, "P2").classes.has("task-relevant"), true);
  assert.equal(calls.length, 1);
  await api.load();
  assert.equal(nodes.get("dashboard-scene-task").value, "place");
  assert.equal(nodes.get("dashboard-scene-list").value, "right");
  assert.equal(state.SCENE_PLAN, selectedPlan);
});

test("看板按选择的 scope 与任务集独立加载场景", async () => {
  const {api, nodes, calls} = loadDashboardSceneMap(async (url) => (
    makePlan(new URL(url, "http://test").searchParams.get("set"))
  ));
  await api.init();
  select(nodes, "dashboard-scene-set", "collect-b");
  await new Promise(setImmediate);
  assert.equal(cell(nodes, "P1").querySelector("b").textContent, "collect-b");
  select(nodes, "dashboard-scene-scope", "eval");
  await new Promise(setImmediate);
  const lastQuery = new URL(calls.at(-1), "http://test").searchParams;
  assert.equal(lastQuery.get("scope"), "eval");
  assert.equal(lastQuery.get("set"), "eval-a");
  assert.equal(nodes.get("dashboard-scene-set").value, "eval-a");
  assert.equal(cell(nodes, "P1").querySelector("b").textContent, "eval-a");
});

test("加载与请求失败清空旧图，刷新后可重绘同一布局", async () => {
  let resolveRefresh;
  let fail = false;
  const {api, nodes} = loadDashboardSceneMap(async () => {
    if (fail) throw new Error("Scene catalog unavailable");
    if (resolveRefresh === null) return new Promise((resolve) => { resolveRefresh = resolve; });
    return makePlan();
  });
  await api.init();
  const grid = nodes.get("dashboard-scene-grid");
  assert.ok(grid.dataset.layoutKey);
  resolveRefresh = null;
  const pending = api.load();
  assert.equal(grid.hidden, true);
  assert.equal(grid.children.length, 0);
  assert.equal(grid.dataset.layoutKey, undefined);
  resolveRefresh(makePlan());
  await pending;
  assert.equal(cell(nodes, "P1").querySelector("b").textContent, "Cup");
  fail = true;
  await api.load();
  assert.equal(grid.hidden, true);
  assert.equal(grid.children.length, 0);
  assert.equal(grid.dataset.layoutKey, undefined);
  assert.equal(nodes.get("dashboard-scene-status").textContent, "Scene catalog unavailable");
  fail = false;
  await api.load();
  assert.equal(grid.hidden, false);
  assert.equal(cell(nodes, "P1").querySelector("b").textContent, "Cup");
});

test("任务集切换后的旧异步响应无法覆盖新场景", async () => {
  const pending = new Map();
  const {api, nodes} = loadDashboardSceneMap((url) => new Promise((resolve) => {
    pending.set(new URL(url, "http://test").searchParams.get("set"), resolve);
  }));
  const first = api.init();
  select(nodes, "dashboard-scene-scope", "eval");
  pending.get("eval-a")(makePlan("Eval cup"));
  await new Promise(setImmediate);
  pending.get("collect-a")(makePlan("Old cup"));
  await first;
  assert.equal(nodes.get("dashboard-scene-scope").value, "eval");
  assert.equal(nodes.get("dashboard-scene-set").value, "eval-a");
  assert.equal(cell(nodes, "P1").querySelector("b").textContent, "Eval cup");
});

test("没有任务集、任务或场景时给出对应提示", async () => {
  const empty = loadDashboardSceneMap(undefined, {collect: [], eval: []});
  await empty.api.init();
  assert.equal(empty.calls.length, 0);
  assert.match(empty.nodes.get("dashboard-scene-status").textContent, /No scene task sets/);
  assert.equal(empty.nodes.get("dashboard-scene-refresh").disabled, true);
  const noTasks = loadDashboardSceneMap(async () => ({tasks: [], scenes: []}));
  await noTasks.api.init();
  assert.match(noTasks.nodes.get("dashboard-scene-status").textContent, /No tasks found/);
  const noScenes = loadDashboardSceneMap(async () => ({
    tasks: [{task_id: "pick", prompt_en: "Pick", scene_ids: []}], scenes: [],
  }));
  await noScenes.api.init();
  assert.match(noScenes.nodes.get("dashboard-scene-status").textContent, /No scene layout/);
});

test("目录发现失败显示具体错误，保留切回正常工作区的入口", async () => {
  const {api, nodes, state} = loadDashboardSceneMap(undefined, {
    collect: ["collect-a"], eval: [],
  });
  state.CFG.scene_plan_errors = {eval: "duplicate scene task-set name: eval-a"};
  await api.init();
  await select(nodes, "dashboard-scene-scope", "eval");
  assert.equal(nodes.get("dashboard-scene-status").textContent,
    "duplicate scene task-set name: eval-a");
  assert.equal(nodes.get("dashboard-scene-refresh").disabled, true);
  await select(nodes, "dashboard-scene-scope", "collect");
  assert.equal(nodes.get("dashboard-scene-status").textContent, "");
  assert.equal(nodes.get("dashboard-scene-grid").hidden, false);
});

test("默认预览保留全部统计，主动选场景发送精确筛选且可恢复 ALL DATA", async () => {
  const {api, nodes, statsCalls} = loadDashboardSceneMap();
  await api.init();
  await api.dashboard();
  assert.equal(statsCalls.at(-1), "/api/dashboard");
  assert.match(nodes.get("dashboard-scene-filter-label").textContent, /ALL DATA.*preview/);
  api.dates("2026-10-01", "2026-10-24");
  await select(nodes, "dashboard-scene-list", "right");
  const query = new URL(statsCalls.at(-1), "http://test").searchParams;
  assert.equal(query.get("scene_scope"), "collect");
  assert.equal(query.get("task_set"), "collect-a");
  assert.equal(query.get("task_id"), "pick");
  assert.equal(query.get("scene_id"), "right");
  assert.equal(query.get("from"), "2026-10-01");
  assert.equal(query.get("to"), "2026-10-24");
  assert.match(nodes.get("dashboard-scene-filter-label").textContent,
    /COLLECTION: collect-a \/ pick \/ right.*EVALUATION: ALL DATA/);
  await nodes.get("dashboard-scene-all").onclick();
  const allQuery = new URL(statsCalls.at(-1), "http://test").searchParams;
  assert.equal(allQuery.has("scene_scope"), false);
  assert.equal(allQuery.get("from"), "2026-10-01");
  assert.equal(nodes.get("dashboard-scene-all").disabled, true);
});

test("改变 scope、任务集和任务均刷新统计，旧响应不能覆盖新的场景统计", async () => {
  const pending = [];
  const {api, nodes, statsCalls, renderedStats} = loadDashboardSceneMap(
    undefined, undefined, (url) => new Promise((resolve) => pending.push({url, resolve})),
  );
  await api.init();
  const initial = api.dashboard();
  const switched = select(nodes, "dashboard-scene-scope", "eval");
  await new Promise(setImmediate);
  assert.equal(pending.length, 2);
  const newer = {available_range: {}, marker: "selected eval scene"};
  pending[1].resolve(newer);
  await switched;
  pending[0].resolve({available_range: {}, marker: "stale all data"});
  await initial;
  assert.equal(renderedStats.length, 1);
  assert.equal(renderedStats[0].marker, newer.marker);
  let query = new URL(statsCalls.at(-1), "http://test").searchParams;
  assert.equal(query.get("scene_scope"), "eval");
  assert.equal(query.get("task_set"), "eval-a");
  const taskChange = select(nodes, "dashboard-scene-task", "place");
  query = new URL(statsCalls.at(-1), "http://test").searchParams;
  assert.equal(query.get("task_id"), "place");
  assert.equal(query.get("scene_id"), "right");
  pending.at(-1).resolve({available_range: {}, marker: "place"});
  await taskChange;
  const collectChange = select(nodes, "dashboard-scene-scope", "collect");
  await new Promise(setImmediate);
  pending.at(-1).resolve({available_range: {}, marker: "collect"});
  await collectChange;
  const setChange = select(nodes, "dashboard-scene-set", "collect-b");
  await new Promise(setImmediate);
  query = new URL(statsCalls.at(-1), "http://test").searchParams;
  assert.equal(query.get("task_set"), "collect-b");
  pending.at(-1).resolve({available_range: {}, marker: "collect-b"});
  await setChange;
});
