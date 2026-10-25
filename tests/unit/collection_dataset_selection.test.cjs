const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const COLLECT_SOURCE = fs.readFileSync(
  path.join(__dirname, "../../src/core/app/console/static/js/collect.js"),
  "utf8",
);

function makeDom() {
  const nodes = new Map();
  const error = {textContent: ""};
  const transfer = {
    textContent: "",
    title: "",
    style: {},
    classList: {toggle() {}},
  };
  nodes.set("collect-err", error);
  nodes.set("collect-transfer-info", transfer);
  return {
    get(id) {
      if (id === "collect-control-col") return null;
      return nodes.get(id) || null;
    },
    error,
  };
}

function makeState() {
  return {
    CFG: {
      collect_tasks: {
        old_dataset: [["旧任务", -1]],
        larybench_eval_v1_supplement: [["New task", -1]],
      },
    },
    STATUS: {
      collect: {collecting: false},
      selected_collect_task: "旧任务",
      selected_collect_set: "old_dataset",
      selected_collect_task_index: 0,
      collection_slot_id: "old-slot",
      collection_task_id: "OLD01",
    },
    collectionSlots: {
      dataset: "old_dataset",
      datasetDir: "/tmp/old_dataset",
      loaded: true,
      active: {dataset: "old_dataset", slot_id: "old-slot", task: "旧任务"},
      counts: {},
      scenes: [],
      tasks: [],
      slots: [],
      page: 1,
      pageCount: 1,
      filteredTotal: 0,
      viewerActive: false,
      lastSaveRevision: 0,
      sessionDataset: "old_dataset",
      sessionSlotId: "old-slot",
      sceneFilter: "",
      taskFilter: "",
      showAll: true,
      selectedSlotId: "",
      cursorPinned: false,
      followActivePage: true,
    },
    SCENE_PLAN: {tasks: [], scenes: [], positions: [], objects: []},
    collectTaskSelectionPending: false,
    collectToggleBusy: null,
    ACTIVE_TAB: "collect",
  };
}

function makePayload({active = null, dataset = "larybench_eval_v1_supplement", sessionDataset = "", sessionSlotId = ""} = {}) {
  const task = {
    task_id: "TASK01",
    prompt_en: "New task",
    prompt_zh: "请把蓝色杯子放到绿色盘子里",
    runtime_prompt: "New task",
    scene_ids: ["SCENE01"],
    scene_epsiodes_count: [1],
  };
  const scene = {scene_id: "SCENE01", placements: []};
  const slot = active ? {...active} : null;
  return {
    ok: true,
    dataset,
    active: slot,
    counts: {total: slot ? 1 : 0},
    scenes: slot ? [{id: "SCENE01"}] : [],
    tasks: slot ? [{id: "TASK01"}] : [],
    slots: slot ? [slot] : [],
    page: 1,
    page_count: 1,
    filtered_total: slot ? 1 : 0,
    session_dataset: sessionDataset,
    session_slot_id: sessionSlotId,
    scene_plan: {tasks: slot ? [task] : [], scenes: slot ? [scene] : [], positions: [], objects: []},
  };
}

async function loadCollectModule({apiPost, apiGet, state = makeState()} = {}) {
  const dom = makeDom();
  const selected = {dataset: state.STATUS.selected_collect_set, task: state.STATUS.selected_collect_task};
  const calls = {post: [], get: [], applied: [], navigation: 0};
  const deps = {
    apiPost: apiPost || (async () => ({ok: true})),
    apiGet: apiGet || (async () => makePayload()),
  };
  const context = vm.createContext({
    URLSearchParams,
    AbortController,
    clearTimeout,
    setTimeout,
    performance: {now: () => 100},
    queueMicrotask,
    console,
    Option: class Option {},
    document: {activeElement: null, createElement: () => ({})},
  });
  const modules = new Map();
  const synthetic = (specifier, exports) => {
    const module = new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value());
    }, {context});
    modules.set(specifier, module);
  };
  synthetic("./core.js", {
    $: () => (id) => dom.get(id),
    LIVE: () => ({replayOwner: ""}),
    S: () => state,
    apiGet: () => async (...args) => {
      calls.get.push(args);
      return deps.apiGet(...args);
    },
    apiPost: () => async (...args) => {
      calls.post.push(args);
      return deps.apiPost(...args);
    },
    clientTrace: () => () => {},
  });
  synthetic("./i18n.js", {t: () => (value) => String(value)});
  synthetic("./charts.js", {updateScrub: () => () => {}});
  synthetic("./run.js", {
    applyCollectTaskSelection: () => (prompt, dataset, taskIndex, sceneId) => {
      calls.applied.push({prompt, dataset, taskIndex, sceneId});
      state.STATUS.selected_collect_task = prompt;
      state.STATUS.selected_collect_set = dataset;
      state.STATUS.selected_collect_task_index = Number(taskIndex);
      return true;
    },
    collectTaskIndexValue: () => () => 0,
    collectSetValue: () => () => selected.dataset,
    selectCollectSet: () => (dataset) => {
      selected.dataset = dataset;
      selected.task = state.CFG.collect_tasks[dataset][0][0];
      state.STATUS.selected_collect_set = dataset;
      state.STATUS.selected_collect_task = selected.task;
      state.STATUS.selected_collect_task_index = 0;
      return true;
    },
    collectTaskValue: () => () => selected.task,
    setPanel: () => () => {},
    applyStatus: () => async () => {},
    syncCollectTaskNavigation: () => () => { calls.navigation += 1; },
    uiMode: () => () => "collect",
  });
  synthetic("./replay.js", {
    exitReplayMode: () => () => {},
    loadReviewPlayback: () => async () => {},
    refreshCameraStreams: () => () => {},
    replayStop: () => () => {},
  });
  synthetic("./main.js", {setActiveTab: () => () => {}});
  synthetic("./scene_map.js", {renderSceneGrid: () => () => {}});

  const module = new vm.SourceTextModule(COLLECT_SOURCE, {
    context,
    identifier: path.join(__dirname, "../../src/core/app/console/static/js/collect.js"),
  });
  await module.link((specifier) => {
    const dependency = modules.get(specifier);
    if (!dependency) throw new Error(`Unexpected collect.js import: ${specifier}`);
    return dependency;
  });
  await module.evaluate();
  return {
    selectCollectionDataset: module.namespace.selectCollectionDataset,
    pollCollectionSlots: module.namespace.pollCollectionSlots,
    changeCollectionSlotPage: module.namespace.changeCollectionSlotPage,
    calls, dom, state,
  };
}

test("无 active 槽位时仍确认后端并保留新 dataset 名称", async () => {
  const response = [];
  const harness = await loadCollectModule({
    apiPost: async (pathName, body) => {
      response.push({path: pathName, body});
      return {ok: true, dataset: body.dataset, task: body.task, task_index: 0};
    },
    apiGet: async () => makePayload({active: null}),
  });

  assert.equal(await harness.selectCollectionDataset("larybench_eval_v1_supplement"), true);
  assert.equal(response.length, 1);
  assert.equal(response[0].path, "/api/select_collect_task");
  assert.equal(JSON.stringify(response[0].body), JSON.stringify({
    dataset: "larybench_eval_v1_supplement", task: "New task", task_index: 0,
  }));
  assert.equal(harness.state.collectionSlots.dataset, "larybench_eval_v1_supplement");
  assert.equal(harness.state.STATUS.selected_collect_set, "larybench_eval_v1_supplement");
  assert.equal(harness.state.collectionSlots.active, null);
});

test("后端拒绝切换时不覆盖旧 dataset 并显示错误", async () => {
  const harness = await loadCollectModule({
    apiPost: async () => ({ok: false, error: "任务集不可用"}),
    apiGet: async () => { throw new Error("拒绝后不应加载槽位"); },
  });

  assert.equal(await harness.selectCollectionDataset("larybench_eval_v1_supplement"), false);
  assert.equal(harness.calls.get.length, 0);
  assert.equal(harness.state.collectionSlots.dataset, "old_dataset");
  assert.equal(harness.state.STATUS.selected_collect_set, "old_dataset");
  assert.equal(harness.dom.error.textContent, "任务集不可用");
  assert.equal(harness.state.collectTaskSelectionPending, false);
});

test("请求进行中再次切换会被挡住且不重复请求", async () => {
  let resolvePost;
  const firstPost = new Promise((resolve) => { resolvePost = resolve; });
  const harness = await loadCollectModule({
    apiPost: async (pathName) => {
      assert.equal(pathName, "/api/select_collect_task");
      return firstPost;
    },
    apiGet: async () => makePayload({active: null}),
  });

  const first = harness.selectCollectionDataset("larybench_eval_v1_supplement");
  while (!harness.state.collectTaskSelectionPending) await new Promise((resolve) => setImmediate(resolve));
  assert.equal(await harness.selectCollectionDataset("larybench_eval_v1_supplement"), false);
  assert.equal(harness.calls.post.length, 1);
  resolvePost({ok: true, dataset: "larybench_eval_v1_supplement", task: "New task", task_index: 0});
  assert.equal(await first, true);
  assert.equal(harness.calls.post.length, 1);
});

test("有 active 槽位时使用正确 dataset/slot，并保留中文 task_zh", async () => {
  const active = {
    dataset: "larybench_eval_v1_supplement",
    slot_id: "larybench_eval_v1_supplement:SCENE01:0",
    task_id: "TASK01",
    task_index: 0,
    task: "New task",
    task_zh: "请把蓝色杯子放到绿色盘子里",
    scene_id: "SCENE01",
    round_index: 0,
    round_total: 1,
  };
  const harness = await loadCollectModule({
    apiPost: async (pathName, body) => {
      if (pathName === "/api/select_collect_task") {
        return {ok: true, dataset: body.dataset, task: body.task, task_index: body.task_index};
      }
      assert.equal(pathName, "/api/select_collection_slot");
      assert.equal(JSON.stringify(body), JSON.stringify({
        dataset: "larybench_eval_v1_supplement",
        slot_id: "larybench_eval_v1_supplement:SCENE01:0",
        manual: false,
      }));
      return {ok: true, active, counts: {total: 1}};
    },
    apiGet: async () => makePayload({
      active,
      sessionDataset: "old_dataset",
      sessionSlotId: "old-slot",
    }),
  });

  assert.equal(await harness.selectCollectionDataset("larybench_eval_v1_supplement"), true);
  assert.equal(harness.calls.post.length, 2);
  assert.equal(harness.state.collectionSlots.dataset, "larybench_eval_v1_supplement");
  assert.equal(harness.state.collectionSlots.active.slot_id, "larybench_eval_v1_supplement:SCENE01:0");
  assert.equal(harness.state.collectionSlots.active.task_zh, "请把蓝色杯子放到绿色盘子里");
  assert.equal(harness.state.SCENE_PLAN.tasks[0].prompt_zh, "请把蓝色杯子放到绿色盘子里");
  assert.deepEqual(harness.calls.applied.at(-1), {
    prompt: "New task",
    dataset: "larybench_eval_v1_supplement",
    taskIndex: 0,
    sceneId: "SCENE01",
  });
});

test("翻页立即取消旧请求，乱序响应不覆盖当前页", async () => {
  const requests = [];
  const harness = await loadCollectModule({
    apiGet: (url, options) => new Promise((resolve) => {
      requests.push({url, options, resolve});
    }),
  });
  harness.state.collectionSlots.pageCount = 3;
  harness.state.collectionSlots.followActivePage = false;
  const first = harness.pollCollectionSlots(true);
  harness.changeCollectionSlotPage(1);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].options.signal.aborted, true);
  assert.equal(new URL(requests[1].url, "http://localhost").searchParams.get("page"), "2");
  assert.equal(harness.state.collectionSlots.loading, true);
  requests[1].resolve({...makePayload({dataset: "old_dataset"}), page: 2, page_count: 3});
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(harness.state.collectionSlots.page, 2);
  requests[0].resolve({...makePayload({dataset: "old_dataset"}), page: 1, page_count: 3});
  await first;
  assert.equal(harness.state.collectionSlots.page, 2);
  assert.equal(harness.state.collectionSlots.loading, false);
  assert.equal(harness.dom.error.textContent, "");
});

test("相同版本增量响应保留场景计划和筛选项", async () => {
  let requestCount = 0;
  const payload = {...makePayload({dataset: "old_dataset"}), scene_plan_version: "plan-1"};
  payload.scenes = [{id: "SCENE01"}];
  payload.tasks = [{id: "TASK01"}];
  const harness = await loadCollectModule({
    apiGet: async () => {
      requestCount += 1;
      if (requestCount === 1) return payload;
      const incremental = {...payload};
      delete incremental.scene_plan;
      delete incremental.scenes;
      delete incremental.tasks;
      return incremental;
    },
  });
  await harness.pollCollectionSlots(true);
  const plan = harness.state.SCENE_PLAN;
  await harness.pollCollectionSlots(true);
  assert.equal(harness.state.SCENE_PLAN, plan);
  assert.equal(harness.state.collectionSlots.scenes, payload.scenes);
  assert.equal(harness.state.collectionSlots.tasks, payload.tasks);
  assert.equal(new URL(harness.calls.get[1][0], "http://localhost").searchParams.get("scene_plan_version"), "plan-1");
});

test("翻页失败显示错误并允许重试，后台轮询不与在途请求重叠", async () => {
  let fail = true;
  const harness = await loadCollectModule({
    apiGet: async () => {
      if (fail) throw new Error("HTTP 503");
      return makePayload({dataset: "old_dataset"});
    },
  });
  const first = harness.pollCollectionSlots(true);
  await harness.pollCollectionSlots();
  assert.equal(harness.calls.get.length, 1);
  await first;
  assert.equal(harness.state.collectionSlots.loading, false);
  assert.equal(harness.state.collectionSlots.error, "collect.error.page");
  assert.equal(harness.dom.error.textContent, "collect.error.page");
  fail = false;
  await harness.pollCollectionSlots(true);
  assert.equal(harness.state.collectionSlots.loaded, true);
  assert.equal(harness.state.collectionSlots.error, "");
  assert.equal(harness.dom.error.textContent, "");
});
