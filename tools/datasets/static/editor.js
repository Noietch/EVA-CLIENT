import {
  configureEntityUi,
  jumpScene,
  newEntity,
  openObjectDialog,
  renderCurrentEditor,
  renderGridCells,
  renderInfo,
  renderIssues,
  renderObjectEditor,
  renderObjectList,
  renderSceneList,
  renderTaskEditor,
  renderTaskFilters,
  renderTaskList,
  requireBatch,
  validateDraft,
} from "./entity-ui.js";
import {
  configureReview,
  drawReviewCharts,
  openSlot,
  renderReview,
  stopPlayback,
} from "./review.js";

const $ = (id) => document.getElementById(id);
const app = {
  state: null,
  tab: "tasks",
  batch: "",
  robot: "",
  selected: { scenes: "", tasks: "", objects: "" },
  draft: { scenes: null, tasks: null, objects: null },
  original: { scenes: "", tasks: "", objects: "" },
  expandedTask: "",
  selectedSlot: "",
  placementIndex: -1,
  review: null,
  reviewToken: 0,
  playing: false,
  playbackTimer: 0,
  playbackStartedAt: 0,
  playbackFrame: -1,
  playbackChartAt: 0,
  playbackVideoSyncAt: 0,
  robotViewer: null,
  writeBusy: false,
  toastTimer: 0,
  editMode: false,
  objectThumbObserver: null,
};

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") element.textContent = text;
  return element;
}

function input(type, name = "", value = "", placeholder = "") {
  const control = node("input");
  control.type = type;
  control.name = name;
  control.value = value == null ? "" : value;
  control.placeholder = placeholder;
  return control;
}

function textarea(name, value = "", placeholder = "") {
  const control = node("textarea");
  control.name = name;
  control.value = value == null ? "" : value;
  control.placeholder = placeholder;
  return control;
}

function button(label, action = "", className = "button") {
  const control = node("button", className, label);
  control.type = "button";
  if (action) control.dataset.action = action;
  return control;
}

function field(label, control, full = false, hint = "") {
  const wrapper = node("label", "field" + (full ? " full" : ""));
  wrapper.append(node("span", "", label), control);
  if (hint) wrapper.append(node("small", "", hint));
  return wrapper;
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function encodePath(value) {
  return String(value).split("/").map(encodeURIComponent).join("/");
}

function recordKey(record, idField) {
  return (record.batch_id || "") + "::" + record[idField];
}

function planFor(batch) {
  return app.state.plans.find((plan) => plan.batch_id === batch);
}

function objectFor(objectId) {
  return app.state.objects.find((item) => item.object_id === objectId);
}

function sceneFor(batch, sceneId) {
  return app.state.scenes.find(
    (item) => item.batch_id === batch && item.scene_id === sceneId,
  );
}

function showToast(message, error = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("show");
  clearTimeout(app.toastTimer);
  app.toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
}

function showGlobalError(message = "") {
  $("global-error-text").textContent = message;
  $("global-error").hidden = !message;
}

function requireEditMode() {
  if (app.editMode) return true;
  showToast("请先开启编辑模式", true);
  return false;
}

function restoreSelectedDrafts() {
  const specs = [
    ["scenes", "scene_id"],
    ["tasks", "task_id"],
    ["objects", "object_id"],
  ];
  for (const [kind, idField] of specs) {
    const record = app.state[kind].find(
      (item) => recordKey(item, idField) === app.selected[kind],
    );
    app.draft[kind] = record ? clone(record) : null;
    app.original[kind] = record ? record[idField] : "";
  }
}

function syncEditMode() {
  document.body.classList.toggle("read-only", !app.editMode);
  $("edit-mode").checked = app.editMode;
  $("edit-mode-label").textContent = app.editMode ? "编辑中" : "只读";
  document.querySelectorAll("[data-edit-only]").forEach((control) => {
    control.hidden = !app.editMode;
  });
  if (!app.state) return;
  renderInfo();
  if (app.tab === "tasks" && app.selectedSlot && app.review) {
    const task = app.state.tasks.find(
      (item) => recordKey(item, "task_id") === app.selected.tasks,
    );
    const slot = task && task.slots.find((item) => item.slot_id === app.selectedSlot);
    if (task && slot) renderReview(task, slot, app.review);
  } else {
    renderCurrentEditor();
  }
}

async function changeEditMode(control) {
  const enabled = control.checked;
  if (enabled) {
    const confirmed = await confirmAction(
      "开启编辑模式",
      "开启后可修改计划、物体照片和 QC 结果。请确认当前操作不会污染正式数据。",
      "开启编辑",
    );
    if (!confirmed) {
      control.checked = false;
      return;
    }
  } else {
    restoreSelectedDrafts();
  }
  app.editMode = enabled;
  syncEditMode();
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && !["GET", "HEAD"].includes(options.method)) {
    headers.set("X-EVA-Dataset-Editor", "1");
    if (app.editMode) headers.set("X-EVA-Edit-Mode", "1");
  }
  const response = await fetch(path, { ...options, headers });
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error((payload && payload.error) || "请求失败 (" + response.status + ")");
  }
  return payload;
}

function writeJson(path, method, payload) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function runWrite(control, operation, successMessage) {
  if (app.writeBusy || (control && control.disabled)) return;
  app.writeBusy = true;
  if (control) control.disabled = true;
  try {
    await operation();
    showToast(successMessage);
  } catch (error) {
    showToast(error.message || String(error), true);
  } finally {
    app.writeBusy = false;
    if (control) control.disabled = false;
  }
}

async function loadState(preserveReview = false) {
  showGlobalError();
  const params = new URLSearchParams();
  if (app.batch) params.set("batch", app.batch);
  if (app.robot) params.set("robot_type", app.robot);
  const query = params.toString() ? "?" + params.toString() : "";
  try {
    const state = await api("/api/state" + query);
    applyState(state, preserveReview);
    $("loading-state").hidden = true;
    $("app").hidden = false;
    requestAnimationFrame(moveTabThumb);
  } catch (error) {
    $("loading-state").hidden = true;
    showGlobalError(error.message || String(error));
  }
}

function applyState(state, preserveReview = false) {
  app.state = state;
  retainSelections();
  renderBatchSelect();
  renderRobotSelect();
  updateSummary();
  if (app.tab === "tasks") {
    renderTaskFilters();
    renderTaskList();
  } else if (app.tab === "scenes") {
    renderSceneList();
  } else if (app.tab === "objects") {
    renderObjectList();
  } else {
    renderInfo();
    renderIssues();
  }
  if (!preserveReview || !app.selectedSlot) renderCurrentEditor();
}

function retainSelections() {
  const specs = [
    ["scenes", "scene_id"],
    ["tasks", "task_id"],
    ["objects", "object_id"],
  ];
  for (const [kind, idField] of specs) {
    const selected = app.selected[kind];
    if (selected && !app.state[kind].some((item) => recordKey(item, idField) === selected)) {
      app.selected[kind] = "";
      app.original[kind] = "";
      app.draft[kind] = null;
    }
  }
}

function renderBatchSelect() {
  const select = $("batch-select");
  select.replaceChildren(new Option("全部批次", ""));
  for (const batch of app.state.batches) {
    select.add(new Option(batch.batch_id + " · " + batch.robot_type, batch.batch_id));
  }
  select.value = app.batch;
  const url = app.batch
    ? "/api/batches/" + encodeURIComponent(app.batch) + "/export"
    : "";
  for (const link of [$("export-trigger"), $("dataset-export")]) {
    link.href = url || "#";
    link.classList.toggle("disabled", !url);
    link.setAttribute("aria-disabled", String(!url));
  }
}

function renderRobotSelect() {
  const select = $("robot-select");
  select.replaceChildren(new Option("全部机器人", ""));
  for (const robot of app.state.robot_types || []) {
    select.add(new Option(robot, robot));
  }
  select.value = app.robot;
}

function updateSummary() {
  const slots = app.state.tasks.reduce((sum, task) => sum + task.slots.length, 0);
  const collected = app.state.tasks.reduce(
    (sum, task) => sum + task.slots.filter((slot) => slot.episode).length,
    0,
  );
  $("dataset-path").textContent = app.state.plans_root;
  $("dataset-dot").className = "status-dot " + (app.state.issues.length ? "warning" : "ready");
  $("task-count").textContent = app.state.tasks.length;
  $("scene-count").textContent = app.state.scenes.length;
  $("object-count").textContent = app.state.objects.length;
  $("issue-count").textContent = app.state.issues.length;
  $("metric-batches").textContent = app.state.batches.length;
  $("metric-tasks").textContent = app.state.tasks.length;
  $("metric-objects").textContent = app.state.objects.length;
  $("metric-episodes").textContent = collected + " / " + slots;
  $("collected-count").textContent = collected;
  $("slot-total-count").textContent = slots;
  $("pending-count").textContent = slots - collected;
}

function moveTabThumb() {
  const tabs = $("tab-thumb").parentElement;
  const active = tabs.querySelector(".tab.active");
  if (!active) return;
  tabs.style.setProperty("--thumb-x", active.offsetLeft + "px");
  tabs.style.setProperty("--thumb-width", active.offsetWidth + "px");
}

function switchTab(tab) {
  stopPlayback();
  app.tab = tab;
  document.querySelectorAll(".tab").forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === tab);
  });
  document.querySelectorAll(".view").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === tab);
  });
  moveTabThumb();
  if (tab === "tasks" && app.robotViewer) app.robotViewer.resize();
  if (tab === "tasks") {
    renderTaskFilters();
    renderTaskList();
  } else if (tab === "scenes") {
    renderSceneList();
  } else if (tab === "objects") {
    renderObjectList();
  } else {
    renderInfo();
    renderIssues();
  }
  renderCurrentEditor();
}

function emptyList(host, message) {
  host.replaceChildren(node("div", "inline-empty", message));
}

function saveEntity(kind, control) {
  if (!requireEditMode()) return;
  const draft = validateDraft(kind);
  const idField = kind === "scenes" ? "scene_id" : kind === "tasks" ? "task_id" : "object_id";
  const id = draft[idField];
  const current = app.original[kind];
  const batch = draft.batch_id;
  let path;
  if (kind === "objects") {
    path = "/api/objects" + (current ? "/" + encodeURIComponent(current) : "")
      + (app.batch ? "?batch=" + encodeURIComponent(app.batch) : "");
  } else {
    path = "/api/batches/" + encodeURIComponent(batch) + "/" + kind
      + (current ? "/" + encodeURIComponent(current) : "");
  }
  runWrite(control, async () => {
    await writeJson(path, current ? "PUT" : "POST", draft);
    app.original[kind] = id;
    app.selected[kind] = (batch || "") + "::" + id;
    await loadState();
    const saved = app.state[kind].find((item) => recordKey(item, idField) === app.selected[kind]);
    app.draft[kind] = clone(saved);
    renderCurrentEditor();
  }, id + " 已保存");
}

function confirmAction(title, message, label = "确认") {
  const dialog = $("confirm-dialog");
  $("confirm-title").textContent = title;
  $("confirm-message").textContent = message;
  $("confirm-accept").textContent = label;
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "confirm"),
      { once: true },
    );
  });
}

async function deleteEntity(kind, control) {
  if (!requireEditMode()) return;
  const id = app.original[kind];
  if (!id || !await confirmAction("确认删除", "删除 " + id + "？被引用的记录无法删除。", "删除")) {
    return;
  }
  await runWrite(control, async () => {
    let path;
    if (kind === "objects") {
      path = "/api/objects/" + encodeURIComponent(id)
        + (app.batch ? "?batch=" + encodeURIComponent(app.batch) : "");
    } else {
      path = "/api/batches/" + encodeURIComponent(app.draft[kind].batch_id)
        + "/" + kind + "/" + encodeURIComponent(id);
    }
    await api(path, { method: "DELETE" });
    app.selected[kind] = "";
    app.original[kind] = "";
    app.draft[kind] = null;
    await loadState();
    if (kind === "objects") $("object-dialog").close();
  }, id + " 已删除");
}

async function validatePlan(control) {
  if (!app.batch) {
    await loadState();
    showToast("全部批次已重新读取并检查");
    return;
  }
  await runWrite(control, async () => {
    const payload = await api("/api/batches/" + encodeURIComponent(app.batch) + "/validate");
    app.state.issues = payload.issues;
    renderIssues();
    updateSummary();
  }, "计划检查完成");
}

async function importPlan(file, control) {
  if (!requireEditMode() || !file || !requireBatch()) return;
  const confirmed = await confirmAction(
    "导入计划",
    "导入 " + file.name + " 将替换批次 " + app.batch + " 的四个计划文件。",
    "导入",
  );
  if (!confirmed) return;
  const form = new FormData();
  form.append("file", file);
  await runWrite(control, async () => {
    await api("/api/batches/" + encodeURIComponent(app.batch) + "/import", {
      method: "POST",
      body: form,
    });
    clearEditors();
    await loadState();
  }, "计划 ZIP 已导入");
  $("import-file").value = "";
}

async function uploadPhotos(files) {
  if (!requireEditMode() || !files.length || !app.original.objects) return;
  const form = new FormData();
  [...files].forEach((file) => form.append("photos", file));
  await runWrite(null, async () => {
    await api("/api/objects/" + encodeURIComponent(app.original.objects) + "/photos", {
      method: "POST",
      body: form,
    });
    await loadState();
    const fresh = objectFor(app.original.objects);
    app.draft.objects = clone(fresh);
    renderObjectEditor();
  }, "物体照片已上传");
  $("photo-file").value = "";
}

function clearEditors() {
  stopPlayback();
  app.selected = { scenes: "", tasks: "", objects: "" };
  app.original = { scenes: "", tasks: "", objects: "" };
  app.draft = { scenes: null, tasks: null, objects: null };
  app.expandedTask = "";
  app.selectedSlot = "";
  app.review = null;
}

function handleAction(event) {
  const control = event.target.closest("[data-action]");
  if (!control) return;
  const action = control.dataset.action;
  try {
    if (action === "new-task") newEntity("tasks");
    else if (action === "new-scene") newEntity("scenes");
    else if (action === "new-object") newEntity("objects");
    else if (action.startsWith("save-")) saveEntity(action.slice(5), control);
    else if (action.startsWith("delete-")) deleteEntity(action.slice(7), control);
    else if (action === "validate") validatePlan(control);
    else if (action === "import") {
      if (requireEditMode() && requireBatch()) $("import-file").click();
    }
  } catch (error) {
    showToast(error.message || String(error), true);
  }
}

function bindEvents() {
  document.addEventListener("click", handleAction);
  $("edit-mode").addEventListener("change", (event) => changeEditMode(event.currentTarget));
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
  $("batch-select").addEventListener("change", async (event) => {
    app.batch = event.target.value;
    clearEditors();
    $("loading-state").hidden = false;
    $("app").hidden = true;
    await loadState();
  });
  $("robot-select").addEventListener("change", async (event) => {
    app.robot = event.target.value;
    app.batch = "";
    clearEditors();
    $("loading-state").hidden = false;
    $("app").hidden = true;
    await loadState();
  });
  $("task-group-by").addEventListener("change", () => {
    app.expandedTask = "";
    renderTaskList();
  });
  $("task-search").addEventListener("input", renderTaskList);
  $("task-action-filter").addEventListener("change", renderTaskList);
  $("task-category-filter").addEventListener("change", renderTaskList);
  $("scene-search").addEventListener("input", renderSceneList);
  $("object-search").addEventListener("input", renderObjectList);
  $("object-photo-filter").addEventListener("change", renderObjectList);
  $("retry-button").addEventListener("click", () => loadState());
  $("validate-button").addEventListener("click", (event) => validatePlan(event.currentTarget));
  $("import-trigger").addEventListener("click", () => {
    if (requireEditMode() && requireBatch()) $("import-file").click();
  });
  $("import-file").addEventListener("change", (event) => {
    importPlan(event.target.files[0], $("import-trigger"));
  });
  $("photo-file").addEventListener("change", (event) => uploadPhotos(event.target.files));
  document.querySelector("[data-close-object]").addEventListener("click", () => {
    $("object-dialog").close();
  });
  for (const link of [$("export-trigger"), $("dataset-export")]) {
    link.addEventListener("click", (event) => {
      if (!app.batch) {
        event.preventDefault();
        requireBatch();
      }
    });
  }
  window.addEventListener("resize", () => {
    moveTabThumb();
    drawReviewCharts(Number($("review-range") && $("review-range").value || 0));
  });
}

configureEntityUi({
  app,
  node,
  input,
  textarea,
  button,
  field,
  clone,
  encodePath,
  recordKey,
  planFor,
  objectFor,
  sceneFor,
  showToast,
  runWrite,
  writeJson,
  loadState,
  requireEditMode,
  emptyList,
  switchTab,
  stopPlayback,
  openSlot,
});
configureReview({
  app,
  node,
  button,
  input,
  clone,
  encodePath,
  recordKey,
  api,
  writeJson,
  runWrite,
  loadState,
  renderTaskList,
  renderTaskEditor,
  jumpScene,
  objectFor,
  openObjectDialog,
  sceneFor,
  renderGridCells,
  requireEditMode,
});

syncEditMode();
bindEvents();
loadState();
