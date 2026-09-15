import { loadThree } from "./robot-viewer.js";
import {
  configureEntityUi,
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
  renderTaskList,
  navigateQcSlot,
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
// Standard UI glossary: every key is [Chinese, English].
const LOCALE_TABLE = {
  "brand.dataset": ["采集数据", "Collection Dataset"],
  "path.loading": ["正在读取计划批次...", "Loading task plans..."],
  "actions.refreshDatabase": ["刷新数据库", "Refresh database"],
  "actions.retry": ["重试", "Retry"],
  "actions.exportQc": ["导出补采清单", "Export reshoot list"],
  "actions.importPlan": ["导入计划压缩包", "Import plan ZIP"],
  "actions.importObjects": ["批量导入物体 CSV（可同时选择照片）", "Import objects CSV (photos optional)"],
  "actions.exportPlan": ["导出计划压缩包", "Export plan ZIP"],
  "actions.publishTaskSet": ["发布任务集到 HF", "Publish task set to HF"],
  "sync.title": ["数据传输", "Data transfer"],
  "sync.taskSet": ["下载任务集", "Download task set"],
  "sync.assets": ["上传资源", "Upload assets"],
  "sync.dataset": ["下载数据集", "Download dataset"],
  "sync.uploadQc": ["上传质检结果", "Upload QC"],
  "sync.downloadQc": ["下载质检结果", "Download QC"],
  "sync.status": ["状态", "Status"],
  "sync.idle": ["就绪", "Ready"],
  "sync.running": ["正在处理", "In progress"],
  "sync.done": ["已完成", "Completed"],
  "sync.failed": ["同步失败", "Sync failed"],
  "tabs.aria": ["数据管理", "Data management"],
  "tabs.dashboard": ["看板", "Dashboard"],
  "tabs.qc": ["质检", "QC"],
  "tabs.objects": ["物体", "Objects"],
  "loading.state": ["正在载入计划与采集数据", "Loading plans and collection data"],
  "qc.eyebrow": ["质量检查", "Quality check"],
  "qc.title": ["质检审核", "QC review"],
  "qc.slotOverview": ["质检槽位概览", "QC slot overview"],
  "qc.operation": ["质检操作", "Quality check operation"],
  "qc.operationEyebrow": ["质检操作", "QC operation"],
  "qc.noSlot": ["未选择槽位", "No slot selected"],
  "qc.chooseSlotTitle": ["选择槽位开始质检", "Select a slot to start QC"],
  "qc.eta": ["预计完成", "ETA"],
  "qc.collected": ["已采集", "COLLECTED"],
  "qc.fixedPlan": ["固定计划 · 质检状态", "FIXED PLAN · QC STATUS"],
  "qc.remaining": ["待处理", "REMAINING"],
  "qc.chooseBatch": ["选择计划批次后查看质检槽位", "Choose a plan batch to view QC slots"],
  "qc.noSlots": ["当前批次没有可审核的槽位", "This batch has no reviewable slots"],
  "qc.noMatches": ["当前筛选没有匹配的槽位", "No slots match the current filters"],
  "qc.loadingSlot": ["正在载入槽位数据", "Loading slot data"],
  "qc.loadingState": ["正在加载质检数据", "Loading QC data"],
  "qc.loadingHint": ["正在读取批次、槽位和质检状态…", "Reading batches, slots, and QC status…"],
  "qc.pendingSummary": ["个待补采 / 待复核槽位", "slots pending reshoot / review"],
  "qc.round": ["轮次", "Round"],
  "qc.episode": ["数据集片段", "Episode"],
  "qc.noVideo": ["未发现相机视频", "No camera video found"],
  "qc.emptySlot": ["空槽位 · 等待采集", "EMPTY SLOT · WAITING FOR COLLECTION"],
  "qc.review": ["复核", "REVIEW"],
  "qc.pendingState": ["待采集", "PENDING"],
  "qc.frameLabels": ["帧标签", "FRAME LABELS"],
  "qc.staticFrames": ["静止帧", "STATIC FRAMES"],
  "qc.nonStaticFrames": ["非静止帧", "NON-STATIC FRAMES"],
  "qc.noFrameLabels": ["暂无帧标签", "NO FRAME LABELS"],
  "qc.qualityControl": ["质量控制", "Quality control"],
  "qc.notePlaceholder": ["质检备注（可选）", "QC note (optional)"],
  "qc.reasonPlaceholder": ["选择主要原因", "Select primary reason"],
  "qc.otherReason": ["其他", "Other"],
  "qc.imageReason": ["图像质量问题", "Image quality"],
  "qc.trajectoryReason": ["轨迹质量问题", "Trajectory quality"],
  "qc.taskReason": ["任务完成不符合要求", "Task does not meet requirements"],
  "qc.staticFramesExcessive": ["中间静止帧过多", "Too many middle static frames"],
  "qc.staticFramesHint": ["机器识别：中间存在静止段，请人工复核", "Machine flag: middle static segment needs review"],
  "qc.trimRange": ["保留帧范围", "Keep frame range"],
  "qc.trimApply": ["截取并保存", "Trim and save"],
  "qc.trimSaved": ["已按选定帧范围截取数据", "Episode trimmed to the selected frame range"],
  "qc.trimNoChange": ["当前范围未改变，无需截取", "The selected range is unchanged"],
  "qc.trimStart": ["起始边界", "Start boundary"],
  "qc.trimEnd": ["结束边界", "End boundary"],
  "qc.trimTag": ["帧范围调整", "FRAME RANGE"],
  "qc.moveLeft": ["左移", "Move left"],
  "qc.moveRight": ["右移", "Move right"],
  "qc.frameBoundary": ["边界", "Boundary"],
  "qc.pass": ["通过", "Pass"],
  "qc.fail": ["未通过", "Fail"],
  "qc.save": ["保存", "Save"],
  "qc.saved": ["质检备注已保存", "QC note saved"],
  "qc.nextItem": ["下一条", "Next"],
  "qc.mark": ["标记问题", "Mark issue"],
  "qc.savePassed": ["数据集片段已通过", "Episode passed"],
  "qc.saveFailed": ["数据集片段已标记未通过", "Episode marked failed"],
  "filters.robot": ["机器人", "Robot"],
  "filters.planDirectory": ["计划文件目录", "Plan directory"],
  "filters.allRobots": ["全部机器人", "All robots"],
  "filters.batch": ["计划批次", "Plan batch"],
  "filters.chooseBatch": ["选择批次", "Choose a batch"],
  "filters.batchNavigation": ["批次导航", "Batch navigation"],
  "filters.previousBatch": ["上一批次", "Previous batch"],
  "filters.nextBatch": ["下一批次", "Next batch"],
  "filters.scene": ["场景", "Scene"],
  "filters.allScenes": ["全部场景", "All scenes"],
  "filters.task": ["任务", "Task"],
  "filters.allTasks": ["全部任务", "All tasks"],
  "status.passed": ["已质检通过", "Passed QC"],
  "status.unreviewed": ["未质检", "Unreviewed"],
  "status.failed": ["质检未通过", "Failed QC"],
  "status.pending": ["待采集", "Pending"],
  "objects.eyebrow": ["共享资产", "Shared assets"],
  "objects.title": ["公用物体表", "Object catalog"],
  "objects.search": ["搜索编号或名称", "Search ID or name"],
  "objects.all": ["全部物体", "All objects"],
  "objects.missingPhotos": ["未拍照片", "Missing photos"],
  "objects.readyPhotos": ["已有照片", "Photos ready"],
  "dashboard.eyebrow": ["原始数据运营", "Raw data operations"],
  "dashboard.title": ["运营看板", "Operations dashboard"],
  "dashboard.subtitle": ["原始数据产出、采集需求与质检健康度。", "Raw episode throughput, collection demand, and QC health."],
  "dashboard.allData": ["全部数据", "ALL DATA"],
  "dashboard.range": ["采集范围", "COLLECTION RANGE"],
  "dashboard.allPlans": ["全部计划", "ALL PLANS"],
  "dashboard.calculating": ["正在统计...", "Calculating..."],
  "dashboard.overviewEyebrow": ["数据集概览", "Dataset overview"],
  "dashboard.overviewTitle": ["数据集概览", "Dataset overview"],
  "dashboard.overviewSubtitle": ["采集产出、任务需求与数据质量。", "Collection output, demand, and quality."],
  "kpi.target": ["计划总量", "DATA TARGET"],
  "kpi.targetNote": ["计划槽位数", "planned slots"],
  "kpi.collected": ["已采集", "COLLECTED"],
  "kpi.collectedNote": ["已采集片段数", "episodes captured"],
  "kpi.frames": ["帧数", "FRAMES"],
  "kpi.framesNote": ["轨迹帧", "trajectory frames"],
  "kpi.duration": ["平均时长", "AVG DURATION"],
  "kpi.durationNote": ["每条已采集片段", "per collected episode"],
  "kpi.efficiency": ["采集效率", "EFFICIENCY"],
  "kpi.efficiencyNote": ["已采集 / 计划总量", "collected / target"],
  "kpi.validRate": ["通过率", "VALID RATE"],
  "kpi.validRateNote": ["通过质检 / 已采集", "QC passed / collected"],
  "kpi.passed": ["质检通过", "PASSED QC"],
  "kpi.passedNote": ["已通过，不再补采", "approved, not remaining"],
  "kpi.toCollect": ["待补数据", "TO COLLECT"],
  "kpi.toCollectNote": ["待采集或需返修", "pending or repair"],
  "dashboard.collectionEyebrow": ["采集需求", "Collection demand"],
  "dashboard.collectionTitle": ["按机器人统计", "Collection by robot"],
  "dashboard.healthEyebrow": ["流程健康度", "Pipeline health"],
  "dashboard.healthTitle": ["质检健康度", "QC health"],
  "dashboard.qcEyebrow": ["质量控制", "Quality control"],
  "dashboard.qcTitle": ["质检报告", "QC reports"],
  "dashboard.qcSubtitle": ["只有质检报告可以打开任务数据进行复核。", "Only QC reports open task data for review."],
  "dashboard.plans": ["个计划", "plans"],
  "dashboard.episodes": ["条片段", "episodes"],
  "dashboard.frames": ["帧", "frames"],
  "dashboard.robots": ["个机器人", "robots"],
  "dashboard.pending": ["待处理", "pending"],
  "dashboard.remaining": ["还差", "remaining"],
  "dashboard.unknownRobot": ["未知机器人", "Unknown robot"],
  "dashboard.noRobotPlans": ["暂无机器人计划", "No robot plans"],
  "dashboard.openQc": ["查看质检", "Open QC"],
  "dashboard.noReports": ["暂无质检报告", "No QC reports"],
  "dashboard.pendingSummary": ["待处理 0", "0 pending"],
  "ui.requestFailed": ["请求失败", "Request failed"],
  "ui.chooseBatchExport": ["选择批次后可导出", "Choose a batch to export"],
  "ui.saved": ["已保存", "saved"],
  "ui.confirmDelete": ["确认删除", "Confirm deletion"],
  "ui.delete": ["删除", "Delete"],
  "ui.deleteQuestion": ["？被引用的记录无法删除。", "? Referenced records cannot be deleted."],
  "ui.deleted": ["已删除", "deleted"],
  "ui.importPlan": ["导入计划", "Import plan"],
  "ui.importPrefix": ["导入 ", "Import "],
  "ui.importMiddle": [" 将替换批次 ", " will replace the four plan files in batch "],
  "ui.import": ["导入", "Import"],
  "ui.planImported": ["计划压缩包已导入", "Plan ZIP imported"],
  "ui.photosUploaded": ["物体照片已上传", "Object photos uploaded"],
  "ui.objectsImported": ["物体和照片已批量导入", "Objects and photos imported"],
  "actions.cancel": ["取消", "Cancel"],
  "actions.confirm": ["确认", "Confirm"],
  "actions.close": ["关闭", "Close"],
  "qc.previousSlot": ["上一个槽位", "Previous slot"],
  "qc.nextSlot": ["下一个槽位", "Next slot"],
  "entity.noEpisodes": ["该任务尚未配置数据槽位", "This task has no episode slots"],
  "entity.noSceneMatches": ["没有匹配的场景", "No matching scenes"],
  "entity.noScenes": ["当前批次没有场景", "No scenes in the current batch"],
  "entity.groups": ["组", "groups"],
  "entity.noObjectMatches": ["没有匹配的物体", "No matching objects"],
  "entity.noObjects": ["还没有物体资产", "No object assets yet"],
  "entity.noEnglishName": ["未填写英文名", "English name not set"],
  "entity.photos": ["图", "photos"],
  "entity.delete": ["删除", "Delete"],
  "entity.saveChanges": ["保存更改", "Save changes"],
  "entity.chooseScene": ["选择或新增场景", "Select or add a scene"],
  "entity.newScene": ["新场景", "New scene"],
  "entity.sceneEditor": ["场景编辑器", "Scene editor"],
  "entity.placementGroups": ["摆放分组", "Placement groups"],
  "entity.objectsPositions": ["物体与位置", "Objects and positions"],
  "entity.add": ["添加", "Add"],
  "entity.placeHint": ["添加物体后，在左侧九宫格选择位置", "Add an object, then choose positions in the grid"],
  "entity.cameraPosition": ["相机位置", "Camera position"],
  "entity.notConfigured": ["未配置", "Not configured"],
  "entity.external": ["外置", "External"],
  "entity.removePlacement": ["移除摆放分组", "Remove placement group"],
  "entity.choosePosition": ["未选择位置", "No position selected"],
  "entity.chooseGroup": ["先添加或选择一个物体组", "Add or select an object group first"],
  "entity.chooseTask": ["选择任务，展开查看数据槽位", "Select a task to view episode slots"],
  "entity.taskEditor": ["任务编辑器", "Task editor"],
  "entity.operationObjects": ["操作物体", "Operation objects"],
  "entity.addObject": ["添加物体...", "Add object..."],
  "entity.noOperationObjects": ["未设置操作物体", "No operation objects set"],
  "entity.sceneCoverage": ["场景与采集数量", "Scene coverage and collection count"],
  "entity.addScene": ["添加场景...", "Add scene..."],
  "entity.noConfiguredScenes": ["未配置场景", "No scenes configured"],
  "entity.totalEpisodeSlots": ["总槽位数", "Total episode slots"],
  "entity.chooseObject": ["选择或新增物体", "Select or add an object"],
  "entity.newObject": ["新物体", "New object"],
  "entity.objectAsset": ["共享物体资产", "Shared object asset"],
  "entity.objectPhotos": ["实物照片", "Object photos"],
  "entity.uploadPhotos": ["上传照片", "Upload photos"],
  "entity.scanStatus": ["扫描状态", "Scan status"],
  "entity.color": ["颜色", "Color"],
  "entity.photoDirectory": ["照片目录", "Photo directory"],
  "entity.noObject": ["未找到物体", "Object not found"],
  "entity.noName": ["未填写名称", "Name not set"],
  "entity.unlabeled": ["未标注", "Unlabeled"],
  "entity.unbound": ["未绑定", "Unbound"],
  "entity.noPlanIssues": ["当前范围未发现计划问题", "No plan issues in the current scope"],
  "entity.noScene": ["未找到场景", "Scene not found"],
  "entity.chooseBatch": ["请先在质检左栏选择一个计划批次", "Choose a plan batch in the QC panel first"],
  "entity.sceneIdRequired": ["场景编号不能为空", "Scene ID cannot be empty"],
  "entity.placementRequired": ["每个物体组都需要物体和至少一个位置", "Each object group needs an object and at least one position"],
  "entity.promptRequired": ["任务编号和英文提示词不能为空", "Task ID and English prompt cannot be empty"],
  "entity.sceneRequired": ["任务至少需要一个场景", "A task needs at least one scene"],
  "entity.objectNameRequired": ["至少填写一种物体名称", "At least one object name is required"],
  "entity.sceneId": ["场景编号", "Scene ID"],
  "entity.taskId": ["任务编号", "Task ID"],
  "entity.action": ["动作", "Action"],
  "entity.category": ["分类", "Category"],
  "entity.englishPrompt": ["英文提示词", "English prompt"],
  "entity.chinesePrompt": ["中文提示词", "Chinese prompt"],
  "entity.objectId": ["物体编号", "Object ID"],
  "entity.chineseName": ["中文名称", "Chinese name"],
  "entity.collectedSummary": ["已采集 / 总数", "Collected / total"],
  "entity.newTask": ["新任务", "New task"],
  "entity.operationObjectsTitle": ["操作物体", "Operation objects"],
  "entity.sceneCoverageTitle": ["场景与采集数量", "Scene coverage and collection count"],
  "entity.physicalPhotos": ["实物照片", "Object photos"],
  "entity.valueNotSet": ["未填写", "Not set"],
  "entity.valueUnlabeled": ["未标注", "Unlabeled"],
  "entity.valueUnbound": ["未绑定", "Unbound"],
  "common.episodes": ["片段", "episodes"],
  "common.photoSlot": ["照片槽位", "Photo slot"],
  "common.englishName": ["英文名称", "English name"],
  "review.loading3d": ["正在载入三维模型", "Loading 3D"],
  "review.fetchingMeshes": ["正在读取机器人模型", "Fetching robot meshes"],
  "review.taskDescription": ["任务描述", "Task description"],
  "review.compareRobots": ["机器人对比", "Compare robots"],
  "review.compareTitle": ["机器人视频对比", "Robot video comparison"],
  "review.compareLoading": ["正在准备机器人视频", "Preparing robot videos"],
  "review.compareEmpty": ["暂无可对比的视频数据", "No video data available for comparison"],
  "review.compareHint": ["按任务、场景和轮次对齐播放时间", "Playback is aligned by task, scene, and round"],
  "review.comparePlayAll": ["全部播放", "Play all"],
  "review.comparePauseAll": ["全部暂停", "Pause all"],
  "review.robot": ["机器人", "Robot"],
  "review.scene": ["场景", "Scene"],
  "review.action": ["动作", "Action"],
  "review.state": ["状态", "State"],
  "review.noDimensions": ["没有可用维度", "No dimensions"],
  "review.awaitingData": ["等待数据", "Awaiting data"],
  "review.unset": ["未设置", "Unset"],
  "review.failureReasonRequired": ["请先选择未通过的主要原因", "Select a primary failure reason first"],
  "review.otherReasonRequired": ["选择其他时请填写具体原因", "Describe the other reason before saving"],
  "review.otherReasonPlaceholder": ["请填写其他原因", "Describe the other reason"],
};
const app = {
  state: null,
  tab: "dataset",
  locale: document.documentElement.dataset.locale === "zh" ? "zh" : "en",
  batch: "",
  robot: "",
  selected: { scenes: "", tasks: "", objects: "" },
  draft: { scenes: null, tasks: null, objects: null },
  original: { scenes: "", tasks: "", objects: "" },
  expandedTask: "",
  qcSceneFilter: "",
  qcTaskFilter: "",
  qcPage: 0,
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
  objectThumbObserver: null,
  stateCache: new Map(),
  stateRequests: new Map(),
  reviewCache: new Map(),
  qcLoadingToken: 0,
};

function t(key) {
  const values = LOCALE_TABLE[key];
  return values ? values[app.locale === "zh" ? 0 : 1] : key;
}

function qcState(slot) {
  if (slot && slot.qc_state) return slot.qc_state;
  if (!slot || slot.state === "pending") return "pending";
  if (slot.state === "repair") return "failed";
  return slot.episode && slot.episode.qc_verdict === "pass" ? "passed" : "unreviewed";
}

function applyLocale() {
  document.documentElement.lang = app.locale === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((element) => {
    element.title = t(element.dataset.i18nTitle);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });
}

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

function setQcLoading(loading, token = app.qcLoadingToken) {
  if (token !== app.qcLoadingToken) return;
  const indicator = $("qc-loading-indicator");
  if (indicator) indicator.hidden = !loading;
  const panel = $("collection-qc-panel");
  if (panel) {
    panel.classList.toggle("is-loading", loading);
    panel.setAttribute("aria-busy", String(loading));
  }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && !["GET", "HEAD"].includes(options.method)) {
    headers.set("X-EVA-Dataset-Editor", "1");
    headers.set("X-EVA-Edit-Mode", "1");
  }
  const response = await fetch(path, { ...options, headers });
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error((payload && payload.error) || t("ui.requestFailed") + " (" + response.status + ")");
  }
  return payload;
}

const STATE_STORAGE_PREFIX = "eva-dataset-state::";

function readStoredState(cacheKey) {
  const storageKey = STATE_STORAGE_PREFIX + cacheKey;
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return null;
    const stored = JSON.parse(raw);
    return stored && stored.state ? stored.state : null;
  } catch (error) {
    localStorage.removeItem(storageKey);
    return null;
  }
}

function storeState(cacheKey, state) {
  try {
    localStorage.setItem(STATE_STORAGE_PREFIX + cacheKey, JSON.stringify({
      timestamp: Date.now(),
      state,
    }));
  } catch (error) {
    // Browser storage is an optional acceleration layer.
  }
}

function writeJson(path, method, payload) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function runWrite(control, operation, successMessage) {
  if (app.writeBusy || (control && control.disabled)) return false;
  app.writeBusy = true;
  app.stateCache.clear();
  app.reviewCache.clear();
  if (control) control.disabled = true;
  try {
    await operation();
    showToast(successMessage);
    return true;
  } catch (error) {
    showToast(error.message || String(error), true);
    return false;
  } finally {
    app.writeBusy = false;
    if (control) control.disabled = false;
  }
}

async function loadState(preserveReview = false, force = false) {
  showGlobalError();
  const loadingToken = ++app.qcLoadingToken;
  setQcLoading(true, loadingToken);
  const params = new URLSearchParams();
  if (app.batch) params.set("batch", app.batch);
  if (app.robot) params.set("robot_type", app.robot);
  const query = params.toString() ? "?" + params : "";
  const cacheKey = (app.robot || "*") + "::" + (app.batch || "*");
  const memory = app.stateCache.get(cacheKey);
  const cached = memory || {
    timestamp: 0,
    state: readStoredState(cacheKey),
  };
  const hasCachedState = Boolean(cached.state);
  if (!force && hasCachedState) {
    app.stateCache.set(cacheKey, cached);
    applyState(cached.state, preserveReview);
    setQcLoading(false, loadingToken);
    $("loading-state").hidden = true;
    $("app").hidden = false;
    requestAnimationFrame(moveTabThumb);
  }
  let request = app.stateRequests.get(cacheKey);
  if (!request || force) {
    request = api("/api/state" + query);
    app.stateRequests.set(cacheKey, request);
  }
  try {
    const state = await request;
    if (app.stateRequests.get(cacheKey) === request) app.stateRequests.delete(cacheKey);
    app.stateCache.set(cacheKey, { timestamp: Date.now(), state });
    storeState(cacheKey, state);
    applyState(state, preserveReview);
    setQcLoading(false, loadingToken);
    $("loading-state").hidden = true;
    $("app").hidden = false;
    requestAnimationFrame(moveTabThumb);
  } catch (error) {
    if (app.stateRequests.get(cacheKey) === request) app.stateRequests.delete(cacheKey);
    setQcLoading(false, loadingToken);
    $("loading-state").hidden = true;
    if (!hasCachedState) showGlobalError(error.message || String(error));
  }
}

function applyState(state, preserveReview = false) {
  app.state = state;
  retainSelections();
  renderBatchSelect();
  renderRobotSelect();
  updateSummary();
  if (app.tab === "tasks") {
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
  select.replaceChildren(new Option(t("filters.chooseBatch"), ""));
  const batches = app.state.batches;
  for (const batch of batches) {
    const label = batch.batch_kind === "unmatched"
      ? "unmatched · " + batch.robot_type
      : batch.batch_id + " · " + batch.robot_type;
    select.add(new Option(label, batch.batch_id));
  }
  select.value = app.batch;
  const currentIndex = batches.findIndex((batch) => batch.batch_id === app.batch);
  const previous = $("batch-prev");
  const next = $("batch-next");
  if (previous) previous.disabled = !batches.length || currentIndex === 0;
  if (next) next.disabled = !batches.length || currentIndex === batches.length - 1;
  const url = app.batch
    ? "/api/batches/" + encodeURIComponent(app.batch) + "/export"
    : "";
  for (const link of [$('export-trigger')].filter(Boolean)) {
    link.href = url || "#";
    link.classList.toggle("disabled", !url);
    link.setAttribute("aria-disabled", String(!url));
  }
  const publish = $("publish-task-set");
  if (publish) publish.disabled = !app.batch;
  for (const id of ["hf-publish-task-set", "hf-sync-task-set"]) {
    const control = $(id);
    if (control) control.disabled = !app.batch;
  }
  const download = $("hf-download-dataset");
  if (download) download.disabled = !app.batch;
  for (const id of ["hf-upload-qc", "hf-download-qc"]) {
    const control = $(id);
    if (control) control.disabled = !app.batch;
  }
}

async function runHfSyncAction(path, message) {
  const status = $("hf-sync-status");
  const fill = $("hf-sync-progress-fill");
  const bar = $("hf-sync-progress");
  if (status) status.textContent = `${t(message)} · ${t("sync.running")}`;
  if (bar) { bar.classList.add("indeterminate"); bar.removeAttribute("aria-valuenow"); }
  try {
    const result = await api(path, { method: "POST" });
    if (fill) fill.style.width = "100%";
    if (bar) { bar.classList.remove("indeterminate"); bar.setAttribute("aria-valuenow", "100"); }
    if (status) status.textContent = `${t(message)} · ${t("sync.done")}`;
    await loadState(false, true);
  } catch (error) {
    if (status) status.textContent = `${t("sync.failed")}: ${error.message}`;
    if (fill) fill.style.width = "0%";
    if (bar) { bar.classList.remove("indeterminate"); bar.setAttribute("aria-valuenow", "0"); }
  }
}

function renderRobotSelect() {
  const select = $("robot-select");
  select.replaceChildren(new Option(t("filters.allRobots"), ""));
  for (const robot of app.state.robot_types || []) {
    select.add(new Option(robot, robot));
  }
  select.value = app.robot;
}

function updateSummary() {
  const detailsLoaded = Boolean(app.batch && app.state.plans.length);
  const slots = detailsLoaded
    ? app.state.tasks.reduce((sum, task) => sum + task.slots.length, 0)
    : app.state.batches.reduce((sum, batch) => sum + Number(batch.episodes || 0), 0);
  const taskCount = detailsLoaded
    ? app.state.tasks.length
    : app.state.batches.reduce((sum, batch) => sum + Number(batch.tasks || 0), 0);
  const collected = detailsLoaded
    ? app.state.tasks.reduce(
      (sum, task) => sum + task.slots.filter((slot) => qcState(slot) !== "pending").length,
      0,
    )
    : app.state.batches.reduce((sum, batch) => sum + Number(batch.collected || 0), 0);
  $("dataset-path").textContent = app.state.plans_root;
  $("dataset-dot").className = "status-dot " + (app.state.issues.length ? "warning" : "ready");
  $("task-count").textContent = taskCount;
  $("object-count").textContent = app.state.objects.length;
  $("issue-count").textContent = app.state.issues.length;
  const metricBatches = $("metric-batches");
  if (metricBatches) metricBatches.textContent = app.state.batches.length;
  const metricTasks = $("metric-tasks");
  if (metricTasks) metricTasks.textContent = taskCount;
  const metricObjects = $("metric-objects");
  if (metricObjects) metricObjects.textContent = app.state.objects.length;
  const metricEpisodes = $("metric-episodes");
  if (metricEpisodes) metricEpisodes.textContent = collected + " / " + slots;
  for (const [id, value] of [["collected-count", collected], ["slot-total-count", slots], ["pending-count", Math.max(0, slots - collected)]]) {
    const element = $(id);
    if (element) element.textContent = value;
  }
  const pending = detailsLoaded
    ? app.state.tasks.reduce(
      (sum, task) => sum + task.slots.filter((slot) => ["pending", "failed"].includes(slot.state)).length,
      0,
    )
    : null;
  const qcExport = $("qc-export");
  if (qcExport) {
    const url = app.batch ? "/api/batches/" + encodeURIComponent(app.batch) + "/qc/export" : "#";
    qcExport.href = url;
    qcExport.classList.toggle("disabled", !app.batch);
    qcExport.setAttribute("aria-disabled", String(!app.batch));
  }
  const qcPending = $("qc-pending-count");
  if (qcPending) qcPending.textContent = pending == null ? t("ui.chooseBatchExport") : pending + " " + t("qc.pendingSummary");
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

async function saveEntity(kind, control) {
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
  return runWrite(control, async () => {
    await writeJson(path, current ? "PUT" : "POST", draft);
    app.original[kind] = id;
    app.selected[kind] = kind === "objects" ? "::" + id : (batch || "") + "::" + id;
    await loadState();
    const saved = app.state[kind].find((item) => recordKey(item, idField) === app.selected[kind]);
    app.draft[kind] = clone(saved);
    renderCurrentEditor();
  }, id + " " + t("ui.saved"));
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
  const id = app.original[kind];
  if (!id || !await confirmAction(t("ui.confirmDelete"), t("ui.importPrefix") + id + t("ui.deleteQuestion"), t("ui.delete"))) {
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
  }, id + " " + t("ui.deleted"));
}

async function importPlan(file, control) {
  if (!file || !requireBatch()) return;
  const confirmed = await confirmAction(
    t("ui.importPlan"),
    t("ui.importPrefix") + file.name + t("ui.importMiddle") + app.batch + ".",
    t("ui.import"),
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
  }, t("ui.planImported"));
  $("import-file").value = "";
}

async function uploadPhotos(files) {
  if (!files.length) return;
  if (!app.original.objects) {
    await saveEntity("objects", null);
    if (!app.original.objects) return;
  }
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
  }, t("ui.photosUploaded"));
  $("photo-file").value = "";
}

function openPhotoPicker() {
  if (!app.draft.objects) return;
  $("photo-file").click();
}

async function importObjects(files, control) {
  const selected = [...files];
  const csvFiles = selected.filter((file) => file.name.toLowerCase().endsWith(".csv"));
  const csv = csvFiles[0];
  if (csvFiles.length !== 1) {
    showToast("请选择对象 CSV 文件", true);
    return;
  }
  const form = new FormData();
  form.append("file", csv);
  selected.filter((file) => file !== csv).forEach((file) => form.append("photos", file));
  await runWrite(control, async () => {
    await api("/api/objects/import", { method: "POST", body: form });
    await loadState();
  }, t("ui.objectsImported"));
  $("object-import-file").value = "";
}

function clearEditors() {
  stopPlayback();
  app.selected = { scenes: "", tasks: "", objects: "" };
  app.original = { scenes: "", tasks: "", objects: "" };
  app.draft = { scenes: null, tasks: null, objects: null };
  app.expandedTask = "";
  app.qcSceneFilter = "";
  app.qcTaskFilter = "";
  app.qcPage = 0;
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
    else if (action === "import") {
      if (requireBatch()) $("import-file").click();
    }
  } catch (error) {
    showToast(error.message || String(error), true);
  }
}

function bindEvents() {
  document.addEventListener("click", handleAction);
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
  $("batch-select").addEventListener("change", async (event) => {
    app.batch = event.target.value;
    clearEditors();
    await loadState();
  });
  const navigateBatch = async (offset) => {
    const batches = app.state ? app.state.batches : [];
    if (!batches.length) return;
    const currentIndex = batches.findIndex((batch) => batch.batch_id === app.batch);
    const nextIndex = currentIndex < 0
      ? (offset > 0 ? 0 : batches.length - 1)
      : currentIndex + offset;
    if (nextIndex < 0 || nextIndex >= batches.length) return;
    app.batch = batches[nextIndex].batch_id;
    clearEditors();
    await loadState();
  };
  $("batch-prev").addEventListener("click", () => navigateBatch(-1));
  $("batch-next").addEventListener("click", () => navigateBatch(1));
  $("robot-select").addEventListener("change", async (event) => {
    app.robot = event.target.value;
    app.batch = "";
    clearEditors();
    await loadState();
  });
  $("qc-scene-select").addEventListener("change", (event) => {
    app.qcSceneFilter = event.target.value;
    renderTaskList();
  });
  $("qc-task-select").addEventListener("change", (event) => {
    app.qcTaskFilter = event.target.value;
    renderTaskList();
  });
  $("qc-prev-slot").addEventListener("click", () => { app.qcPage = Math.max(0, (Number(app.qcPage) || 0) - 1); renderTaskList(); });
  $("qc-next-slot").addEventListener("click", () => { app.qcPage += 1; renderTaskList(); });
  $("object-search").addEventListener("input", renderObjectList);
  $("object-photo-filter").addEventListener("change", renderObjectList);
  $("object-modeling-filter").addEventListener("change", renderObjectList);
  $("retry-button").addEventListener("click", () => loadState());
  $("refresh-button").addEventListener("click", () => loadState(false, true));
  $("publish-task-set").addEventListener("click", async () => {
    if (!app.batch) return requireBatch();
    const control = $("publish-task-set");
    control.disabled = true;
    try {
      const result = await api("/api/batches/" + encodeURIComponent(app.batch) + "/hf/publish", {method: "POST"});
      showToast(result.task_set + " @ " + result.revision);
    } catch (error) {
      showToast(error.message, true);
    } finally {
      control.disabled = !app.batch;
    }
  });
  $("hf-publish-task-set").addEventListener("click", () => {
    if (app.batch) runHfSyncAction(`/api/batches/${encodeURIComponent(app.batch)}/hf/publish`, "actions.publishTaskSet");
  });
  $("hf-sync-assets").addEventListener("click", () => runHfSyncAction("/api/hf/assets/publish", "sync.assets"));
  $("hf-download-dataset").addEventListener("click", () => {
    if (app.batch) runHfSyncAction(`/api/batches/${encodeURIComponent(app.batch)}/hf/download`, "sync.dataset");
  });
  $("hf-upload-qc").addEventListener("click", () => {
    if (app.batch) runHfSyncAction(`/api/batches/${encodeURIComponent(app.batch)}/hf/qc/upload`, "sync.uploadQc");
  });
  $("import-trigger").addEventListener("click", () => {
    if (requireBatch()) $("import-file").click();
  });
  $("import-file").addEventListener("change", (event) => {
    importPlan(event.target.files[0], $("import-trigger"));
  });
  $("photo-file").addEventListener("change", (event) => uploadPhotos(event.target.files));
  document.addEventListener("object-photo-upload", () => openPhotoPicker());
  $("object-import-trigger").addEventListener("click", () => $("object-import-file").click());
  $("object-import-file").addEventListener("change", (event) => importObjects(event.target.files, $("object-import-trigger")));
  document.querySelector("[data-close-object]").addEventListener("click", () => {
    $("object-dialog").close();
  });
  document.querySelector("[data-close-robot-compare]").addEventListener("click", () => {
    $("robot-compare-dialog").close();
  });
  for (const link of [$('export-trigger'), $("qc-export")].filter(Boolean)) {
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
  emptyList,
  switchTab,
  stopPlayback,
  openSlot,
  translate: t,
});
configureReview({
  app,
  node,
  button,
  input,
  textarea,
  clone,
  encodePath,
  recordKey,
  api,
  writeJson,
  runWrite,
  loadState,
  renderTaskList,
  renderTaskEditor,
  sceneFor,
  renderGridCells,
  navigateQcSlot,
  showToast,
  translate: t,
});

applyLocale();
bindEvents();
switchTab(app.tab);
// Warm the shared 3D module while the initial catalog request is in flight.
// Review clicks then only wait for the selected episode data.
loadThree().catch(() => {});
loadState();
