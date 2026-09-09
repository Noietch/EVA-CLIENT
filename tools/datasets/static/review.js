import { loadThree, RobotViewer } from "./robot-viewer.js";

const COLORS = ["#e8590c", "#1f1e1c", "#2563eb", "#2f9e44", "#d6336c", "#0b7285"];
const $ = (id) => document.getElementById(id);
let app;
let node;
let button;
let input;
let clone;
let encodePath;
let recordKey;
let api;
let writeJson;
let runWrite;
let loadState;
let renderTaskList;
let renderTaskEditor;
let jumpScene;
let objectFor;
let openObjectDialog;
let sceneFor;
let renderGridCells;
let requireEditMode;

function configureReview(context) {
  ({
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
  } = context);
}

function stopPlayback() {
  app.playing = false;
  cancelAnimationFrame(app.playbackTimer);
  document.querySelectorAll(".cam-strip video").forEach((video) => video.pause());
  const control = $("review-play");
  if (control) control.textContent = "▶";
}


async function openSlot(task, slot) {
  stopPlayback();
  app.selected.tasks = recordKey(task, "task_id");
  app.original.tasks = task.task_id;
  app.draft.tasks = clone(task);
  app.selectedSlot = slot.slot_id;
  renderTaskList();
  renderReviewLoading(task, slot);
  const token = ++app.reviewToken;
  try {
    const query = new URLSearchParams({
      batch: task.batch_id,
      slot_id: slot.slot_id,
    });
    const [payload] = await Promise.all([
      api("/api/review?" + query),
      loadThree(),
    ]);
    if (token !== app.reviewToken) return;
    app.review = payload;
    renderReview(task, slot, payload);
  } catch (error) {
    if (token === app.reviewToken) {
      $("task-editor").replaceChildren(node("div", "empty-state", error.message));
    }
  }
}

function renderReviewLoading(task, slot) {
  const host = $("task-editor");
  const shell = node("div", "editor-shell");
  const head = reviewHeader(task, slot);
  const loading = node("div", "loading-state");
  loading.append(node("span", "spinner"), node("strong", "", "正在载入 slot 数据"));
  shell.append(head, loading);
  host.replaceChildren(shell);
}

function reviewHeader(task, slot) {
  const head = node("header", "editor-head review-head");
  const heading = node("div");
  heading.append(
    node("span", "eyebrow", task.batch_id + " · " + slot.slot_id),
    node("h1", "", task.task_id + " / " + slot.scene_id),
  );
  const actions = node("div", "editor-actions");
  const scene = button("查看场景", "", "button");
  scene.addEventListener("click", () => jumpScene(task.batch_id, slot.scene_id));
  const edit = button(app.editMode ? "编辑任务" : "查看任务", "", "button");
  edit.addEventListener("click", () => {
    app.selectedSlot = "";
    renderTaskList();
    renderTaskEditor();
  });
  actions.append(scene, edit);
  head.append(heading, actions);
  return head;
}

function renderReview(task, slot, payload) {
  if (app.robotViewer) app.robotViewer.dispose();
  const host = $("task-editor");
  const shell = node("div", "editor-shell");
  const body = node("div", "review-body");
  const context = buildReviewContext(task, slot);
  const stage = buildQcStage(task, slot, payload);
  body.append(context, stage);
  shell.append(reviewHeader(task, slot), body);
  host.replaceChildren(shell);
  app.robotViewer = new RobotViewer($("review-gl"), $("robot-empty"));
  const episodeIndex = slot.episode ? Number(slot.episode.episode_index) : null;
  app.robotViewer.load(payload.robot_type, payload, episodeIndex);
  requestAnimationFrame(() => {
    renderChartControls();
    drawReviewCharts(0);
  });
}

function buildReviewContext(task, slot) {
  const section = node("section", "review-context");
  const taskCopy = node("div", "review-task-copy");
  taskCopy.append(
    node("span", "eyebrow", "CURRENT TASK"),
    node("strong", "", task.prompt_zh || task.prompt_en),
    node("small", "", task.prompt_en),
  );
  const objectRow = node("div", "object-chip-row");
  for (const objectId of task.operation_object_ids) {
    const asset = objectFor(objectId);
    const chip = button(
      asset ? asset.object_name_zh || asset.object_name : objectId,
      "",
      "object-chip",
    );
    chip.addEventListener("click", () => openObjectDialog(objectId));
    objectRow.append(chip);
  }
  const status = node("div", "slot-status");
  status.append(
    node("span", "", "ROUND " + (slot.round_index + 1) + " / " + slot.round_total),
    node("b", "", slot.episode ? "EPISODE " + slot.episode.episode_index : "EMPTY SLOT"),
  );
  section.append(taskCopy, objectRow, status);
  return section;
}

function buildQcStage(task, slot, payload) {
  const stage = node("section", "qc-stage" + (payload.series ? "" : " no-series"));
  const top = node("div", "stage-top");
  const visual = node("div", "stage-visual-stack");
  visual.append(buildReadOnlyScene(task.batch_id, slot.scene_id));
  const robot = node("div", "col-canvas");
  robot.append(
    node("span", "canvas-tag", "3D · URDF · " + (payload.robot_type || "UNSET")),
    node("canvas", "robot-canvas"),
  );
  robot.querySelector("canvas").id = "review-gl";
  const empty = node("div", "canvas-empty");
  empty.id = "robot-empty";
  empty.append(node("b", "", "LOADING 3D"), node("small", "", "fetching robot meshes"));
  robot.append(empty);
  visual.append(robot);
  top.append(visual, buildCameraStrip(task.batch_id, slot, payload));
  stage.append(top, buildChartRow(), buildScrubber(payload), buildQcControls(task, slot));
  return stage;
}

function buildReadOnlyScene(batch, sceneId) {
  const card = node("div", "review-scene-card");
  const head = node("div", "review-scene-head");
  const link = button(sceneId, "", "text-button");
  link.addEventListener("click", () => jumpScene(batch, sceneId));
  head.append(node("span", "eyebrow", "SCENE"), link);
  const grid = node("div", "scene-grid compact");
  const scene = sceneFor(batch, sceneId);
  renderGridCells(grid, batch, scene, true);
  card.append(head, grid);
  return card;
}

function buildCameraStrip(batch, slot, payload) {
  const strip = node("div", "cam-strip");
  if (!payload.videos.length) {
    strip.append(node("div", "cam-empty", slot.episode ? "未发现相机视频" : "EMPTY SLOT · 等待采集"));
    return strip;
  }
  for (const video of payload.videos) {
    const cell = node("div", "cam-cell");
    cell.append(node("span", "cam-label", video.label));
    const media = node("video", "cam");
    media.preload = "auto";
    media.muted = true;
    media.playsInline = true;
    media.src = "/api/batches/" + encodeURIComponent(batch)
      + "/episodes/" + slot.episode.episode_index
      + "/video/" + encodePath(video.key);
    cell.append(media);
    strip.append(cell);
  }
  return strip;
}

function buildChartRow() {
  const row = node("div", "stage-charts");
  row.append(buildChart("action", "ACTION"), buildChart("state", "STATE"));
  return row;
}

function buildChart(kind, label) {
  const panel = node("div", "stage-chart");
  const head = node("div", "chart-head");
  const title = node("span", "chart-title", label);
  title.append(node("small", "", " t · x"));
  const dims = node("div", "chart-dims");
  dims.id = "review-" + kind + "-dims";
  const all = button("ALL", "", "chart-allbtn");
  all.addEventListener("click", () => toggleAllDims(kind));
  head.append(title, dims, all);
  const body = node("div", "chart-body");
  const canvas = node("canvas");
  canvas.id = "review-" + kind + "-chart";
  body.append(canvas);
  panel.append(head, body);
  return panel;
}

function buildScrubber(payload) {
  const bar = node("div", "stage-scrub");
  const play = button("▶", "", "scrub-play");
  play.id = "review-play";
  const label = node("span", "scrub-state", payload.series ? "REVIEW" : "PENDING");
  const range = input("range");
  range.id = "review-range";
  const length = payload.series ? payload.series.state.length : 0;
  range.min = "0";
  range.max = String(Math.max(0, length - 1));
  range.value = "0";
  range.disabled = !length;
  const position = node("span", "scrub-pos", length ? "1 / " + length : "0 / 0");
  position.id = "review-position";
  const time = node("span", "scrub-time", "0.0s");
  time.id = "review-time";
  play.disabled = !length;
  play.addEventListener("click", togglePlayback);
  range.addEventListener("input", () => seekReview(Number(range.value)));
  bar.append(play, label, range, position, time);
  return bar;
}

function buildQcControls(task, slot) {
  const section = node("div", "qc-controls");
  if (!slot.episode) {
    section.append(node("span", "pending-copy", "计划 slot 已保留，采集后会自动按 slot_id 关联。"));
    return section;
  }
  if (!app.editMode) {
    section.classList.add("read-only");
    const verdict = slot.episode.qc_verdict === "pass"
      ? "通过"
      : slot.episode.qc_verdict === "fail" ? "返修" : "未标注";
    const summary = node("span", "qc-summary", "QC · " + verdict);
    const note = node("span", "qc-note-readonly", slot.episode.qc_note || "无备注");
    section.append(summary, note);
    return section;
  }
  const note = input("text", "", slot.episode.qc_note || "", "QC note");
  note.id = "qc-note";
  const pass = button("通过", "", "button qc-pass");
  const fail = button("返修", "", "button danger");
  pass.classList.toggle("selected", slot.episode.qc_verdict === "pass");
  fail.classList.toggle("selected", slot.episode.qc_verdict === "fail");
  pass.addEventListener("click", () => saveQc(task, slot, "pass", pass));
  fail.addEventListener("click", () => saveQc(task, slot, "fail", fail));
  section.append(node("span", "eyebrow", "QUALITY CONTROL"), note, pass, fail);
  return section;
}

async function saveQc(task, slot, verdict, control) {
  if (!requireEditMode()) return;
  await runWrite(control, async () => {
    await writeJson(
      "/api/batches/" + encodeURIComponent(task.batch_id)
        + "/episodes/" + slot.episode.episode_index + "/qc",
      "PUT",
      { verdict, note: $("qc-note").value.trim() },
    );
    await loadState(true);
    const freshTask = app.state.tasks.find(
      (item) => item.batch_id === task.batch_id && item.task_id === task.task_id,
    );
    const freshSlot = freshTask.slots.find((item) => item.slot_id === slot.slot_id);
    await openSlot(freshTask, freshSlot);
  }, verdict === "pass" ? "Episode 已通过" : "Episode 已标记返修");
}

function reviewSeries() {
  return app.review && app.review.series;
}

function renderChartControls() {
  const series = reviewSeries();
  if (!series) return;
  app.review.dims = app.review.dims || {
    action: series.action_names.map(() => true),
    state: series.state_names.map(() => true),
  };
  for (const kind of ["action", "state"]) {
    const host = $("review-" + kind + "-dims");
    const names = series[kind + "_names"];
    host.replaceChildren(...names.map((name, index) => {
      const chip = button(name || kind + " " + (index + 1), "", "dim");
      chip.classList.toggle("off", !app.review.dims[kind][index]);
      chip.style.borderLeftColor = COLORS[index % COLORS.length];
      chip.addEventListener("click", () => {
        app.review.dims[kind][index] = !app.review.dims[kind][index];
        renderChartControls();
        drawReviewCharts(Number($("review-range").value));
      });
      return chip;
    }));
  }
}

function toggleAllDims(kind) {
  const dims = app.review && app.review.dims && app.review.dims[kind];
  if (!dims) return;
  const turnOn = dims.some((value) => !value);
  app.review.dims[kind] = dims.map(() => turnOn);
  renderChartControls();
  drawReviewCharts(Number($("review-range").value));
}

function drawReviewCharts(cursor) {
  const series = reviewSeries();
  for (const kind of ["action", "state"]) {
    const canvas = $("review-" + kind + "-chart");
    if (!canvas) continue;
    drawSeriesChart(
      canvas,
      series ? series[kind] : [],
      series ? series.timestamp : [],
      series && app.review.dims ? app.review.dims[kind] : [],
      cursor,
    );
  }
}

function drawSeriesChart(canvas, matrix, timestamps, dimsOn, cursor) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const ratio = Math.min(window.devicePixelRatio, 2);
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const pad = 25;
  const dims = matrix.length
    ? matrix[0].map((_value, index) => index).filter((index) => dimsOn[index])
    : [];
  context.clearRect(0, 0, width, height);
  if (!matrix.length || !dims.length) {
    context.fillStyle = "rgba(31,30,28,.38)";
    context.font = "10px monospace";
    context.fillText(matrix.length ? "no dimensions" : "awaiting data", pad, height / 2);
    return;
  }
  let low = Infinity;
  let high = -Infinity;
  for (const row of matrix) {
    for (const dim of dims) {
      const value = Number(row[dim]);
      if (Number.isFinite(value)) {
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
    }
  }
  if (!Number.isFinite(low)) { low = -1; high = 1; }
  if (high - low < 1e-6) { high += 1; low -= 1; }
  const start = Number(timestamps[0]) || 0;
  const end = Number(timestamps[timestamps.length - 1]) || start + 1;
  const x = (time) => pad + (width - 2 * pad) * ((time - start) / Math.max(end - start, 1e-6));
  const y = (value) => height - pad - (height - 2 * pad) * ((value - low) / (high - low));
  drawChartAxes(context, width, height, pad, end - start, high, low);
  for (const dim of dims) {
    context.strokeStyle = COLORS[dim % COLORS.length];
    context.globalAlpha = 0.72;
    context.lineWidth = 1.4;
    context.beginPath();
    matrix.forEach((row, index) => {
      const px = x(Number(timestamps[index]) || start);
      const py = y(Number(row[dim]) || 0);
      if (index) context.lineTo(px, py); else context.moveTo(px, py);
    });
    context.stroke();
  }
  context.globalAlpha = 1;
  if (cursor >= 0 && cursor < timestamps.length) {
    const px = x(Number(timestamps[cursor]) || start);
    context.strokeStyle = "rgba(31,30,28,.55)";
    context.beginPath();
    context.moveTo(px, pad);
    context.lineTo(px, height - pad);
    context.stroke();
  }
}

function drawChartAxes(context, width, height, pad, elapsed, high, low) {
  context.strokeStyle = "rgba(31,30,28,.09)";
  context.fillStyle = "rgba(31,30,28,.45)";
  context.font = "9px monospace";
  context.fillText(high.toFixed(2), 2, pad + 3);
  context.fillText(low.toFixed(2), 2, height - pad);
  context.textAlign = "center";
  for (const ratio of [0, 0.5, 1]) {
    const px = pad + (width - 2 * pad) * ratio;
    context.beginPath();
    context.moveTo(px, pad);
    context.lineTo(px, height - pad);
    context.stroke();
    context.fillText((elapsed * ratio).toFixed(1) + "s", px, height - 4);
  }
  context.textAlign = "start";
}

function seekReview(frame, syncVideos = true, updateCharts = true) {
  const series = reviewSeries();
  if (!series || !series.state.length) return;
  const index = Math.max(0, Math.min(Math.round(frame), series.state.length - 1));
  const range = $("review-range");
  range.value = String(index);
  range.style.setProperty(
    "--pct",
    (index / Math.max(series.state.length - 1, 1) * 100).toFixed(2) + "%",
  );
  $("review-position").textContent = (index + 1) + " / " + series.state.length;
  const elapsed = (Number(series.timestamp[index]) || 0) - (Number(series.timestamp[0]) || 0);
  $("review-time").textContent = elapsed.toFixed(1) + "s";
  if (syncVideos) {
    document.querySelectorAll(".cam-strip video").forEach((video) => {
      if (Math.abs(video.currentTime - elapsed) > 0.04) video.currentTime = elapsed;
    });
    if (app.playing) app.playbackStartedAt = performance.now() - elapsed * 1000;
  }
  if (updateCharts) drawReviewCharts(index);
  if (app.robotViewer) app.robotViewer.applyEpisodeFrame(index);
}

function frameForElapsed(timestamps, elapsed) {
  const start = Number(timestamps[0]) || 0;
  let low = 0;
  let high = timestamps.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if ((Number(timestamps[middle]) || start) - start <= elapsed) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

function togglePlayback() {
  const series = reviewSeries();
  if (!series || !series.state.length) return;
  if (app.playing) {
    stopPlayback();
    return;
  }
  let frame = Number($("review-range").value);
  if (frame >= series.state.length - 1) {
    frame = 0;
    seekReview(frame);
  }
  const elapsed = (Number(series.timestamp[frame]) || 0) - (Number(series.timestamp[0]) || 0);
  const videos = document.querySelectorAll(".cam-strip video");
  videos.forEach((video) => {
    if (Math.abs(video.currentTime - elapsed) > 0.04) video.currentTime = elapsed;
    video.play().catch(() => {});
  });
  app.playbackStartedAt = performance.now() - elapsed * 1000;
  app.playbackFrame = frame;
  app.playbackChartAt = 0;
  app.playbackVideoSyncAt = 0;
  app.playing = true;
  $("review-play").textContent = "Ⅱ";
  app.playbackTimer = requestAnimationFrame(playbackTick);
}

function playbackTick(timestamp) {
  if (!app.playing || !$("review-range")) return;
  const series = reviewSeries();
  const videos = [...document.querySelectorAll(".cam-strip video")];
  const master = videos.find((video) => !video.error);
  const start = Number(series.timestamp[0]) || 0;
  const duration = (Number(series.timestamp[series.timestamp.length - 1]) || start) - start
    + 1 / Math.max(Number(app.review.fps) || 30, 1);
  let elapsed = master ? master.currentTime : (timestamp - app.playbackStartedAt) / 1000;
  if ((master && master.ended) || elapsed >= duration) {
    elapsed = 0;
    app.playbackStartedAt = timestamp;
    videos.forEach((video) => {
      video.currentTime = 0;
      video.play().catch(() => {});
    });
  }
  if (master && timestamp - app.playbackVideoSyncAt >= 1000) {
    videos.slice(1).forEach((video) => {
      if (Math.abs(video.currentTime - elapsed) > 0.12) video.currentTime = elapsed;
    });
    app.playbackVideoSyncAt = timestamp;
  }
  const frame = frameForElapsed(series.timestamp, elapsed);
  if (frame !== app.playbackFrame) {
    const updateCharts = timestamp - app.playbackChartAt >= 100;
    seekReview(frame, false, updateCharts);
    app.playbackFrame = frame;
    if (updateCharts) app.playbackChartAt = timestamp;
  }
  app.playbackTimer = requestAnimationFrame(playbackTick);
}


export {
  configureReview,
  drawReviewCharts,
  openSlot,
  renderReview,
  stopPlayback,
};
