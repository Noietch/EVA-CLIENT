import { loadThree, RobotViewer } from "./robot-viewer.js";

const COLORS = ["#e8590c", "#1f1e1c", "#2563eb", "#2f9e44", "#d6336c", "#0b7285"];
const $ = (id) => document.getElementById(id);
function qcState(slot) {
  if (slot && slot.qc_state) return slot.qc_state;
  if (!slot) return "pending";
  if (slot.state === "repair") return "failed";
  if (slot.state === "pending") return "pending";
  return slot.episode && slot.episode.qc_verdict === "pass" ? "passed" : "unreviewed";
}

let app;
let node;
let button;
let input;
let textarea;
let clone;
let encodePath;
let recordKey;
let api;
let writeJson;
let runWrite;
let loadState;
let renderTaskList;
let renderTaskEditor;
let sceneFor;
let renderGridCells;
let navigateQcSlot;
let showToast;
let translate;
const STATIC_REASON = "static_frames_excessive";

function configureReview(context) {
  ({
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
    translate,
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
  const leftControls = $("qc-left-controls");
  if (leftControls) leftControls.replaceChildren(buildQcControls(task, slot));
  renderTaskList();
  renderReviewLoading(task, slot);
  const token = ++app.reviewToken;
  try {
    const query = new URLSearchParams({
      batch: task.batch_id,
      slot_id: slot.slot_id,
    });
    const reviewKey = task.batch_id + "::" + slot.slot_id;
    const cached = app.reviewCache.get(reviewKey);
    const payloadPromise = cached && Date.now() - cached.timestamp < 300000
      ? Promise.resolve(cached.payload)
      : api("/api/review?" + query).then((payload) => {
        app.reviewCache.set(reviewKey, { timestamp: Date.now(), payload });
        return payload;
      });
    const [payload] = await Promise.all([payloadPromise, loadThree()]);
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
  const head = reviewHeader(task);
  const loading = node("div", "loading-state");
  loading.append(node("span", "spinner"), node("strong", "", translate("qc.loadingSlot")));
  shell.append(head, loading);
  host.replaceChildren(shell);
}

function reviewHeader(task) {
  const head = node("header", "editor-head review-head");
  const heading = node("div");
  const promptZh = String(task.prompt_zh || "").trim();
  const promptEn = String(task.prompt_en || "").trim();
  const primary = app.locale === "zh" ? promptZh || promptEn : promptEn || promptZh;
  const secondary = app.locale === "zh" ? promptEn : promptZh;
  heading.append(
    node("span", "eyebrow", translate("review.taskDescription")),
    node("h1", "review-task-description", primary || task.task_id),
  );
  if (secondary && secondary !== primary) {
    heading.append(node("p", "review-task-description-secondary", secondary));
  }
  const compare = button(translate("review.compareRobots"), "", "button robot-compare-trigger");
  compare.addEventListener("click", () => openRobotCompare(task, app.selectedSlot));
  head.append(heading, compare);
  return head;
}

async function openRobotCompare(task, slotId) {
  stopPlayback();
  const dialog = $("robot-compare-dialog");
  const title = $("robot-compare-title");
  const body = $("robot-compare-body");
  if (!dialog || !title || !body) return;
  title.textContent = String(task.prompt_zh || task.prompt_en || task.task_id || translate("review.compareTitle"));
  body.replaceChildren(node("div", "compare-loading", translate("review.compareLoading")));
  const playAll = $("robot-compare-play-all");
  if (playAll) {
    playAll.textContent = translate("review.comparePlayAll");
    playAll.disabled = true;
    playAll.onclick = toggleComparePlayback;
  }
  dialog.showModal();
  try {
    const sourceSlot = (task.slots || []).find((item) => item.slot_id === slotId) || (task.slots || [])[0];
    const query = new URLSearchParams({
      task_id: task.task_id,
      scene_id: sourceSlot ? sourceSlot.scene_id : "",
      round_index: sourceSlot ? sourceSlot.round_index : 0,
      current_batch: app.batch || task.batch_id || "",
    });
    const result = await api("/api/compare?" + query);
    const groups = result.groups || [];
    const hasVideo = groups.filter((group) => group.payload && group.payload.videos && group.payload.videos.length).length;
    if (!groups.length || !hasVideo) {
      body.replaceChildren(node("div", "compare-empty", translate("review.compareEmpty")));
      return;
    }
    body.replaceChildren(...groups.map(buildCompareGroup));
    installCompareSync(body);
    if ($("robot-compare-play-all")) $("robot-compare-play-all").disabled = !body.querySelector("video");
  } catch (error) {
    body.replaceChildren(node("div", "compare-empty", error.message || translate("review.compareEmpty")));
  }
  dialog.onclose = () => {
    body.querySelectorAll("video").forEach((video) => video.pause());
    const playAll = $("robot-compare-play-all");
    if (playAll) playAll.textContent = translate("review.comparePlayAll");
  };
}

function toggleComparePlayback() {
  const dialog = $("robot-compare-dialog");
  if (!dialog) return;
  const videos = [...dialog.querySelectorAll("video")];
  if (!videos.length) return;
  const playing = videos.some((video) => !video.paused && !video.ended);
  if (playing) {
    videos.forEach((video) => video.pause());
    return;
  }
  const source = videos.find((video) => !video.ended) || videos[0];
  const elapsed = Number(source.currentTime) || 0;
  videos.forEach((video) => {
    if (video.ended || Math.abs(video.currentTime - elapsed) > 0.08) {
      video.currentTime = Math.min(elapsed, Number.isFinite(video.duration) ? video.duration : elapsed);
    }
    if (!video.ended) video.play().catch(() => {});
  });
}

function updateComparePlayButton() {
  const control = $("robot-compare-play-all");
  const dialog = $("robot-compare-dialog");
  if (!control || !dialog) return;
  const playing = [...dialog.querySelectorAll("video")].some((video) => !video.paused && !video.ended);
  control.textContent = translate(playing ? "review.comparePauseAll" : "review.comparePlayAll");
}

function buildCompareGroup(group) {
  const card = node("section", "robot-compare-group");
  const head = node("header", "robot-compare-group-head");
  head.append(node("strong", "", group.robot_type), node("small", "", group.batch_id));
  const views = node("div", "robot-compare-views");
  const videos = group.payload && group.payload.videos || [];
  if (!videos.length || !group.slot.episode) {
    views.append(node("div", "compare-empty compare-group-empty", translate("review.compareEmpty")));
  } else {
    for (const video of videos.slice(0, 3)) {
      const cell = node("div", "robot-compare-view");
      cell.append(node("span", "cam-label", video.label));
      const media = node("video", "compare-video");
      media.preload = "metadata";
      media.controls = true;
      media.muted = true;
      media.playsInline = true;
      const videoBatch = group.payload.video_batch || group.batch_id;
      const videoEpisode = group.payload.video_episode_index ?? group.slot.episode.episode_index;
      media.src = "/api/batches/" + encodeURIComponent(videoBatch)
        + "/episodes/" + videoEpisode
        + "/video/" + encodePath(video.key);
      cell.append(media);
      views.append(cell);
    }
  }
  card.append(head, views);
  return card;
}

function installCompareSync(container) {
  const videos = [...container.querySelectorAll(".compare-video")];
  let syncing = false;
  const syncPeers = (source, action) => {
    if (syncing) return;
    syncing = true;
    for (const peer of videos) {
      if (peer === source) continue;
      if (action === "time") {
        if (!peer.ended && Math.abs(peer.currentTime - source.currentTime) > 0.08) {
          peer.currentTime = Math.min(source.currentTime, Number.isFinite(peer.duration) ? peer.duration : source.currentTime);
        }
      } else if (action === "play") {
        if (!peer.ended) {
          peer.currentTime = Math.min(source.currentTime, Number.isFinite(peer.duration) ? peer.duration : source.currentTime);
          peer.play().catch(() => {});
        }
      } else if (action === "pause") {
        peer.pause();
      }
    }
    syncing = false;
  };
  videos.forEach((video) => {
    video.addEventListener("play", () => { syncPeers(video, "play"); updateComparePlayButton(); });
    video.addEventListener("pause", () => { if (!video.ended) syncPeers(video, "pause"); updateComparePlayButton(); });
    video.addEventListener("seeking", () => syncPeers(video, "time"));
    video.addEventListener("timeupdate", () => syncPeers(video, "time"));
  });
}

function renderReview(task, slot, payload) {
  if (app.robotViewer) app.robotViewer.dispose();
  const host = $("task-editor");
  const shell = node("div", "editor-shell");
  const body = node("div", "review-body");
  const stage = buildQcStage(task, slot, payload);
  body.append(stage);
  shell.append(reviewHeader(task), body);
  host.replaceChildren(shell);
  app.robotViewer = new RobotViewer($("review-gl"), $("robot-empty"));
  const episodeIndex = slot.episode ? Number(slot.episode.episode_index) : null;
  app.robotViewer.load(payload.robot_type, payload, episodeIndex);
  requestAnimationFrame(() => {
    renderChartControls();
    drawReviewCharts(0);
    autoStartPlayback();
  });
}

function buildQcStage(task, slot, payload) {
  const stage = node("section", "qc-stage" + (payload.series ? "" : " no-series"));
  const main = node("div", "qc-main-column");
  const top = node("div", "stage-top");
  const visual = node("div", "stage-visual-stack");
  visual.append(buildReadOnlyScene(task.batch_id, slot.scene_id));
  const robot = node("div", "col-canvas");
  robot.append(
    node("span", "canvas-tag", "3D · URDF · " + (payload.robot_type || translate("review.unset"))),
    node("canvas", "robot-canvas"),
  );
  robot.querySelector("canvas").id = "review-gl";
  const empty = node("div", "canvas-empty");
  empty.id = "robot-empty";
  empty.append(node("b", "", translate("review.loading3d")), node("small", "", translate("review.fetchingMeshes")));
  robot.append(empty);
  visual.append(robot);
  top.append(visual, buildCameraStrip(task.batch_id, slot, payload));
  main.append(top, buildChartRow(), buildFrameLabelCard(task, slot, payload), buildScrubber(payload));
  stage.append(main);
  return stage;
}

function buildReadOnlyScene(batch, sceneId) {
  const card = node("div", "review-scene-card");
  const head = node("div", "review-scene-head");
  head.append(node("span", "eyebrow", translate("review.scene")), node("strong", "", sceneId));
  const grid = node("div", "scene-grid compact");
  const scene = sceneFor(batch, sceneId);
  renderGridCells(grid, batch, scene, true);
  card.append(head, grid);
  return card;
}

function buildCameraStrip(batch, slot, payload) {
  const strip = node("div", "cam-strip");
  if (!payload.videos.length) {
    strip.append(node("div", "cam-empty", slot.episode ? translate("qc.noVideo") : translate("qc.emptySlot")));
    return strip;
  }
  const videoBatch = payload.video_batch || batch;
  const videoEpisode = payload.video_episode_index ?? slot.episode.episode_index;
  for (const video of payload.videos) {
    const cell = node("div", "cam-cell");
    cell.append(node("span", "cam-label", video.label));
    const media = node("video", "cam");
    media.preload = "metadata";
    media.autoplay = true;
    media.loop = true;
    media.muted = true;
    media.playsInline = true;
    media.src = "/api/batches/" + encodeURIComponent(videoBatch)
      + "/episodes/" + videoEpisode
      + "/video/" + encodePath(video.key);
    cell.append(media);
    strip.append(cell);
  }
  return strip;
}

function buildChartRow() {
  const row = node("div", "stage-charts");
  row.append(buildChart("action", translate("review.action")), buildChart("state", translate("review.state")));
  return row;
}

function buildChart(kind, label) {
  const panel = node("div", "stage-chart");
  const head = node("div", "chart-head");
  const title = node("span", "chart-title", label);
  title.append(node("small", "", " t · x"));
  const dims = node("div", "chart-dims");
  dims.id = "review-" + kind + "-dims";
  head.append(title, dims);
  const body = node("div", "chart-body");
  const canvas = node("canvas");
  canvas.id = "review-" + kind + "-chart";
  body.append(canvas);
  panel.append(head, body);
  return panel;
}

function buildFrameLabelCard(task, slot, payload) {
  const card = node("section", "frame-label-card");
  const head = node("div", "frame-label-head");
  head.append(
    node("span", "chart-title", translate("qc.frameLabels")),
    node("span", "frame-label-head-tag", translate("qc.trimTag")),
  );
  const series = payload.series;
  const labels = series && (series.frame_labels || series.labels) || [];
  const total = series ? series.state.length : 0;
  const analysis = frameLabelAnalysis(series, labels);
  const track = node("div", "frame-label-track");
  track.id = "review-frame-labels";
  let controls = null;
  if (labels.length === total && total) {
    for (const segmentInfo of analysis.segments) {
      const {label, start, end} = segmentInfo;
      const segment = node("span", "frame-label-segment " + label);
      segment.style.flex = `${end - start} 0 0%`;
      segment.title = `${frameLabelText(label)} · ${start + 1}-${end}`;
      track.append(segment);
    }
    for (const segment of analysis.static_segments || []) {
      for (const boundary of [segment.start, segment.end]) {
        if (boundary <= 0 || boundary >= total) continue;
        const marker = node("span", "frame-label-boundary");
        marker.style.setProperty("--boundary-pct", `${boundary / total * 100}%`);
        marker.title = `${translate("qc.frameBoundary")} ${boundary}`;
        track.append(marker);
      }
    }
    controls = buildFrameControls(track, task, slot, analysis, total);
    track.append(controls.startMarker, controls.endMarker);
  } else {
    track.classList.add("empty");
    track.append(node("span", "frame-label-empty", translate("qc.noFrameLabels")));
  }
  card.append(head, track);
  if (total) {
    const range = node("div", "frame-label-range");
    range.id = "review-frame-range";
    range.textContent = `${translate("qc.trimRange")} ${Math.max(1, Number(analysis.trim_start_frame) + 1)}-${Math.min(total, Number(analysis.trim_end_frame) || total)} / ${total}`;
    card.append(range);
  }
  if (analysis.middle_static_frames) {
    const warning = node("div", "frame-label-warning");
    warning.append(node("b", "", translate("qc.staticFramesExcessive")), node("span", "", translate("qc.staticFramesHint")));
    card.append(warning);
  }
  if (controls) card.append(controls.controls);
  if (controls) {
    const actions = node("div", "frame-label-actions");
    actions.append(controls.save);
    card.append(actions);
  }
  const legend = node("div", "frame-label-legend");
  for (const label of ["static", "non-static"]) {
    const item = node("span", "frame-label-key " + label);
    item.append(node("i"), node("span", "", frameLabelText(label)));
    legend.append(item);
  }
  card.append(legend);
  return card;
}

function frameLabelAnalysis(series, labels) {
  if (series && series.frame_label_analysis) return series.frame_label_analysis;
  const segments = [];
  let start = 0;
  while (start < labels.length) {
    const label = labels[start] || "unlabeled";
    let end = start + 1;
    while (end < labels.length && (labels[end] || "unlabeled") === label) end += 1;
    segments.push({label, start, end, start_frame: start, end_frame: end - 1, length: end - start});
    start = end;
  }
  const staticSegments = segments.filter((segment) => segment.label === "static");
  const leading = staticSegments[0] && staticSegments[0].start === 0 ? staticSegments[0] : null;
  const trailing = staticSegments.at(-1) && staticSegments.at(-1).end === labels.length
    ? staticSegments.at(-1) : null;
  const edge = new Set([leading, trailing].filter(Boolean));
  const middle = staticSegments.filter((segment) => !edge.has(segment));
  const onlyStatic = leading && leading === trailing;
  return {
    segments,
    static_segments: staticSegments,
    middle_static_frames: middle.reduce((sum, segment) => sum + segment.length, 0),
    static_frames_excessive: middle.length > 0,
    trim_start_frame: onlyStatic ? 0 : leading ? leading.end : 0,
    trim_end_frame: onlyStatic ? labels.length : trailing ? trailing.start : labels.length,
  };
}

function buildFrameControls(track, task, slot, analysis, total) {
  const points = [...new Set([
    0,
    total,
    ...(analysis.static_segments || []).flatMap((segment) => [segment.start, segment.end]),
  ])].sort((left, right) => left - right);
  const nearest = (value) => points.reduce((best, point) => (
    Math.abs(point - value) < Math.abs(best - value) ? point : best
  ), points[0]);
  let start = Math.max(0, Math.min(total - 1, nearest(Number(analysis.trim_start_frame) || 0)));
  let end = Math.max(start + 1, Math.min(total, nearest(Number(analysis.trim_end_frame) || total)));
  if (end <= start) end = Math.min(total, start + 1);
  const startMarker = node("span", "frame-label-trim-marker start");
  const endMarker = node("span", "frame-label-trim-marker end");
  const controls = node("div", "frame-label-controls");
  const save = button(translate("qc.trimApply"), "", "button primary frame-label-trim");
  const startControl = buildFrameBoundaryControl("start", translate("qc.trimStart"));
  const endControl = buildFrameBoundaryControl("end", translate("qc.trimEnd"));
  controls.append(startControl.wrapper, endControl.wrapper);

  function buildFrameBoundaryControl(type, label) {
    const wrapper = node("div", "frame-label-boundary-control " + type);
    const heading = node("strong", "", label);
    const left = button("←", "", "button frame-label-shift");
    const value = node("span", "frame-label-control-value");
    const right = button("→", "", "button frame-label-shift");
    left.setAttribute("aria-label", `${label} ${translate("qc.moveLeft")}`);
    right.setAttribute("aria-label", `${label} ${translate("qc.moveRight")}`);
    left.title = `${label} ${translate("qc.moveLeft")}`;
    right.title = `${label} ${translate("qc.moveRight")}`;
    left.addEventListener("click", () => shift(type, -1));
    right.addEventListener("click", () => shift(type, 1));
    wrapper.append(heading, left, value, right);
    return {wrapper, left, value, right};
  }

  const render = () => {
    startMarker.style.setProperty("--trim-pct", `${start / Math.max(total, 1) * 100}%`);
    endMarker.style.setProperty("--trim-pct", `${end / Math.max(total, 1) * 100}%`);
    startControl.value.textContent = String(start + 1);
    endControl.value.textContent = String(end);
    startControl.value.title = `${translate("qc.trimStart")} ${start + 1}`;
    endControl.value.title = `${translate("qc.trimEnd")} ${end}`;
    const startIndex = points.indexOf(start);
    const endIndex = points.indexOf(end);
    startControl.left.disabled = startIndex <= 0;
    startControl.right.disabled = startIndex >= points.length - 1 || points[startIndex + 1] >= end;
    endControl.left.disabled = endIndex <= 0 || points[endIndex - 1] <= start;
    endControl.right.disabled = endIndex >= points.length - 1;
    const range = $("review-frame-range");
    if (range) range.textContent = `${translate("qc.trimRange")} ${start + 1}-${end} / ${total}`;
    save.disabled = start === 0 && end === total;
  };

  function shift(type, direction) {
    const current = type === "start" ? start : end;
    const index = points.indexOf(current);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= points.length) return;
    const value = points[nextIndex];
    if (type === "start") {
      if (value >= end) return;
      start = value;
    } else {
      if (value <= start) return;
      end = value;
    }
    render();
  }

  save.addEventListener("click", () => saveTrim(task, slot, start, end, save));
  render();
  return {startMarker, endMarker, controls, save};
}

async function saveTrim(task, slot, start, end, control) {
  if (start === 0 && end === Number(app.review && app.review.series && app.review.series.state.length)) {
    showToast(translate("qc.trimNoChange"), true);
    return;
  }
  const saved = await runWrite(control, async () => {
    await writeJson(
      "/api/batches/" + encodeURIComponent(task.batch_id)
        + "/episodes/" + slot.episode.episode_index + "/trim",
      "POST",
      {start_frame: start, end_frame: end},
    );
    await loadState(true, true);
  }, translate("qc.trimSaved"));
  if (!saved) return;
  const currentTask = (app.state.tasks || []).find((item) => item.batch_id === task.batch_id && item.task_id === task.task_id);
  const currentSlot = currentTask && currentTask.slots.find((item) => item.slot_id === slot.slot_id);
  if (currentTask && currentSlot) openSlot(currentTask, currentSlot);
}

function frameLabelText(label) {
  return translate(label === "static" ? "qc.staticFrames" : "qc.nonStaticFrames");
}

function buildScrubber(payload) {
  const bar = node("div", "stage-scrub");
  const play = button("▶", "", "scrub-play");
  play.id = "review-play";
  const label = node("span", "scrub-state", payload.series ? translate("qc.review") : translate("qc.pendingState"));
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
  const hasEpisode = Boolean(slot.episode);
  const episode = slot.episode;
  const note = textarea("qc_note", episode ? episode.qc_note || "" : "", translate("qc.notePlaceholder"));
  note.id = "qc-note";
  const reason = document.createElement("select");
  reason.id = "qc-reason";
  reason.append(new Option(translate("qc.reasonPlaceholder"), ""));
  const reasonOptions = [
    ["image_quality", translate("qc.imageReason")],
    ["trajectory_quality", translate("qc.trajectoryReason")],
    ["task_mismatch", translate("qc.taskReason")],
    [STATIC_REASON, translate("qc.staticFramesExcessive")],
    ["other", translate("qc.otherReason")],
  ];
  for (const [value, label] of reasonOptions) reason.add(new Option(label, value));
  reason.value = episode ? episode.qc_reason || "" : "";
  reason.addEventListener("change", () => { note.placeholder = reason.value === "other" ? translate("review.otherReasonPlaceholder") : translate("qc.notePlaceholder"); });
  const pass = button(translate("qc.pass"), "", "button qc-verdict-action qc-pass");
  const fail = button(translate("qc.fail"), "", "button qc-verdict-action qc-fail");
  pass.classList.toggle("selected", Boolean(episode && episode.qc_verdict === "pass"));
  fail.classList.toggle("selected", Boolean(episode && episode.qc_verdict === "fail"));
  pass.addEventListener("click", () => saveQc(task, slot, "pass", pass));
  fail.addEventListener("click", () => saveQc(task, slot, "fail", fail));
  reason.disabled = !hasEpisode;
  note.disabled = !hasEpisode;
  pass.disabled = !hasEpisode;
  fail.disabled = !hasEpisode;
  const passRow = node("div", "qc-verdict-row");
  passRow.append(pass);
  const failRow = node("div", "qc-verdict-row");
  failRow.append(fail);
  section.append(passRow, reason, note, failRow);
  return section;
}

async function saveQc(task, slot, verdict, control) {
  const reasonValue = $("qc-reason").value;
  const noteValue = $("qc-note").value.trim();
  if (verdict === "fail" && !reasonValue) {
    showToast(translate("review.failureReasonRequired"), true);
    return;
  }
  if (reasonValue === "other" && !noteValue) {
    showToast(translate("review.otherReasonRequired"), true);
    return;
  }
  const localEpisode = slot.episode;
  const previous = {
    verdict: localEpisode && localEpisode.qc_verdict,
    note: localEpisode && localEpisode.qc_note,
    reason: localEpisode && localEpisode.qc_reason,
    qcState: slot.qc_state,
    state: slot.state,
  };
  const effectiveVerdict = verdict || (localEpisode && localEpisode.qc_verdict) || "";
  if (localEpisode) {
    localEpisode.qc_verdict = effectiveVerdict;
    localEpisode.qc_note = noteValue;
    localEpisode.qc_reason = verdict === "fail" || !verdict ? reasonValue : "";
  }
  slot.qc_state = effectiveVerdict === "pass" ? "passed" : effectiveVerdict === "fail" ? "failed" : "unreviewed";
  slot.state = slot.qc_state === "failed" ? "repair" : slot.qc_state === "pending" ? "pending" : "complete";
  if (control && verdict) {
    control.classList.add("selected");
    const sibling = control.parentElement && [...control.parentElement.children].find((item) => item !== control);
    if (sibling) sibling.classList.remove("selected");
  }
  renderTaskList();
  const saved = await runWrite(control, async () => {
    await writeJson(
      "/api/batches/" + encodeURIComponent(task.batch_id)
        + "/episodes/" + slot.episode.episode_index + "/qc",
      "PUT",
      { verdict, note: noteValue, reason: verdict === "fail" || !verdict ? reasonValue : "" },
    );
  }, verdict === "pass"
    ? translate("qc.savePassed")
    : verdict === "fail" ? translate("qc.saveFailed") : translate("qc.saved"));
  if (saved) {
    navigateQcSlot(1);
  } else {
    if (localEpisode) {
      localEpisode.qc_verdict = previous.verdict;
      localEpisode.qc_note = previous.note;
      localEpisode.qc_reason = previous.reason;
    }
    slot.qc_state = previous.qcState;
    slot.state = previous.state;
    renderTaskList();
  }
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
    context.fillText(matrix.length ? translate("review.noDimensions") : translate("review.awaitingData"), pad, height / 2);
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
  const labelTrack = $("review-frame-labels");
  if (labelTrack) {
    labelTrack.style.setProperty(
      "--frame-pct",
      (index / Math.max(series.state.length - 1, 1) * 100).toFixed(2) + "%",
    );
  }
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

function autoStartPlayback() {
  const series = reviewSeries();
  if (!series || !series.state.length) return;
  const videos = [...document.querySelectorAll(".cam-strip video")];
  const begin = () => {
    if (!app.playing) togglePlayback();
    videos.forEach((video) => video.play().catch(() => {}));
  };
  videos.forEach((video) => {
    video.addEventListener("canplay", begin, {once: true});
  });
  begin();
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
  const master = videos.find((video) => !video.error && !video.paused && video.readyState >= 2);
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
  frameLabelAnalysis,
  openSlot,
  openRobotCompare,
  renderReview,
  stopPlayback,
};
