let app;
let node;
let input;
let textarea;
let button;
let field;
let clone;
let encodePath;
let recordKey;
let planFor;
let objectFor;
let sceneFor;
let showToast;
let runWrite;
let writeJson;
let loadState;
let emptyList;
let switchTab;
let stopPlayback;
let openSlot;
let translate;

const $ = (id) => document.getElementById(id);
const OBJECT_MEASUREMENTS = [
  ["length_cm", "长（cm）"], ["width_cm", "宽（cm）"],
  ["height_cm", "高（cm）"], ["mass_g", "质量（g）"],
];
const MODELING_METHODS = {
  A: "image2assets",
  B: "agent-primitive",
  C: "agent-cad",
  D: "agent-blender",
};

function objectMeasurementsText(object) {
  const dimensions = [object.length_cm, object.width_cm, object.height_cm];
  return dimensions.some(Boolean)
    ? dimensions.map((value) => value || "—").join(" × ") + " cm"
    : "未填写";
}

function qcState(slot) {
  if (slot && slot.qc_state) return slot.qc_state;
  if (!slot) return "pending";
  if (slot.state === "repair") return "failed";
  if (slot.state === "pending") return "pending";
  return slot.episode && slot.episode.qc_verdict === "pass" ? "passed" : "unreviewed";
}

function localized(zh, en, fallback = "") {
  return app.locale === "zh" ? (zh || en || fallback) : (en || zh || fallback);
}

function configureEntityUi(context) {
  ({
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
    translate,
  } = context);
}

function renderTaskList() {
  if (!app.state) return;
  const host = $("task-list");
  const tasks = (app.state.tasks || [])
    .filter((task) => !app.batch || task.batch_id === app.batch)
    .sort(compareTaskIds);
  const entries = tasks.flatMap((task) => (task.slots || []).map((slot) => ({task, slot})));
  const sceneSelect = $("qc-scene-select");
  const taskSelect = $("qc-task-select");
  const sceneIds = [...new Set(entries.map(({slot}) => slot.scene_id).filter(Boolean))].sort();
  const taskIds = [...new Set(entries.map(({task}) => task.task_id).filter(Boolean))].sort(
    (left, right) => String(left).localeCompare(String(right), "en", {numeric: true, sensitivity: "base"}),
  );
  if (sceneSelect) {
    const current = app.qcSceneFilter || "";
    sceneSelect.replaceChildren(new Option(translate("filters.allScenes"), ""), ...sceneIds.map((id) => new Option(id, id)));
    sceneSelect.value = sceneIds.includes(current) ? current : "";
    app.qcSceneFilter = sceneSelect.value;
    sceneSelect.disabled = !entries.length;
  }
  if (taskSelect) {
    const current = app.qcTaskFilter || "";
    taskSelect.replaceChildren(new Option(translate("filters.allTasks"), ""), ...taskIds.map((id) => new Option(id, id)));
    taskSelect.value = taskIds.includes(current) ? current : "";
    app.qcTaskFilter = taskSelect.value;
    taskSelect.disabled = !entries.length;
  }
  const counts = entries.reduce((total, {slot}) => {
    const state = qcState(slot);
    total[state] = (total[state] || 0) + 1;
    return total;
  }, {passed: 0, failed: 0, unreviewed: 0, pending: 0});
  const total = entries.length;
  const passed = counts.passed || 0;
  const unreviewed = counts.unreviewed || 0;
  const failed = counts.failed || 0;
  const pending = counts.pending || 0;
  // Collection progress counts usable captures; failed captures remain in
  // the QC repair bucket and are part of the supplement export instead.
  const collected = passed + unreviewed;
  const setText = (id, value) => { const element = $(id); if (element) element.textContent = value; };
  setText("collected-count", collected);
  setText("slot-total-count", total);
  setText("pending-count", pending + failed);
  setText("qc-complete-count", passed);
  setText("qc-repair-count", failed);
  setText("qc-unreviewed-count", unreviewed);
  setText("qc-pending-slot-count", pending);
  setText("qc-plan-progress", `${collected} / ${total || 0}`);
  setText("qc-remaining", total ? `${pending + failed} ${translate("qc.remaining")}` : "—");
  setText("qc-eta", "—");
  const progress = $("qc-progress-fill");
  if (progress) progress.style.width = `${total ? Math.min(100, collected / total * 100) : 0}%`;
  const filtered = entries.filter(({task, slot}) => (
    (!app.qcSceneFilter || slot.scene_id === app.qcSceneFilter)
      && (!app.qcTaskFilter || task.task_id === app.qcTaskFilter)
  ));
  const pageSize = 50;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  app.qcPage = Math.max(0, Math.min(pageCount - 1, Number(app.qcPage) || 0));
  const pageStart = app.qcPage * pageSize;
  const visible = filtered.slice(pageStart, pageStart + pageSize);
  if (!app.batch) {
    emptyList(host, translate("qc.chooseBatch"));
  } else if (!entries.length) {
    emptyList(host, translate("qc.noSlots"));
  } else if (!filtered.length) {
    emptyList(host, translate("qc.noMatches"));
  } else {
    host.replaceChildren(...visible.map(({task, slot}, index) => buildSlotTile(task, slot, pageStart + index)));
  }
  setText("qc-position", filtered.length ? `${app.qcPage + 1} / ${pageCount}` : "— / —");
  const previous = $("qc-prev-slot");
  const next = $("qc-next-slot");
  if (previous) previous.disabled = app.qcPage <= 0;
  if (next) next.disabled = app.qcPage >= pageCount - 1;
}

function navigateQcSlot(offset) {
  if (!app.state || !app.batch) return;
  const entries = (app.state.tasks || [])
    .filter((task) => task.batch_id === app.batch)
    .sort(compareTaskIds)
    .flatMap((task) => (task.slots || []).map((slot) => ({task, slot})));
  const filtered = entries.filter(({task, slot}) => (
    (!app.qcSceneFilter || slot.scene_id === app.qcSceneFilter)
      && (!app.qcTaskFilter || task.task_id === app.qcTaskFilter)
  ));
  const current = filtered.findIndex(({ slot }) => slot.slot_id === app.selectedSlot);
  const nextIndex = current < 0 ? 0 : current + offset;
  if (nextIndex >= 0 && nextIndex < filtered.length) {
    app.qcPage = Math.floor(nextIndex / 50);
    renderTaskList();
    const nextEntry = filtered[nextIndex];
    if (nextEntry) openSlot(nextEntry.task, nextEntry.slot);
  }
}

function compareTaskIds(left, right) {
  const leftId = String(left.task_id || "");
  const rightId = String(right.task_id || "");
  const leftMatch = /^TASK-(\d+)(?:-(.*))?$/i.exec(leftId);
  const rightMatch = /^TASK-(\d+)(?:-(.*))?$/i.exec(rightId);
  if (!leftMatch || !rightMatch) return leftId.localeCompare(rightId);
  const numberOrder = Number(leftMatch[1]) - Number(rightMatch[1]);
  if (numberOrder) return numberOrder;
  const suffixOrder = (leftMatch[2] || "").localeCompare(rightMatch[2] || "", "en", {
    numeric: true,
    sensitivity: "base",
  });
  return suffixOrder || leftId.localeCompare(rightId);
}

function buildTaskAccordion(task) {
  const key = recordKey(task, "task_id");
  const total = task.slots.length;
  const counts = task.counts || {};
  const passed = Number.isFinite(counts.passed)
    ? counts.passed
    : task.slots.filter((slot) => qcState(slot) === "passed").length;
  const failed = Number.isFinite(counts.failed)
    ? counts.failed
    : task.slots.filter((slot) => qcState(slot) === "failed").length;
  const collected = task.slots.filter((slot) => qcState(slot) !== "pending").length;
  const status = total > 0 && passed === total
    ? "complete"
    : failed > 0
      ? "repair"
      : collected > 0
        ? "partial"
        : "pending";
  const wrapper = node("section", "task-accordion" + (app.expandedTask === key ? " open" : ""));
  const row = node("button", "task-row");
  row.type = "button";
  const stateBar = node("span", "task-statebar " + status);
  const copy = node("span", "entity-copy");
  copy.append(
    node("strong", "", task.task_id),
    node("small", "", localized(task.prompt_zh, task.prompt_en, task.action)),
  );
  const slotTotal = node(
    "span",
    "task-slot-total" + (status === "complete" ? " complete" : ""),
    collected + "/" + total,
  );
  slotTotal.title = translate("entity.collectedSummary") + " " + collected + " / " + total + " · " + translate("status.passed") + " " + passed + " · " + translate("status.failed") + " " + failed;
  row.append(stateBar, copy, slotTotal, node("span", "task-chevron", "⌄"));
  row.addEventListener("click", () => {
    app.expandedTask = app.expandedTask === key ? "" : key;
    selectTask(task);
    renderTaskList();
  });
  wrapper.append(row);
  if (app.expandedTask === key) {
    const slots = node("div", "task-slots");
    for (const [index, slot] of task.slots.entries()) {
      slots.append(buildSlotTile(task, slot, index));
    }
    if (!task.slots.length) emptyList(slots, translate("entity.noEpisodes"));
    wrapper.append(slots);
  }
  return wrapper;
}

function buildSlotTile(task, slot, index) {
  const tile = node(
    "button",
    "slot-tile " + qcState(slot)
      + (app.selected.tasks === recordKey(task, "task_id")
        && app.selectedSlot === slot.slot_id ? " active" : ""),
  );
  tile.type = "button";
  const stateLabels = {
    pending: translate("status.pending"),
    unreviewed: translate("status.unreviewed"),
    passed: translate("status.passed"),
    failed: translate("status.failed"),
  };
  tile.title = slot.slot_id + " · " + stateLabels[qcState(slot)] + (slot.episode ? " · " + translate("qc.episode") + " " + slot.episode.episode_index : "");
  tile.setAttribute("aria-label", tile.title);
  tile.append(node("b", "", String(index + 1)));
  tile.addEventListener("click", (event) => {
    event.stopPropagation();
    openSlot(task, slot);
  });
  return tile;
}

function selectTask(task) {
  stopPlayback();
  app.selectedSlot = "";
  app.review = null;
  const key = recordKey(task, "task_id");
  app.selected.tasks = key;
  app.original.tasks = task.task_id;
  app.draft.tasks = clone(task);
  renderTaskEditor();
}


function renderSceneList() {
  if (!app.state) return;
  const query = $("scene-search").value.trim().toLowerCase();
  const scenes = app.state.scenes.filter((scene) => {
    return (scene.scene_id + " " + scene.batch_id).toLowerCase().includes(query);
  });
  const host = $("scene-list");
  if (!scenes.length) {
    emptyList(host, query ? translate("entity.noSceneMatches") : translate("entity.noScenes"));
    return;
  }
  host.replaceChildren(...scenes.map((scene, index) => {
    const key = recordKey(scene, "scene_id");
    const row = entityRow(
      index,
      scene.scene_id,
      scene.placements.map((item) => item.object_id).join(" · "),
      scene.placements.length + " " + translate("entity.groups"),
      app.selected.scenes === key,
    );
    row.addEventListener("click", () => selectScene(scene));
    return row;
  }));
}

function objectMatchesFilters(object, query, photoFilter, modelingFilter) {
  const matchesQuery = [object.object_id, object.object_name, object.object_name_zh]
    .join(" ").toLowerCase().includes(query);
  const matchesPhoto = !photoFilter
    || (photoFilter === "missing" && !object.photos.length)
    || (photoFilter === "ready" && object.photos.length);
  const matchesModeling = !modelingFilter
    || (modelingFilter === "missing" && !object.modeling_method)
    || object.modeling_method === modelingFilter;
  return matchesQuery && matchesPhoto && matchesModeling;
}

function renderObjectList() {
  if (!app.state) return;
  const query = $("object-search").value.trim().toLowerCase();
  const photoFilter = $("object-photo-filter").value;
  const modelingFilter = $("object-modeling-filter").value;
  const objects = app.state.objects.filter(
    (object) => objectMatchesFilters(object, query, photoFilter, modelingFilter),
  );
  $("object-filter-count").textContent = objects.length + " / " + app.state.objects.length;
  const host = $("object-list");
  if (!objects.length) {
    if (app.objectThumbObserver) app.objectThumbObserver.disconnect();
    emptyList(
      host,
      query || photoFilter || modelingFilter
        ? translate("entity.noObjectMatches")
        : translate("entity.noObjects"),
    );
    return;
  }
  host.replaceChildren(...objects.map((object) => {
    const row = node(
      "button",
      "object-card" + (
        app.selected.objects === recordKey(object, "object_id") ? " active" : ""
      ),
    );
    row.type = "button";
    const preview = objectPhotoThumb(object);
    const copy = node("span", "object-card-copy");
    copy.append(
      node("small", "", object.object_id),
      node("strong", "", localized(object.object_name_zh, object.object_name, translate("entity.noName"))),
    );
    row.append(preview, copy, node("span", "badge", object.photos.length + " " + translate("entity.photos")));
    row.addEventListener("click", () => selectObject(object));
    return row;
  }));
  observeObjectThumbnails(host);
}

function entityRow(index, title, subtitle, badge, active) {
  const row = node("button", "entity-row" + (active ? " active" : ""));
  row.type = "button";
  row.append(node("span", "entity-index", String(index + 1).padStart(2, "0")));
  const copy = node("span", "entity-copy");
  copy.append(node("strong", "", title), node("small", "", subtitle || "—"));
  row.append(copy, node("span", "badge", badge));
  return row;
}

function objectPhotoThumb(object) {
  const holder = node("span", "object-thumb");
  if (object.photos.length) {
    const image = node("img");
    image.loading = "lazy";
    image.decoding = "async";
    image.dataset.src = objectPreviewUrl(object, object.photos[0], "thumb");
    image.alt = "";
    holder.append(image);
  } else {
    holder.append(node("span", "", "□"));
  }
  return holder;
}

function observeObjectThumbnails(host) {
  if (app.objectThumbObserver) app.objectThumbObserver.disconnect();
  const images = host.querySelectorAll("img[data-src]");
  if (!("IntersectionObserver" in window)) {
    images.forEach((image) => { image.src = image.dataset.src; });
    return;
  }
  app.objectThumbObserver = new IntersectionObserver((entries, observer) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.src = entry.target.dataset.src;
      delete entry.target.dataset.src;
      observer.unobserve(entry.target);
    }
  }, { root: host, rootMargin: "180px 0px" });
  images.forEach((image) => app.objectThumbObserver.observe(image));
}

function objectPreviewUrl(object, filename, variant) {
  return "/api/objects/" + encodeURIComponent(object.object_id)
    + "/previews/" + variant + "/" + encodePath(filename);
}

function selectScene(scene) {
  stopPlayback();
  const key = recordKey(scene, "scene_id");
  app.selected.scenes = key;
  app.original.scenes = scene.scene_id;
  app.draft.scenes = clone(scene);
  app.placementIndex = scene.placements.length ? 0 : -1;
  renderSceneList();
  renderSceneEditor();
}

function selectObject(object) {
  app.selected.objects = recordKey(object, "object_id");
  app.original.objects = object.object_id;
  app.draft.objects = clone(object);
  renderObjectList();
  renderObjectEditor();
  $("object-dialog").showModal();
}

function editorShell(kind, title, label) {
  const shell = node("div", "editor-shell");
  const head = node("header", "editor-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", label), node("h1", "", title));
  const actions = node("div", "editor-actions");
  if (app.original[kind]) actions.append(button(translate("entity.delete"), "delete-" + kind, "button danger"));
  actions.append(button(translate("entity.saveChanges"), "save-" + kind, "button primary"));
  head.append(heading, actions);
  const body = node("div", "editor-body");
  shell.append(head, body);
  return { shell, body };
}

function gridGeometry(batch) {
  const plan = planFor(batch);
  const configured = plan && Array.isArray(plan.layout.sampling_points)
    ? plan.layout.sampling_points
    : [];
  const defaultRows = [
    [10, 11, 1, 2, 3, 12, 13],
    [14, 15, 4, 5, 6, 16, 17],
    [18, 19, 7, 8, 9, 20, 21],
  ];
  const points = configured.length ? configured : defaultRows.flatMap((row, y) => (
    row.map((id, x) => ({ position_id: "P" + id, x: x * 250, y: y * 250 }))
  ));
  const normalized = points.map((point, index) => ({
    position_id: String(point.position_id || "P" + (index + 1)),
    x: Number(point.x),
    y: Number(point.y),
  }));
  const xs = [...new Set(normalized.map((point) => point.x))].sort((a, b) => a - b);
  const ys = [...new Set(normalized.map((point) => point.y))].sort((a, b) => a - b);
  const valid = normalized.length && normalized.every(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
  ) && normalized.length === 21 && xs.length === 7 && ys.length === 3;
  if (!valid) {
    const columns = 7;
    const rows = 3;
    const configuredById = new Map(normalized.map((point) => [point.position_id, point]));
    const fallback = defaultRows.flatMap((row, y) => row.map((id, x) => {
      const configured = configuredById.get("P" + id) || {};
      return {
        ...configured,
        position_id: "P" + id,
        x: Number.isFinite(configured.x) ? configured.x : x * 250,
        y: Number.isFinite(configured.y) ? configured.y : y * 250,
      };
    }));
    return {
      columns,
      rows,
      cells: fallback.map((point, index) => ({
        ...point,
        column: index % columns + 1,
        row: Math.floor(index / columns) + 1,
      })),
    };
  }
  return {
    columns: xs.length,
    rows: ys.length,
    cells: normalized.map((point) => ({
      ...point,
      column: xs.indexOf(point.x) + 1,
      row: ys.indexOf(point.y) + 1,
    })),
  };
}

function renderGridCells(host, batch, scene, readOnly = false) {
  const geometry = gridGeometry(batch);
  host.style.setProperty("--grid-columns", geometry.columns);
  host.style.setProperty("--grid-rows", geometry.rows);
  host.style.setProperty("--grid-aspect", geometry.columns / geometry.rows);
  const placements = scene ? scene.placements : [];
  host.replaceChildren(...geometry.cells.map((point) => {
    const matches = placements.filter((placement) => placement.position_ids.includes(point.position_id));
    const selected = !readOnly && app.placementIndex >= 0
      && placements[app.placementIndex]
      && placements[app.placementIndex].position_ids.includes(point.position_id);
    const cell = node(
      "button",
      "scene-cell" + (matches.length ? " occupied" : "")
        + (selected ? " selected" : ""),
    );
    cell.type = "button";
    cell.style.gridColumn = point.column;
    cell.style.gridRow = point.row;
    cell.append(node("small", "", point.position_id));
    if (matches.length) {
      const photoObjects = matches.map((item) => objectFor(item.object_id)).filter(
        (asset, index, assets) => asset && asset.photos.length && assets.indexOf(asset) === index,
      ).slice(0, 4);
      if (photoObjects.length) {
        const photos = node("div", "scene-cell-photos");
        photos.style.setProperty("--photo-columns", Math.min(photoObjects.length, 2));
        for (const asset of photoObjects) {
          const image = node("img");
          image.loading = "lazy";
          image.decoding = "async";
          image.src = objectPreviewUrl(asset, asset.photos[0], "thumb");
          image.alt = localized(asset.object_name_zh, asset.object_name);
          photos.append(image);
        }
        cell.classList.add("has-photo");
        cell.append(photos);
      }
      const label = matches.map((item) => {
        const asset = objectFor(item.object_id);
        return asset ? localized(asset.object_name_zh, asset.object_name, item.object_id) : item.object_id;
      }).join(" / ");
      cell.append(node("strong", "", label));
      cell.title = matches.map((item) => item.object_id).join(", ");
    }
    if (readOnly && matches.length) {
      cell.addEventListener("click", () => openObjectDialog(matches[0].object_id));
    } else if (!readOnly) {
      cell.addEventListener("click", () => togglePosition(point.position_id));
    } else {
      cell.disabled = true;
    }
    return cell;
  }));
}

function renderSceneEditor() {
  const host = $("scene-editor");
  const draft = app.draft.scenes;
  if (!draft) {
    host.replaceChildren(node("div", "empty-state", translate("entity.chooseScene")));
    return;
  }
  const editor = editorShell("scenes", draft.scene_id || translate("entity.newScene"), translate("entity.sceneEditor") + " · " + draft.batch_id);
  const layout = node("div", "scene-layout");
  const canvas = node("section", "surface scene-canvas");
  const idControl = input("text", "scene_id", draft.scene_id, "SC-001");
  idControl.disabled = Boolean(app.original.scenes);
  idControl.addEventListener("input", () => {
    draft.scene_id = idControl.value.trim();
    editor.shell.querySelector("h1").textContent = draft.scene_id || translate("entity.newScene");
  });
  const grid = node("div", "scene-grid");
  renderGridCells(grid, draft.batch_id, draft, false);
  canvas.append(field(translate("entity.sceneId"), idControl), grid, buildCameraPositionLegend(draft.batch_id));
  const placements = node("section", "surface placement-panel");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", translate("entity.placementGroups")), node("h2", "", translate("entity.objectsPositions")));
  const add = button("+ " + translate("entity.add"), "", "button");
  add.addEventListener("click", () => {
    if (!app.state.objects.length) return;
    draft.placements.push({
      object_id: app.state.objects[0].object_id,
      position_ids: [],
    });
    app.placementIndex = draft.placements.length - 1;
    renderSceneEditor();
  });
  head.append(heading);
  head.append(add);
  const list = node("div", "placement-list");
  if (draft.placements.length) {
    list.replaceChildren(...draft.placements.map(buildPlacementCard));
  } else {
    emptyList(list, translate("entity.placeHint"));
  }
  placements.append(head, list);
  layout.append(canvas, placements);
  editor.body.append(layout);
  host.replaceChildren(editor.shell);
}

function buildCameraPositionLegend(batch) {
  const section = node("div", "camera-position-list");
  const cameras = (app.state.batches.find((item) => item.batch_id === batch) || {}).cameras || [];
  section.append(node("span", "field-label", translate("entity.cameraPosition")));
  if (!cameras.length) {
    section.append(node("span", "camera-position", translate("entity.notConfigured")));
    return section;
  }
  for (const camera of cameras) {
    section.append(node(
      "span",
      "camera-position",
      camera.name + " · " + (camera.attached_to || translate("entity.external")),
    ));
  }
  return section;
}

function buildPlacementCard(placement, index) {
  const card = node("div", "placement-card" + (app.placementIndex === index ? " active" : ""));
  card.addEventListener("click", () => {
    app.placementIndex = index;
    renderSceneEditor();
  });
  const select = node("select");
  for (const object of app.state.objects) {
    select.add(new Option(
      object.object_id + " · " + localized(object.object_name_zh, object.object_name),
      object.object_id,
    ));
  }
  select.value = placement.object_id;
  select.addEventListener("click", (event) => event.stopPropagation());
  select.addEventListener("change", () => { placement.object_id = select.value; });
  const remove = button("×", "", "icon-btn");
  remove.title = translate("entity.removePlacement");
  remove.addEventListener("click", (event) => {
    event.stopPropagation();
    app.draft.scenes.placements.splice(index, 1);
    app.placementIndex = Math.min(app.placementIndex, app.draft.scenes.placements.length - 1);
    renderSceneEditor();
  });
  card.append(select);
  card.append(remove);
  card.append(node("small", "", placement.position_ids.join(", ") || translate("entity.choosePosition")));
  return card;
}

function togglePosition(positionId) {
  const draft = app.draft.scenes;
  if (app.placementIndex < 0 || !draft.placements[app.placementIndex]) {
    showToast(translate("entity.chooseGroup"), true);
    return;
  }
  const ids = draft.placements[app.placementIndex].position_ids;
  const index = ids.indexOf(positionId);
  if (index >= 0) ids.splice(index, 1); else ids.push(positionId);
  renderSceneEditor();
}

function renderTaskEditor() {
  const host = $("task-editor");
  const draft = app.draft.tasks;
  if (!draft) {
    host.replaceChildren(node("div", "empty-state", translate("entity.chooseTask")));
    return;
  }
  if (app.robotViewer) {
    app.robotViewer.dispose();
    app.robotViewer = null;
  }
  const editor = editorShell("tasks", draft.task_id || translate("entity.newTask"), translate("entity.taskEditor") + " · " + draft.batch_id);
  const form = node("form", "form-grid task-form");
  const idControl = input("text", "task_id", draft.task_id, "TASK-001");
  idControl.disabled = Boolean(app.original.tasks);
  form.append(
    field(translate("entity.taskId"), idControl),
    field(translate("entity.action"), input("text", "action", draft.action)),
    field(translate("entity.category"), input("text", "category", draft.category)),
    field(translate("entity.englishPrompt"), textarea("prompt_en", draft.prompt_en), true),
    field(translate("entity.chinesePrompt"), textarea("prompt_zh", draft.prompt_zh), true),
  );
  form.addEventListener("input", () => {
    for (const name of ["task_id", "action", "category", "prompt_en", "prompt_zh"]) {
      draft[name] = form.elements[name].value.trim();
    }
    editor.shell.querySelector("h1").textContent = draft.task_id || translate("entity.newTask");
  });
  editor.body.append(form, buildTaskObjectPicker(), buildTaskScenePicker());
  host.replaceChildren(editor.shell);
  updateTaskTotal();
}

function buildTaskObjectPicker() {
  const surface = node("section", "surface compact-picker");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", translate("entity.operationObjects")), node("h2", "", translate("entity.operationObjectsTitle")));
  const select = node("select", "add-select");
  select.add(new Option(translate("entity.addObject"), ""));
  const chosen = new Set(app.draft.tasks.operation_object_ids);
  for (const object of app.state.objects.filter((item) => !chosen.has(item.object_id))) {
    select.add(new Option(
      object.object_id + " · " + localized(object.object_name_zh, object.object_name),
      object.object_id,
    ));
  }
  select.addEventListener("change", () => {
    if (select.value) app.draft.tasks.operation_object_ids.push(select.value);
    renderTaskEditor();
  });
  head.append(heading);
  head.append(select);
  const rows = node("div", "selected-records");
  for (const objectId of app.draft.tasks.operation_object_ids) {
    const asset = objectFor(objectId);
    const row = node("div", "selected-record");
    const inspect = button(
      asset ? localized(asset.object_name_zh, asset.object_name, objectId) : objectId,
      "",
      "text-button",
    );
    inspect.addEventListener("click", () => openObjectDialog(objectId));
    const remove = button("×", "", "icon-btn");
    remove.addEventListener("click", () => {
      app.draft.tasks.operation_object_ids = app.draft.tasks.operation_object_ids
        .filter((value) => value !== objectId);
      renderTaskEditor();
    });
    row.append(node("code", "", objectId), inspect);
    row.append(remove);
    rows.append(row);
  }
  if (!rows.children.length) emptyList(rows, translate("entity.noOperationObjects"));
  surface.append(head, rows);
  return surface;
}

function buildTaskScenePicker() {
  const draft = app.draft.tasks;
  const surface = node("section", "surface compact-picker");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", translate("entity.sceneCoverage")), node("h2", "", translate("entity.sceneCoverageTitle")));
  const select = node("select", "add-select");
  select.add(new Option(translate("entity.addScene"), ""));
  const chosen = new Set(draft.scene_ids);
  const scenes = app.state.scenes.filter(
    (scene) => scene.batch_id === draft.batch_id && !chosen.has(scene.scene_id),
  );
  for (const scene of scenes) select.add(new Option(scene.scene_id, scene.scene_id));
  select.addEventListener("change", () => {
    if (select.value) {
      draft.scene_ids.push(select.value);
      draft.scene_epsiodes_count.push(1);
    }
    renderTaskEditor();
  });
  head.append(heading);
  head.append(select);
  const rows = node("div", "selected-records");
  draft.scene_ids.forEach((sceneId, index) => {
    const row = node("div", "selected-record scene-record");
    const link = button(sceneId, "", "text-button");
    link.addEventListener("click", () => jumpScene(draft.batch_id, sceneId));
    const count = input("number", "", draft.scene_epsiodes_count[index] || 1);
    count.min = "1";
    count.addEventListener("input", () => {
      draft.scene_epsiodes_count[index] = Math.max(1, Number(count.value) || 1);
      updateTaskTotal();
    });
    const remove = button("×", "", "icon-btn");
    remove.addEventListener("click", () => {
      draft.scene_ids.splice(index, 1);
      draft.scene_epsiodes_count.splice(index, 1);
      renderTaskEditor();
    });
    row.append(link, count, node("small", "", translate("common.episodes")));
    row.append(remove);
    rows.append(row);
  });
  if (!rows.children.length) emptyList(rows, translate("entity.noConfiguredScenes"));
  const total = node("div", "total-bar");
  total.id = "task-total";
  total.append(node("span", "", translate("entity.totalEpisodeSlots")), node("b", "", "0"));
  surface.append(head, rows, total);
  return surface;
}

function updateTaskTotal() {
  if (!app.draft.tasks) return;
  const total = app.draft.tasks.scene_epsiodes_count.reduce(
    (sum, value) => sum + Number(value || 0),
    0,
  );
  app.draft.tasks.total_epsiodes_count = total;
  if ($("task-total")) $("task-total").querySelector("b").textContent = total;
}

function renderObjectEditor() {
  const host = $("object-dialog-body");
  const draft = app.draft.objects;
  if (!draft) {
    host.replaceChildren(node("div", "empty-state", translate("entity.chooseObject")));
    return;
  }
  const editor = editorShell("objects", draft.object_id || translate("entity.newObject"), translate("entity.objectAsset"));
  const form = node("form", "form-grid inset surface");
  const idControl = input("text", "object_id", draft.object_id, "AST-0001");
  idControl.disabled = Boolean(app.original.objects);
  form.append(
    field(translate("entity.objectId"), idControl),
    field(translate("entity.color"), input("text", "color", draft.color, "white / #e8590c")),
    field(translate("common.englishName"), input("text", "object_name", draft.object_name), true),
    field(translate("entity.chineseName"), input("text", "object_name_zh", draft.object_name_zh), true),
    field(translate("entity.scanStatus"), input("text", "scan_status", draft.scan_status), true),
    field(translate("entity.photoDirectory"), input("text", "photo_dir", draft.photo_dir), true),
  );
  form.append(node("p", "full", "尺寸按自然放置时的外接长、宽、高填写；圆盘长宽均填直径。未知可留空。"));
  for (const [name, label] of OBJECT_MEASUREMENTS) {
    const control = input("number", name, draft[name] || "", "未填写");
    control.min = "0";
    control.step = "any";
    form.append(field(label, control));
  }
  const methodControl = node("select");
  methodControl.name = "modeling_method";
  methodControl.add(new Option("未填写", ""));
  for (const [code, label] of Object.entries(MODELING_METHODS)) {
    methodControl.add(new Option(code + " · " + label, code));
  }
  methodControl.value = draft.modeling_method || "";
  form.append(field("建模方法", methodControl, true));
  const updateDraft = () => {
    for (const name of [
      "object_id",
      "object_name",
      "object_name_zh",
      "scan_status",
      "color",
      "photo_dir",
      ...OBJECT_MEASUREMENTS.map(([name]) => name),
      "modeling_method",
    ]) {
      draft[name] = form.elements[name].value.trim();
    }
    editor.shell.querySelector("h1").textContent = draft.object_id || translate("entity.newObject");
  };
  form.addEventListener("input", updateDraft);
  form.addEventListener("change", updateDraft);
  updateDraft();
  editor.body.append(form, buildPhotoGallery(draft, true));
  host.replaceChildren(editor.shell);
}

function buildPhotoGallery(object, editable = false) {
  const section = node("section", "surface photo-surface");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", translate("entity.objectPhotos")), node("h2", "", translate("entity.physicalPhotos")));
  if (editable) {
    const upload = button(translate("entity.uploadPhotos"), "", "button");
    upload.disabled = !app.original.objects;
    upload.addEventListener("click", () => $("photo-file").click());
    head.append(heading, upload);
  } else {
    head.append(heading);
  }
  const gallery = node("div", "photo-grid");
  if (object.photos && object.photos.length) {
    const previewVariant = "display";
    for (const [index, filename] of object.photos.entries()) {
      const figure = node("figure", "photo-tile");
      const image = node("img");
      image.loading = index === 0 ? "eager" : "lazy";
      image.decoding = "async";
      image.src = objectPreviewUrl(object, filename, previewVariant);
      image.alt = localized(object.object_name_zh, object.object_name);
      figure.append(image, node("figcaption", "", filename));
      gallery.append(figure);
    }
  } else {
    const slot = node("div", "photo-slot");
    slot.append(node("b", "", "+"), node("span", "", translate("common.photoSlot")));
    gallery.append(slot);
  }
  section.append(head, gallery);
  return section;
}

function openObjectDialog(objectId) {
  const object = objectFor(objectId);
  if (!object) {
    showToast(translate("entity.noObject") + " " + objectId, true);
    return;
  }
  const body = $("object-dialog-body");
  const header = node("header", "object-modal-head");
  header.append(
    node("span", "eyebrow", object.object_id),
    node("h2", "", localized(object.object_name_zh, object.object_name, translate("entity.noName"))),
  );
  const meta = node("div", "object-modal-meta");
  meta.append(
    node("span", "", translate("entity.scanStatus") + " · " + (object.scan_status || translate("entity.valueNotSet"))),
    node("span", "", translate("entity.color") + " · " + (object.color || translate("entity.valueUnlabeled"))),
    node("span", "", translate("entity.photoDirectory") + " · " + (object.photo_dir || translate("entity.valueUnbound"))),
    node("span", "", translate("entity.measurements") + " · " + objectMeasurementsText(object)),
    node("span", "", translate("entity.mass") + " · " + (object.mass_g ? object.mass_g + " g" : translate("entity.valueNotSet"))),
    node("span", "", translate("entity.modelingMethod") + " · " + (object.modeling_method
      ? object.modeling_method + " · " + (MODELING_METHODS[object.modeling_method] || translate("entity.unknownMethod"))
      : translate("entity.valueNotSet"))),
  );
  body.replaceChildren(header, meta, buildPhotoGallery(object));
  $("object-dialog").showModal();
}

function formatDashboardNumber(value) {
  return new Intl.NumberFormat("en-US").format(Math.max(0, Math.round(Number(value) || 0)));
}

function formatDashboardPercent(value) {
  const percent = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
  return percent > 0 && percent < 1 ? percent.toFixed(1) + "%" : Math.round(percent) + "%";
}

function formatDashboardDuration(seconds) {
  const value = Number(seconds) || 0;
  if (value >= 3600) return (value / 3600).toFixed(1) + "h";
  if (value >= 60) return Math.round(value / 60) + "m";
  return Math.round(value) + "s";
}

function dashboardBatchTarget(batch) {
  return Number(
    batch.benchmark_target_episodes ?? batch.target_episodes ?? batch.episodes ?? 0,
  );
}

function dashboardBatchCount(batch) {
  return batch.benchmark_batch ? 1 : 0;
}

function setDashboardText(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function dashboardSummary(plan) {
  const counts = plan && plan.collection
    ? plan.collection.qc_counts || plan.collection.counts || {}
    : {};
  if (plan) {
    const tasks = app.state.tasks.filter((task) => task.batch_id === plan.batch_id);
    const episodes = tasks.flatMap((task) => task.slots.map((slot) => slot.episode).filter(Boolean));
    const frames = episodes.reduce((sum, episode) => sum + Number(episode.length || 0), 0);
    return {
      target: Number(counts.total || 0),
      collected: Number(counts.unreviewed || 0) + Number(counts.passed || 0) + Number(counts.failed || 0),
      pending: Number(counts.pending || 0),
      unreviewed: Number(counts.unreviewed || 0),
      passed: Number(counts.passed || 0),
      failed: Number(counts.failed || 0),
      frames,
      duration: frames / 30,
    };
  }
  const batches = (app.state.all_batches || app.state.batches || []).filter(
    (batch) => dashboardBatchCount(batch) > 0,
  );
  return batches.reduce(
    (total, batch) => ({
      target: total.target + dashboardBatchTarget(batch),
      collected: total.collected + Number(batch.collected || 0),
      pending: total.pending + Number(batch.pending || 0),
      unreviewed: total.unreviewed + Number(batch.unreviewed || 0),
      passed: total.passed + Number(batch.passed || 0),
      failed: total.failed + Number(batch.failed || 0),
      frames: total.frames + Number(batch.frames || 0),
      duration: total.duration + Number(batch.duration_seconds || 0),
    }),
    { target: 0, collected: 0, pending: 0, unreviewed: 0, passed: 0, failed: 0, frames: 0, duration: 0 },
  );
}

function renderInfo() {
  if (!app.state) return;
  // Dashboard is always a global overview, independent of the QC selection.
  const summary = dashboardSummary(null);
  const batches = app.state.all_batches || app.state.batches || [];
  const robotBatches = new Map();
  for (const batch of batches) {
    const robot = String(batch.robot_type || translate("dashboard.unknownRobot"));
    const group = robotBatches.get(robot) || [];
    group.push(batch);
    robotBatches.set(robot, group);
  }
  const robots = [...robotBatches.entries()]
    .filter(([, grouped]) => grouped.some((batch) => dashboardBatchCount(batch) > 0))
    .map(([robot]) => robot)
    .sort((left, right) => left.localeCompare(right));
  const target = summary.target;
  const collected = summary.collected;
  const qcTotal = collected || 0;
  const dashboardPassed = summary.passed;
  setDashboardText("dashboard-target", formatDashboardNumber(target));
  setDashboardText("dashboard-collected", formatDashboardNumber(collected));
  setDashboardText("dashboard-frames", formatDashboardNumber(summary.frames));
  setDashboardText("dashboard-duration", formatDashboardDuration(
    collected ? summary.duration / collected : 0,
  ));
  setDashboardText("dashboard-efficiency", formatDashboardPercent(target ? collected / target : 0));
  setDashboardText("dashboard-valid-rate", formatDashboardPercent(qcTotal ? dashboardPassed / qcTotal : 0));
  setDashboardText("dashboard-passed", formatDashboardNumber(dashboardPassed));
  setDashboardText("dashboard-pending", formatDashboardNumber(summary.pending + summary.failed));
  setDashboardText(
    "dashboard-range-summary",
    formatDashboardNumber(collected) + " " + translate("dashboard.episodes") + " / " + formatDashboardNumber(summary.frames)
      + " " + translate("dashboard.frames") + " · " + batches.reduce((count, batch) => count + dashboardBatchCount(batch), 0) + " " + translate("dashboard.plans") + " · " + robots.length + " " + translate("dashboard.robots"),
  );
  setDashboardText(
    "dashboard-robot-total",
    formatDashboardNumber(collected) + " / " + formatDashboardNumber(target),
  );
  setDashboardText(
    "dashboard-health-rate",
    formatDashboardPercent(qcTotal ? summary.passed / qcTotal : 0),
  );
  setDashboardText(
    "dashboard-qc-summary",
    formatDashboardNumber(summary.pending + summary.failed) + " " + translate("dashboard.pending"),
  );

  const robotHost = $("dashboard-robot-list");
  if (robotHost) {
    robotHost.replaceChildren(...robots.map((robot) => {
      const groupedBatches = (robotBatches.get(robot) || []).filter(
        (batch) => dashboardBatchCount(batch) > 0,
      );
      const robotSummary = groupedBatches.reduce(
        (total, batch) => ({
          target: total.target + dashboardBatchTarget(batch),
          collected: total.collected + Number(batch.collected || 0),
          pending: total.pending + Number(batch.pending || 0) + Number(batch.failed || 0),
        }),
        { target: 0, collected: 0, pending: 0 },
      );
      const row = node("div", "dashboard-robot-row");
      const copy = node("div", "dashboard-robot-copy");
      copy.append(
        node("strong", "", robot),
        node("span", "", groupedBatches.length + " " + translate("dashboard.plans")),
      );
      const track = node("div", "dashboard-progress");
      const fill = node("span", "");
      fill.style.width = (robotSummary.target
        ? Math.min(100, robotSummary.collected / robotSummary.target * 100)
        : 0) + "%";
      track.append(fill);
      const value = node("div", "dashboard-robot-value");
      value.append(
        node("b", "", formatDashboardNumber(robotSummary.collected) + " / " + formatDashboardNumber(robotSummary.target)),
        node("small", "", translate("dashboard.remaining") + " " + formatDashboardNumber(robotSummary.pending)),
      );
      row.append(copy, track, value);
      return row;
    }));
    if (!robotHost.childElementCount) emptyList(robotHost, translate("dashboard.noRobotPlans"));
  }

  const healthHost = $("dashboard-health-list");
  if (healthHost) {
    const health = [
      ["passed", translate("status.passed"), summary.passed],
      ["unreviewed", translate("status.unreviewed"), summary.unreviewed],
      ["failed", translate("status.failed"), summary.failed],
      ["pending", translate("status.pending"), summary.pending],
    ];
    healthHost.replaceChildren(...health.map(([state, label, value]) => {
      const row = node("div", "dashboard-health-row " + state);
      const bar = node("div", "dashboard-health-bar");
      const fill = node("span", "");
      fill.style.width = (target ? Math.min(100, value / target * 100) : 0) + "%";
      bar.append(fill);
      row.append(node("span", "", label), bar, node("b", "", formatDashboardNumber(value)));
      return row;
    }));
  }

  const reportHost = $("dashboard-qc-list");
  if (!reportHost) return;
  const reports = [...batches].filter(
    (batch) => Number(batch.failed || 0) > 0,
  ).sort((left, right) => (
    Number(right.failed || 0) - Number(left.failed || 0)
  ));
  reportHost.replaceChildren(...reports.map((batch) => {
    const failed = Number(batch.failed || 0);
    const row = node("div", "dashboard-qc-row failed");
    const copy = node("div", "dashboard-qc-copy");
    copy.append(
      node("strong", "", batch.batch_id),
      node(
        "span",
        "",
        (batch.robot_type || translate("dashboard.unknownRobot")) + " · " + formatDashboardNumber(batch.collected)
          + " / " + formatDashboardNumber(dashboardBatchTarget(batch))
          + " · " + translate("status.failed") + " " + formatDashboardNumber(failed),
      ),
    );
    const openQc = button(translate("dashboard.openQc"), "", "button primary");
    openQc.addEventListener("click", async () => {
      app.batch = batch.batch_id;
      app.qcSceneFilter = "";
      app.qcTaskFilter = "";
      app.qcPage = 0;
      app.selectedSlot = "";
      app.selected.tasks = "";
      app.review = null;
      await loadState();
      switchTab("tasks");
    });
    row.append(copy, openQc);
    return row;
  }));
  if (!reportHost.childElementCount) emptyList(reportHost, translate("dashboard.noReports"));
}

function renderIssues() {
  if (!app.state) return;
  const host = $("issues-list");
  if (!host) return;
  if (!app.state.issues.length) {
    const ok = node("div", "issues-ok");
    ok.append(node("b", "", "✓"), node("span", "", translate("entity.noPlanIssues")));
    host.replaceChildren(ok);
    return;
  }
  host.replaceChildren(...app.state.issues.map((issue) => {
    const row = node("div", "issue " + (issue.level === "warning" ? "warning" : "error"));
    const copy = node("div");
    copy.append(
      node("strong", "", issue.message || String(issue)),
      node("small", "", (issue.batch_id ? issue.batch_id + " · " : "") + (issue.entity || "plan")),
    );
    row.append(node("span", "issue-mark", issue.level === "warning" ? "!" : "×"), copy);
    return row;
  }));
}

function renderCurrentEditor() {
  if (app.tab === "scenes" && app.draft.scenes) renderSceneEditor();
  if (app.tab === "tasks" && app.draft.tasks && !app.selectedSlot) renderTaskEditor();
  if (app.tab === "objects" && app.draft.objects) renderObjectEditor();
}

function jumpScene(batch, sceneId) {
  const scene = sceneFor(batch, sceneId);
  if (!scene) {
    showToast(translate("entity.noScene") + " " + sceneId, true);
    return;
  }
  app.batch = batch;
  app.qcSceneFilter = sceneId;
  app.qcTaskFilter = "";
  switchTab("tasks");
}

function nextId(prefix, records, key, width = 3) {
  const numbers = records.map((item) => {
    const match = String(item[key] || "").match(/(\d+)$/);
    return match ? Number(match[1]) : 0;
  });
  return prefix + String(Math.max(0, ...numbers) + 1).padStart(width, "0");
}

function requireBatch() {
  if (app.batch) return app.batch;
  showToast(translate("entity.chooseBatch"), true);
  switchTab("tasks");
  return "";
}

function newEntity(kind) {
  if (kind !== "objects" && !requireBatch()) return;
  app.selectedSlot = "";
  app.original[kind] = "";
  if (kind === "scenes") {
    const local = app.state.scenes.filter((item) => item.batch_id === app.batch);
    app.draft.scenes = {
      batch_id: app.batch,
      scene_id: nextId("SC-", local, "scene_id"),
      placements: [],
    };
    app.selected.scenes = "";
    app.placementIndex = -1;
    renderSceneList();
    renderSceneEditor();
  } else if (kind === "tasks") {
    const local = app.state.tasks.filter((item) => item.batch_id === app.batch);
    app.draft.tasks = {
      batch_id: app.batch,
      task_id: nextId("TASK-", local, "task_id"),
      action: "",
      category: "",
      operation_object_ids: [],
      prompt_en: "",
      prompt_zh: "",
      total_epsiodes_count: 0,
      scene_ids: [],
      scene_epsiodes_count: [],
      slots: [],
    };
    app.selected.tasks = "";
    app.expandedTask = "";
    renderTaskList();
    renderTaskEditor();
  } else {
    app.draft.objects = {
      object_id: nextId("AST-", app.state.objects, "object_id", 4),
      object_name: "",
      object_name_zh: "",
      scan_status: "",
      color: "",
      photo_dir: "",
      length_cm: "",
      width_cm: "",
      height_cm: "",
      mass_g: "",
      modeling_method: "",
      photos: [],
    };
    app.selected.objects = "";
    renderObjectList();
    renderObjectEditor();
    $("object-dialog").showModal();
  }
}

function validateDraft(kind) {
  const draft = app.draft[kind];
  if (kind === "scenes") {
    if (!draft.scene_id) throw new Error(translate("entity.sceneIdRequired"));
    for (const placement of draft.placements) {
      if (!placement.object_id || !placement.position_ids.length) {
        throw new Error(translate("entity.placementRequired"));
      }
    }
  } else if (kind === "tasks") {
    if (!draft.task_id || !draft.prompt_en) throw new Error(translate("entity.promptRequired"));
    if (!draft.scene_ids.length) throw new Error(translate("entity.sceneRequired"));
    updateTaskTotal();
  } else if (!draft.object_id || (!draft.object_name && !draft.object_name_zh)) {
    throw new Error(translate("entity.objectNameRequired"));
  }
  if (kind === "objects") {
    for (const [name, label] of OBJECT_MEASUREMENTS) {
      const value = String(draft[name] ?? "").trim();
      if (value && (!Number.isFinite(Number(value)) || Number(value) <= 0)) {
        throw new Error(label + "必须是大于 0 的有限数值，或留空");
      }
    }
    if (draft.modeling_method && !Object.hasOwn(MODELING_METHODS, draft.modeling_method)) {
      throw new Error("建模方法必须为 A、B、C、D，或留空");
    }
  }
  return draft;
}


export {
  configureEntityUi,
  jumpScene,
  navigateQcSlot,
  newEntity,
  objectMatchesFilters,
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
  requireBatch,
  validateDraft,
};
