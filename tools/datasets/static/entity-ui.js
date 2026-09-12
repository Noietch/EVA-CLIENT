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
let requireEditMode;
let emptyList;
let switchTab;
let stopPlayback;
let openSlot;

const $ = (id) => document.getElementById(id);

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
    requireEditMode,
    emptyList,
    switchTab,
    stopPlayback,
    openSlot,
  } = context);
}

function renderTaskFilters() {
  const filters = [
    [$("task-action-filter"), app.state.tasks.map((task) => task.action)],
    [$("task-category-filter"), app.state.tasks.map((task) => task.category)],
  ];
  for (const [select, values] of filters) {
    const previous = select.value;
    const label = select.options[0] ? select.options[0].textContent : "全部";
    select.replaceChildren(new Option(label, ""));
    [...new Set(values.filter(Boolean))].sort().forEach((value) => {
      select.add(new Option(value, value));
    });
    select.value = previous;
  }
}

function renderTaskList() {
  if (!app.state) return;
  const query = $("task-search").value.trim().toLowerCase();
  const action = $("task-action-filter").value;
  const category = $("task-category-filter").value;
  const tasks = app.state.tasks.filter((task) => {
    const text = [
      task.task_id,
      task.action,
      task.category,
      task.prompt_en,
      task.prompt_zh,
      task.batch_id,
    ].join(" ").toLowerCase();
    return text.includes(query)
      && (!action || task.action === action)
      && (!category || task.category === category);
  }).sort(compareTaskIds);
  const host = $("task-list");
  if (!tasks.length) {
    emptyList(host, query || action || category ? "没有匹配的任务" : "当前批次没有任务");
    return;
  }
  if ($("task-group-by").value === "task") {
    host.replaceChildren(...tasks.map((task) => buildTaskAccordion(task)));
    return;
  }
  const groups = new Map();
  for (const task of tasks) {
    for (const sceneId of task.scene_ids) {
      const key = JSON.stringify([task.batch_id, sceneId]);
      if (!groups.has(key)) {
        const group = node("section", "scene-task-group");
        const slots = node("div", "task-slots");
        group.append(node("strong", "", sceneId + " · " + task.batch_id), slots);
        groups.set(key, {group, slots});
      }
    }
    for (const slot of task.slots) {
      const key = JSON.stringify([task.batch_id, slot.scene_id]);
      const {slots} = groups.get(key);
      slots.append(buildSlotTile(task, slot, slots.childElementCount));
    }
  }
  host.replaceChildren(...[...groups.values()].map(({group}) => group));
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
  const complete = Number.isFinite(counts.complete)
    ? counts.complete
    : task.slots.filter((slot) => slot.state === "complete").length;
  const repair = Number.isFinite(counts.repair)
    ? counts.repair
    : task.slots.filter((slot) => slot.state === "repair").length;
  const collected = complete + repair;
  const status = total > 0 && complete === total
    ? "complete"
    : repair > 0
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
    node("small", "", task.prompt_zh || task.prompt_en || task.action),
  );
  const slotTotal = node(
    "span",
    "task-slot-total" + (status === "complete" ? " complete" : ""),
    complete + "/" + total + "完成",
  );
  slotTotal.title = "完成 " + complete + " 条，返修 " + repair + " 条，共 " + total + " 条";
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
    if (!task.slots.length) emptyList(slots, "该任务尚未配置 episode slot");
    wrapper.append(slots);
  }
  return wrapper;
}

function buildSlotTile(task, slot, index) {
  const tile = node(
    "button",
    "slot-tile " + (slot.episode ? "collected" : "pending")
      + (app.selected.tasks === recordKey(task, "task_id")
        && app.selectedSlot === slot.slot_id ? " active" : ""),
  );
  tile.type = "button";
  tile.title = slot.slot_id + (slot.episode
    ? " · 采集完成 · Episode " + slot.episode.episode_index
    : " · 未采集");
  tile.append(
    node("b", "", String(index + 1).padStart(2, "0")),
    node("small", "", slot.episode ? "完成" : "未采"),
  );
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
    emptyList(host, query ? "没有匹配的场景" : "当前批次没有场景");
    return;
  }
  host.replaceChildren(...scenes.map((scene, index) => {
    const key = recordKey(scene, "scene_id");
    const row = entityRow(
      index,
      scene.scene_id,
      scene.placements.map((item) => item.object_id).join(" · "),
      scene.placements.length + " 组",
      app.selected.scenes === key,
    );
    row.addEventListener("click", () => selectScene(scene));
    return row;
  }));
}

function renderObjectList() {
  if (!app.state) return;
  const query = $("object-search").value.trim().toLowerCase();
  const photoFilter = $("object-photo-filter").value;
  const objects = app.state.objects.filter((object) => {
    const matchesQuery = [object.object_id, object.object_name, object.object_name_zh]
      .join(" ").toLowerCase().includes(query);
    const matchesPhoto = !photoFilter
      || (photoFilter === "missing" && !object.photos.length)
      || (photoFilter === "ready" && object.photos.length);
    return matchesQuery && matchesPhoto;
  });
  $("object-filter-count").textContent = objects.length + " / " + app.state.objects.length;
  const host = $("object-list");
  if (!objects.length) {
    if (app.objectThumbObserver) app.objectThumbObserver.disconnect();
    emptyList(host, query || photoFilter ? "没有匹配的物体" : "还没有物体资产");
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
      node("strong", "", object.object_name_zh || object.object_name),
      node("span", "", object.object_name || "未填写英文名"),
    );
    row.append(preview, copy, node("span", "badge", object.photos.length + " 图"));
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
  if (app.editMode) {
    if (app.original[kind]) actions.append(button("删除", "delete-" + kind, "button danger"));
    actions.append(button("保存更改", "save-" + kind, "button primary"));
  } else {
    actions.append(node("span", "read-only-label", "只读模式"));
  }
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
  const valid = normalized.length && normalized.every(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
  );
  if (!valid) {
    const columns = Math.max(1, Math.ceil(Math.sqrt(normalized.length || 9)));
    const fallback = normalized.length ? normalized : Array.from({ length: 9 }, (_, i) => ({
      position_id: "P" + (i + 1),
    }));
    return {
      columns,
      rows: Math.ceil(fallback.length / columns),
      cells: fallback.map((point, index) => ({
        ...point,
        column: index % columns + 1,
        row: Math.floor(index / columns) + 1,
      })),
    };
  }
  const xs = [...new Set(normalized.map((point) => point.x))].sort((a, b) => a - b);
  const ys = [...new Set(normalized.map((point) => point.y))].sort((a, b) => a - b);
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
          image.alt = asset.object_name_zh || asset.object_name;
          photos.append(image);
        }
        cell.classList.add("has-photo");
        cell.append(photos);
      }
      const label = matches.map((item) => {
        const asset = objectFor(item.object_id);
        return asset ? asset.object_name_zh || asset.object_name : item.object_id;
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
    host.replaceChildren(node("div", "empty-state", "选择或新增场景"));
    return;
  }
  const editor = editorShell("scenes", draft.scene_id || "新场景", "SCENE EDITOR · " + draft.batch_id);
  const layout = node("div", "scene-layout");
  const canvas = node("section", "surface scene-canvas");
  const idControl = input("text", "scene_id", draft.scene_id, "SC-001");
  idControl.disabled = Boolean(app.original.scenes) || !app.editMode;
  idControl.addEventListener("input", () => {
    draft.scene_id = idControl.value.trim();
    editor.shell.querySelector("h1").textContent = draft.scene_id || "新场景";
  });
  const grid = node("div", "scene-grid");
  renderGridCells(grid, draft.batch_id, draft, !app.editMode);
  canvas.append(field("Scene ID", idControl), grid, buildCameraPositionLegend(draft.batch_id));
  const placements = node("section", "surface placement-panel");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", "PLACEMENT GROUPS"), node("h2", "", "物体与位置"));
  const add = button("+ 添加", "", "button");
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
  if (app.editMode) head.append(add);
  const list = node("div", "placement-list");
  if (draft.placements.length) {
    list.replaceChildren(...draft.placements.map(buildPlacementCard));
  } else {
    emptyList(list, "添加物体后，在左侧九宫格选择位置");
  }
  placements.append(head, list);
  layout.append(canvas, placements);
  editor.body.append(layout);
  host.replaceChildren(editor.shell);
}

function buildCameraPositionLegend(batch) {
  const section = node("div", "camera-position-list");
  const cameras = (app.state.batches.find((item) => item.batch_id === batch) || {}).cameras || [];
  section.append(node("span", "field-label", "相机位置"));
  if (!cameras.length) {
    section.append(node("span", "camera-position", "未配置"));
    return section;
  }
  for (const camera of cameras) {
    section.append(node(
      "span",
      "camera-position",
      camera.name + " · " + (camera.attached_to || "外置"),
    ));
  }
  return section;
}

function buildPlacementCard(placement, index) {
  const card = node("div", "placement-card" + (app.placementIndex === index ? " active" : ""));
  if (app.editMode) {
    card.addEventListener("click", () => {
      app.placementIndex = index;
      renderSceneEditor();
    });
  }
  const select = node("select");
  for (const object of app.state.objects) {
    select.add(new Option(
      object.object_id + " · " + (object.object_name_zh || object.object_name),
      object.object_id,
    ));
  }
  select.value = placement.object_id;
  select.disabled = !app.editMode;
  select.addEventListener("click", (event) => event.stopPropagation());
  select.addEventListener("change", () => { placement.object_id = select.value; });
  const remove = button("×", "", "icon-btn");
  remove.title = "移除 placement";
  remove.addEventListener("click", (event) => {
    event.stopPropagation();
    app.draft.scenes.placements.splice(index, 1);
    app.placementIndex = Math.min(app.placementIndex, app.draft.scenes.placements.length - 1);
    renderSceneEditor();
  });
  card.append(select);
  if (app.editMode) card.append(remove);
  card.append(node("small", "", placement.position_ids.join(", ") || "未选择位置"));
  return card;
}

function togglePosition(positionId) {
  const draft = app.draft.scenes;
  if (app.placementIndex < 0 || !draft.placements[app.placementIndex]) {
    showToast("先添加或选择一个物体组", true);
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
    host.replaceChildren(node("div", "empty-state", "选择任务，展开查看 episode slot"));
    return;
  }
  if (app.robotViewer) {
    app.robotViewer.dispose();
    app.robotViewer = null;
  }
  const editor = editorShell("tasks", draft.task_id || "新任务", "TASK EDITOR · " + draft.batch_id);
  const form = node("form", "form-grid task-form");
  const idControl = input("text", "task_id", draft.task_id, "TASK-001");
  idControl.disabled = Boolean(app.original.tasks) || !app.editMode;
  form.append(
    field("Task ID", idControl),
    field("Action", input("text", "action", draft.action)),
    field("Category", input("text", "category", draft.category)),
    field("English prompt", textarea("prompt_en", draft.prompt_en), true),
    field("中文提示词", textarea("prompt_zh", draft.prompt_zh), true),
  );
  if (!app.editMode) {
    form.querySelectorAll("input, textarea").forEach((control) => { control.disabled = true; });
  }
  form.addEventListener("input", () => {
    for (const name of ["task_id", "action", "category", "prompt_en", "prompt_zh"]) {
      draft[name] = form.elements[name].value.trim();
    }
    editor.shell.querySelector("h1").textContent = draft.task_id || "新任务";
  });
  editor.body.append(form, buildTaskObjectPicker(), buildTaskScenePicker());
  host.replaceChildren(editor.shell);
  updateTaskTotal();
}

function buildTaskObjectPicker() {
  const surface = node("section", "surface compact-picker");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", "OPERATION OBJECTS"), node("h2", "", "操作物体"));
  const select = node("select", "add-select");
  select.add(new Option("添加物体...", ""));
  const chosen = new Set(app.draft.tasks.operation_object_ids);
  for (const object of app.state.objects.filter((item) => !chosen.has(item.object_id))) {
    select.add(new Option(
      object.object_id + " · " + (object.object_name_zh || object.object_name),
      object.object_id,
    ));
  }
  select.addEventListener("change", () => {
    if (select.value) app.draft.tasks.operation_object_ids.push(select.value);
    renderTaskEditor();
  });
  head.append(heading);
  if (app.editMode) head.append(select);
  const rows = node("div", "selected-records");
  for (const objectId of app.draft.tasks.operation_object_ids) {
    const asset = objectFor(objectId);
    const row = node("div", "selected-record");
    const inspect = button(
      asset ? asset.object_name_zh || asset.object_name : objectId,
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
    if (app.editMode) row.append(remove);
    rows.append(row);
  }
  if (!rows.children.length) emptyList(rows, "未设置操作物体");
  surface.append(head, rows);
  return surface;
}

function buildTaskScenePicker() {
  const draft = app.draft.tasks;
  const surface = node("section", "surface compact-picker");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", "SCENE COVERAGE"), node("h2", "", "场景与采集数量"));
  const select = node("select", "add-select");
  select.add(new Option("添加场景...", ""));
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
  if (app.editMode) head.append(select);
  const rows = node("div", "selected-records");
  draft.scene_ids.forEach((sceneId, index) => {
    const row = node("div", "selected-record scene-record");
    const link = button(sceneId, "", "text-button");
    link.addEventListener("click", () => jumpScene(draft.batch_id, sceneId));
    const count = input("number", "", draft.scene_epsiodes_count[index] || 1);
    count.min = "1";
    count.disabled = !app.editMode;
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
    row.append(link, count, node("small", "", "episodes"));
    if (app.editMode) row.append(remove);
    rows.append(row);
  });
  if (!rows.children.length) emptyList(rows, "未配置场景");
  const total = node("div", "total-bar");
  total.id = "task-total";
  total.append(node("span", "", "Total episode slots"), node("b", "", "0"));
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
    host.replaceChildren(node("div", "empty-state", "选择或新增物体"));
    return;
  }
  const editor = editorShell("objects", draft.object_id || "新物体", "SHARED OBJECT ASSET");
  const form = node("form", "form-grid inset surface");
  const idControl = input("text", "object_id", draft.object_id, "AST-0001");
  idControl.disabled = Boolean(app.original.objects) || !app.editMode;
  form.append(
    field("Object ID", idControl),
    field("Color", input("text", "color", draft.color, "white / #e8590c")),
    field("English name", input("text", "object_name", draft.object_name), true),
    field("中文名称", input("text", "object_name_zh", draft.object_name_zh), true),
    field("扫描状态", input("text", "scan_status", draft.scan_status), true),
    field("Photo directory", input("text", "photo_dir", draft.photo_dir), true),
  );
  if (!app.editMode) {
    form.querySelectorAll("input").forEach((control) => { control.disabled = true; });
  }
  form.addEventListener("input", () => {
    for (const name of [
      "object_id",
      "object_name",
      "object_name_zh",
      "scan_status",
      "color",
      "photo_dir",
    ]) {
      draft[name] = form.elements[name].value.trim();
    }
    editor.shell.querySelector("h1").textContent = draft.object_id || "新物体";
  });
  editor.body.append(form, buildPhotoGallery(draft, true));
  host.replaceChildren(editor.shell);
}

function buildPhotoGallery(object, editable = false) {
  const section = node("section", "surface photo-surface");
  const head = node("div", "section-head");
  const heading = node("div");
  heading.append(node("span", "eyebrow", "OBJECT PHOTOS"), node("h2", "", "实物照片"));
  if (editable && app.editMode) {
    const upload = button("上传照片", "", "button");
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
      image.alt = object.object_name_zh || object.object_name;
      figure.append(image, node("figcaption", "", filename));
      gallery.append(figure);
    }
  } else {
    const slot = node("div", "photo-slot");
    slot.append(node("b", "", "+"), node("span", "", "PHOTO SLOT"));
    gallery.append(slot);
  }
  section.append(head, gallery);
  return section;
}

function openObjectDialog(objectId) {
  const object = objectFor(objectId);
  if (!object) {
    showToast("未找到物体 " + objectId, true);
    return;
  }
  const body = $("object-dialog-body");
  const header = node("header", "object-modal-head");
  header.append(
    node("span", "eyebrow", object.object_id),
    node("h2", "", object.object_name_zh || object.object_name),
    node("p", "", object.object_name || "未填写英文名称"),
  );
  const meta = node("div", "object-modal-meta");
  meta.append(
    node("span", "", "扫描状态 · " + (object.scan_status || "未填写")),
    node("span", "", "颜色 · " + (object.color || "未标注")),
    node("span", "", "照片目录 · " + (object.photo_dir || "未绑定")),
  );
  body.replaceChildren(header, meta, buildPhotoGallery(object));
  $("object-dialog").showModal();
}

function renderInfo() {
  if (!app.state) return;
  const form = $("info-form");
  const plan = app.batch ? planFor(app.batch) : null;
  if (!plan) {
    const overview = node("div", "batch-overview full");
    overview.append(
      node("strong", "", "全部批次"),
      node("p", "", "在 Tasks 左栏选择一个计划批次后，可查看元数据并导出计划。"),
    );
    form.replaceChildren(overview);
    return;
  }
  const datasetName = input("text", "dataset_name", plan.info.dataset_name || "");
  const robotType = input("text", "robot_type", plan.info.robot_type || "");
  const collectionDir = input("text", "collection_dir", plan.info.collection_dir || "");
  datasetName.disabled = !app.editMode;
  robotType.disabled = !app.editMode;
  collectionDir.disabled = !app.editMode;
  const planDir = input("text", "", plan.dataset_dir);
  planDir.disabled = true;
  const collected = input("text", "", plan.collection.dataset_dir);
  collected.disabled = true;
  const cameras = buildCameraPositionLegend(app.batch);
  const save = button("保存元数据", "", "button primary");
  save.addEventListener("click", () => runWrite(save, async () => {
    await writeJson("/api/batches/" + encodeURIComponent(app.batch) + "/info", "PUT", {
      dataset_name: datasetName.value.trim(),
      robot_type: robotType.value.trim(),
      collection_dir: collectionDir.value.trim(),
    });
    await loadState();
  }, "批次元数据已保存"));
  form.replaceChildren(
    field("Dataset name", datasetName),
    field("Robot type", robotType),
    field("Plan directory", planDir, true),
    field("Collection directory override", collectionDir, true, "留空时按 dataset_name 自动发现"),
    field("Resolved collection directory", collected, true),
    cameras,
  );
  const actions = node("div", "field full");
  if (app.editMode) {
    actions.append(save);
    form.append(actions);
  }
}

function renderIssues() {
  if (!app.state) return;
  const host = $("issues-list");
  if (!app.state.issues.length) {
    const ok = node("div", "issues-ok");
    ok.append(node("b", "", "✓"), node("span", "", "当前范围未发现计划问题"));
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
    showToast("未找到场景 " + sceneId, true);
    return;
  }
  selectScene(scene);
  switchTab("scenes");
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
  showToast("请先在 Tasks 左栏选择一个计划批次", true);
  switchTab("tasks");
  return "";
}

function newEntity(kind) {
  if (!requireEditMode()) return;
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
    if (!draft.scene_id) throw new Error("Scene ID 不能为空");
    for (const placement of draft.placements) {
      if (!placement.object_id || !placement.position_ids.length) {
        throw new Error("每个物体组都需要物体和至少一个位置");
      }
    }
  } else if (kind === "tasks") {
    if (!draft.task_id || !draft.prompt_en) throw new Error("Task ID 和 English prompt 不能为空");
    if (!draft.scene_ids.length) throw new Error("任务至少需要一个场景");
    updateTaskTotal();
  } else if (!draft.object_id || (!draft.object_name && !draft.object_name_zh)) {
    throw new Error("Object ID 和至少一种名称不能为空");
  }
  return draft;
}


export {
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
};
