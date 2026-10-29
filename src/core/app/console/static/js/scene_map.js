function scenePlanGridPositions(plan, scene) {
  const configured = plan && Array.isArray(plan.positions)
    ? plan.positions : [];
  const positions = configured.map((position) => ({
    positionId: String(position.position_id || ""),
    x: position.x === null || position.x === "" ? NaN : Number(position.x),
    y: position.y === null || position.y === "" ? NaN : Number(position.y),
  })).filter((position) => position.positionId);
  const known = new Set(positions.map((position) => position.positionId));
  const groups = scene && Array.isArray(scene.placement_groups)
    ? scene.placement_groups : [];
  groups.flatMap((group) => group.position_ids || []).forEach((value) => {
    const positionId = String(value || "");
    if (!positionId || known.has(positionId)) return;
    known.add(positionId);
    positions.push({ positionId, x: NaN, y: NaN });
  });
  return positions;
}

function scenePlanDisplayGrid(plan) {
  // larybench display_grid uses world Y for columns and world X for rows.
  const displayGrid = plan && plan.display_grid;
  if (!displayGrid || typeof displayGrid !== "object") return null;
  const columnAxis = String(displayGrid.column_axis || "").trim().toLowerCase();
  const rowAxis = String(displayGrid.row_axis || "").trim().toLowerCase();
  const columns = Array.isArray(displayGrid.columns)
    ? displayGrid.columns.map(Number).filter(Number.isFinite) : [];
  const rows = Array.isArray(displayGrid.rows)
    ? displayGrid.rows.map(Number).filter(Number.isFinite) : [];
  if (!columns.length || !rows.length || columnAxis !== "y" || rowAxis !== "x") return null;
  const cellCount = columns.length * rows.length;
  const labels = Array.isArray(displayGrid.position_ids) ? displayGrid.position_ids : [];
  const flatLabels = labels.length && Array.isArray(labels[0]) ? labels.flat() : labels;
  return {
    columns: columns.length,
    rows: rows.length,
    columnAxis,
    rowAxis,
    columnValues: columns,
    rowValues: rows,
    labels: Array.from({length: cellCount}, (_, index) => (
      String(flatLabels[index] || `G${String(index + 1).padStart(2, "0")}`)
    )),
  };
}

function nearestGridIndex(values, target) {
  if (!values.length || !Number.isFinite(target)) return -1;
  return values.reduce((best, value, index) => (
    Math.abs(value - target) < Math.abs(values[best] - target) ? index : best
  ), 0);
}

function sceneCoordinateText(value) {
  const coordinates = Array.isArray(value) ? value : [];
  return coordinates.map((coordinate) => {
    const x = Number(coordinate.x);
    const y = Number(coordinate.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return "";
    return `${coordinate.positionId || "point"}:(${x.toFixed(4)},${y.toFixed(4)})`;
  }).filter(Boolean).join(" / ");
}

function placementPoseText(placement) {
  if (!Array.isArray(placement?.pose_xyzw) || placement.pose_xyzw.length !== 7) return "";
  const values = placement.pose_xyzw.map((value) => Number(value));
  if (values.some((value) => !Number.isFinite(value))) return "";
  return `pose=(${values.map((value) => value.toFixed(4)).join(",")})`;
}

function scenePlanGridGeometry(plan, positions) {
  const count = Math.max(1, positions.length);
  const fallbackColumns = Math.ceil(Math.sqrt(count));
  const fallback = {
    columns: fallbackColumns,
    rows: Math.ceil(count / fallbackColumns),
    cells: positions.map((position, index) => ({
      ...position,
      cellId: position.positionId,
      label: position.positionId,
      positionIds: [position.positionId],
      positionCoordinates: [{positionId: position.positionId, x: position.x, y: position.y}],
      column: index % fallbackColumns + 1,
      row: Math.floor(index / fallbackColumns) + 1,
    })),
  };
  const displayGrid = scenePlanDisplayGrid(plan);
  if (displayGrid) {
    const cells = displayGrid.labels.map((label, index) => ({
      cellId: label,
      label,
      positionIds: [],
      positionCoordinates: [],
      column: index % displayGrid.columns + 1,
      row: Math.floor(index / displayGrid.columns) + 1,
    }));
    positions.forEach((position) => {
      const columnValue = position[displayGrid.columnAxis];
      const rowValue = position[displayGrid.rowAxis];
      const column = nearestGridIndex(displayGrid.columnValues, columnValue);
      const row = nearestGridIndex(displayGrid.rowValues, rowValue);
      if (column < 0 || row < 0) return;
      const cell = cells[row * displayGrid.columns + column];
      cell.positionIds.push(position.positionId);
      cell.positionCoordinates.push({positionId: position.positionId, x: position.x, y: position.y});
    });
    return {columns: displayGrid.columns, rows: displayGrid.rows, cells};
  }
  if (!positions.length || positions.some((position) => (
    !Number.isFinite(position.x) || !Number.isFinite(position.y)
  ))) return fallback;

  const xValues = [...new Set(positions.map((position) => position.x))].sort((a, b) => a - b);
  const yValues = [...new Set(positions.map((position) => position.y))].sort((a, b) => a - b);
  if (xValues.length * yValues.length > positions.length * 2) return fallback;
  return {
    columns: xValues.length,
    rows: yValues.length,
    cells: positions.map((position) => ({
      ...position,
      cellId: position.positionId,
      label: position.positionId,
      positionIds: [position.positionId],
      positionCoordinates: [{positionId: position.positionId, x: position.x, y: position.y}],
      column: xValues.indexOf(position.x) + 1,
      row: yValues.indexOf(position.y) + 1,
    })),
  };
}

function normalizedObjectText(value) {
  return String(value || "").normalize("NFKC").toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

function taskObjectReferences(task) {
  const references = (task && Array.isArray(task.operation_object_ids))
    ? task.operation_object_ids : [];
  return references.concat(String(task && task.operation_object || "").split("/"))
    .map(normalizedObjectText).filter(Boolean);
}

function placementAliases(placement) {
  return ["object_id", "name", "name_zh", "name_en"]
    .map((field) => normalizedObjectText(placement && placement[field])).filter(Boolean);
}

function taskHighlightsPlacement(task, placement) {
  if (!task || !placement) return false;
  const objectIds = Array.isArray(task.operation_object_ids)
    ? task.operation_object_ids.map((value) => String(value || "").trim()).filter(Boolean) : [];
  if (objectIds.length) {
    return objectIds.includes(String(placement.object_id || "").trim());
  }
  const aliases = placementAliases(placement);
  const references = taskObjectReferences(task);
  if (references.some((reference) => aliases.some((alias) => (
    reference === alias || reference.includes(alias) || alias.includes(reference)
  )))) return true;
  const prompt = normalizedObjectText(`${task.prompt_en || ""} ${task.prompt_zh || ""}`);
  return aliases.some((alias) => prompt.includes(alias));
}

function renderSceneReference(host, reference, version) {
  const referenceKey = JSON.stringify([version, reference]);
  if (host.dataset.referenceKey === referenceKey && host.children.length) return;
  const panel = document.createElement("div");
  const label = document.createElement("span");
  const link = document.createElement("a");
  const image = document.createElement("img");
  const status = document.createElement("span");
  panel.className = "collect-scene-reference";
  label.className = "collect-scene-reference-label";
  label.textContent = "参考图 / REFERENCE IMAGE";
  link.className = "collect-scene-reference-link";
  link.href = reference.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  image.className = "collect-scene-reference-image";
  image.alt = "当前数据点参考图";
  status.className = "collect-scene-reference-status";
  status.textContent = "图片加载失败 / Image unavailable";
  status.hidden = true;
  image.onerror = () => {
    link.hidden = true;
    status.hidden = false;
  };
  link.append(image);
  panel.append(label, link, status);
  host.replaceChildren(panel);
  delete host.dataset.layoutKey;
  host.dataset.referenceKey = referenceKey;
  image.src = reference.url;
}

function renderSceneGridCells(host, plan, scene) {
  const geometry = scenePlanGridGeometry(plan, scenePlanGridPositions(plan, scene));
  const layoutKey = geometry.cells.map((cell) => (
    `${cell.cellId}:${cell.column}:${cell.row}:${JSON.stringify(cell.positionCoordinates)}`
  )).join("|");
  host.style.setProperty("--scene-grid-columns", String(geometry.columns));
  host.style.setProperty("--scene-grid-rows", String(geometry.rows));
  if (host.dataset.layoutKey === layoutKey) return;
  host.dataset.layoutKey = layoutKey;
  host.replaceChildren(...geometry.cells.map((position) => {
    const cell = document.createElement("div");
    const label = document.createElement("span");
    const object = document.createElement("b");
    const photo = document.createElement("img");
    cell.className = "collect-scene-cell empty";
    cell.dataset.positionIds = position.positionIds.join("|");
    cell.dataset.positionCoordinates = JSON.stringify(position.positionCoordinates || []);
    cell.style.gridColumn = String(position.column);
    cell.style.gridRow = String(position.row);
    const coordinates = sceneCoordinateText(position.positionCoordinates);
    cell.title = [position.positionIds.join(" / ") || position.cellId, coordinates]
      .filter(Boolean).join(" · ");
    label.textContent = position.label || position.cellId;
    photo.className = "collect-scene-object-photo";
    photo.alt = "";
    photo.hidden = true;
    photo.onerror = () => { photo.hidden = true; };
    cell.append(label, photo, object);
    return cell;
  }));
}

export function renderSceneGrid(host, plan, scene, task) {
  const name = scene?.reference_image_names?.[0];
  const reference = (plan?.reference_images || []).find((image) => image.name === name);
  host.dataset.displayMode = reference ? "reference" : scene?.placements?.length ? "grid" : "prompt";
  if (host.dataset.displayMode === "prompt") {
    host.replaceChildren();
    delete host.dataset.layoutKey;
    delete host.dataset.referenceKey;
    return;
  }
  if (reference) {
    renderSceneReference(host, reference, plan.version);
    return;
  }
  delete host.dataset.referenceKey;
  renderSceneGridCells(host, plan, scene);
  const byPosition = new Map();
  (scene && scene.placements || []).forEach((placement) => {
    const positionId = String(placement.position_id || "");
    if (!positionId) return;
    const values = byPosition.get(positionId) || [];
    values.push(placement);
    byPosition.set(positionId, values);
  });
  host.querySelectorAll(".collect-scene-cell").forEach((cell) => {
    const positionIds = String(cell.dataset.positionIds || "").split("|").filter(Boolean);
    const placements = positionIds.flatMap((positionId) => byPosition.get(positionId) || []);
    const names = [...new Set(placements.map((placement) => String(placement.name_en || (/[^\x00-\x7F]/.test(placement.name || "") ? placement.object_id : placement.name) || "")))]
      .filter(Boolean);
    const poses = [...new Set(placements.map(placementPoseText).filter(Boolean))];
    cell.classList.toggle("empty", placements.length === 0);
    cell.classList.toggle("fixed", placements.length > 0);
    cell.classList.toggle(
      "task-relevant", placements.some((placement) => taskHighlightsPlacement(task, placement))
    );
    const object = cell.querySelector("b");
    object.textContent = names.join(" / ");
    object.title = [names.join(" / "), poses.join(" / ")].filter(Boolean).join(" · ");
    const photo = cell.querySelector("img");
    const photoUrl = placements.find((placement) => placement.photo_url)?.photo_url || "";
    if (photoUrl) {
      if (photo.getAttribute("src") !== photoUrl) {
        photo.hidden = false;
        photo.src = photoUrl;
      }
    } else {
      photo.hidden = true;
      photo.removeAttribute("src");
    }
    const positionLabel = positionIds.join(" / ") || cell.querySelector("span")?.textContent || "";
    const coordinates = JSON.parse(cell.dataset.positionCoordinates);
    const coordinateText = sceneCoordinateText(coordinates);
    cell.title = [positionLabel, coordinateText, names.join(" / "), poses.join(" / ")]
      .filter(Boolean).join(" · ");
  });
}
