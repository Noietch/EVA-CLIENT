"""One scene plan, one canonical collection-slot order.

The console's collect panel and the dataset service's QC grid expand the same
task-set directory (``tasks.csv`` / ``scene.csv`` / ``layout.yaml`` /
``objects.csv``) into collection slots. Both parse it here and order those slots
scene -> task -> round, with each scene's tasks sorted hand-first and then by
their object's row-major layout position, so one slot id occupies the same tile
in either panel.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanSlot:
    """One capture target in the order both panels display it."""

    slot_id: str
    ordinal: int
    task_index: int
    task_id: str
    task: str
    task_zh: str
    scene_id: str
    scene_label: str
    round_index: int
    round_total: int


def read_plan_csv(root: Path, filename: str) -> list[dict[str, str]]:
    path = root / filename
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error) as exc:
        logger.warning("Unable to read scene plan %s: %s", path, exc)
        return []


def read_plan_yaml(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.warning("Unable to read scene plan %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def optional_float(value: Any) -> float | None:
    try:
        text = "" if value is None else str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def plan_placements(value: Any) -> list[dict[str, Any]]:
    try:
        payload = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid scene placements JSON: %s", exc)
        return []
    if not isinstance(payload, list):
        return []
    placements = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id", "") or "").strip()
        position_ids = item.get("position_ids")
        if not object_id or not isinstance(position_ids, list):
            continue
        position_ids = [str(position_id or "").strip() for position_id in position_ids]
        position_ids = [position_id for position_id in position_ids if position_id]
        if position_ids:
            placements.append(
                {
                    "position_ids": position_ids,
                    "object_id": object_id,
                }
            )
    return placements


def plan_positions(layout_doc: dict[str, Any]) -> list[dict[str, Any]]:
    positions = []
    unit = str(layout_doc.get("unit", "") or "")
    frame = str(layout_doc.get("coordinate_frame", "") or "")
    status = str(layout_doc.get("calibration_status", "unverified") or "unverified")
    for point in layout_doc.get("sampling_points", []):
        if not isinstance(point, dict):
            continue
        position_id = str(point.get("position_id", point.get("id", "")) or "").strip()
        if not position_id:
            continue
        positions.append(
            {
                "position_id": position_id,
                "x": optional_float(point.get("x")),
                "y": optional_float(point.get("y")),
                "unit": str(point.get("unit", unit) or unit),
                "coordinate_frame": str(point.get("coordinate_frame", frame) or frame),
                "calibration_status": str(point.get("calibration_status", status) or status),
            }
        )
    return positions


def plan_objects(
    root: Path, photo_url: Callable[[str, str], str] | None = None
) -> dict[str, dict[str, Any]]:
    """Objects named by the plan, keyed by object id.

    ``photo_url`` builds a caller-specific preview URL from an object id and a
    photo filename; the dataset service reads objects through its own asset
    catalog instead.
    """
    payload = {}
    info = read_plan_yaml(root, "info.yaml")
    objects_file = str(info.get("objects_file", "objects.csv") or "objects.csv")
    objects_path = (root / objects_file).resolve()
    photo_root = objects_path.parent / "object_photos"
    for row in read_plan_csv(root, objects_file):
        object_id = str(row.get("object_id") or row.get("物品代码") or "").strip()
        if not object_id:
            continue
        photo_dir = str(
            row.get("photo_dir") or row.get("照片目录") or row.get("名称") or ""
        ).strip()
        photos = sorted((photo_root / photo_dir).glob("*")) if photo_dir else []
        photo = next(
            (
                path
                for path in photos
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ),
            None,
        )
        payload[object_id] = {
            "object_id": object_id,
            "name": str(
                row.get("object_name_zh") or row.get("名称") or row.get("object_name") or object_id
            ),
            "name_zh": str(row.get("object_name_zh") or row.get("名称") or ""),
            "name_en": str(row.get("object_name") or row.get("英文名") or ""),
            "color": str(row.get("color", "") or ""),
            "photo_url": photo_url(object_id, photo.name) if photo and photo_url else "",
        }
    return payload


def plan_scenes(
    rows: list[dict[str, str]],
    objects: dict[str, dict[str, Any]],
    calibration_status: str,
) -> list[dict[str, Any]]:
    scenes: dict[str, dict[str, Any]] = {}
    for row in rows:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            continue
        scene = scenes.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "placements": [],
                "placement_groups": [],
                "calibration_status": calibration_status,
            },
        )
        for group_index, placement in enumerate(plan_placements(row.get("placements"))):
            object_id = placement["object_id"]
            position_ids = placement["position_ids"]
            obj = objects.get(object_id, {"name": object_id, "color": ""})
            group_id = f"{scene_id}:{group_index}"
            scene["placement_groups"].append(
                {
                    "group_id": group_id,
                    "position_ids": position_ids,
                    "object_id": object_id,
                    "name": obj["name"],
                    "name_zh": obj.get("name_zh", ""),
                    "name_en": obj.get("name_en", ""),
                    "color": obj.get("color", ""),
                    "photo_url": obj.get("photo_url", ""),
                }
            )
            for position_index, position_id in enumerate(position_ids):
                scene["placements"].append(
                    {
                        "position_id": position_id,
                        "object_id": object_id,
                        "name": obj["name"],
                        "name_zh": obj.get("name_zh", ""),
                        "name_en": obj.get("name_en", ""),
                        "color": obj.get("color", ""),
                        "photo_url": obj.get("photo_url", ""),
                        "group_id": group_id,
                        "group_position_index": position_index,
                        "group_size": len(position_ids),
                        "group_position_ids": position_ids,
                    }
                )
    return list(scenes.values())


def plan_tasks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        scene_ids = [value for value in str(row.get("scene_ids", "")).split(";") if value]
        if not task_id or not scene_ids:
            continue
        episode_counts = str(row.get("scene_epsiodes_count", "")).split(";")
        tasks.append(
            {
                "task_id": task_id,
                "action": str(row.get("action", "") or ""),
                "category": str(row.get("category", "") or ""),
                "operation_object": str(
                    row.get("operation_object") or row.get("操作对象") or ""
                ).strip(),
                "operation_object_ids": [
                    value.strip()
                    for value in str(row.get("operation_object_ids", "")).split(";")
                    if value.strip()
                ],
                "prompt_en": str(row.get("prompt_en", "") or ""),
                "prompt_zh": str(row.get("prompt_zh", "") or ""),
                "display_name": str(
                    row.get("prompt_zh") or row.get("prompt_en") or task_id
                ).strip(),
                "scene_ids": scene_ids,
                "scene_epsiodes_count": [int(value) for value in episode_counts if value.isdigit()],
                "total_epsiodes_count": int(row.get("total_epsiodes_count", 0) or 0),
            }
        )
    return tasks


def scene_plan_from_dir(
    root: Path, photo_url: Callable[[str, str], str] | None = None
) -> dict[str, Any]:
    """Parse one normalized scene catalog directory into its plan payload."""
    task_rows = read_plan_csv(root, "tasks.csv")
    layout_doc = read_plan_yaml(root, "layout.yaml")
    layout_unit = str(layout_doc.get("unit", "") or "")
    layout_frame = str(layout_doc.get("coordinate_frame", "") or "")
    layout_status = str(layout_doc.get("calibration_status", "unverified") or "unverified")
    positions = plan_positions(layout_doc)
    objects = plan_objects(root, photo_url)
    raw_bounds = layout_doc.get("bounds")
    bounds = raw_bounds if isinstance(raw_bounds, dict) else {}
    return {
        "ok": True,
        "source": str(root / "tasks.csv") if task_rows else "",
        "layout_id": str(layout_doc.get("layout_id", "") or ""),
        "coordinate_frame": layout_frame,
        "bounds": {
            "width": optional_float(bounds.get("width")),
            "height": optional_float(bounds.get("height")),
            "unit": layout_unit,
        },
        "calibrated": bool(positions)
        and all(
            row["x"] is not None
            and row["y"] is not None
            and row["calibration_status"] == "verified"
            for row in positions
        ),
        "positions": positions,
        "objects": list(objects.values()),
        "scenes": plan_scenes(read_plan_csv(root, "scene.csv"), objects, layout_status),
        "tasks": plan_tasks(task_rows),
    }


def scene_label(scene: dict[str, Any]) -> str:
    labels: list[str] = []
    for placement in scene.get("placements") or []:
        name = str(placement.get("name") or "").strip()
        positions = placement.get("group_position_ids") or [placement.get("position_id")]
        position_label = "/".join(str(value) for value in positions if value)
        label = f"{name} · {position_label}" if name and position_label else name
        if label and label not in labels:
            labels.append(label)
    return " / ".join(labels) or str(scene.get("scene_id") or "Scene")


def position_sort_keys(scene_plan: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Build row-major layout keys without inferring spatial order from position IDs."""
    keys: dict[str, tuple[Any, ...]] = {}
    for index, position in enumerate(scene_plan.get("positions") or []):
        if not isinstance(position, dict):
            continue
        position_id = str(position.get("position_id") or "").strip()
        if not position_id:
            continue
        try:
            x = float(position["x"])
            y = float(position["y"])
        except (KeyError, TypeError, ValueError):
            x = y = math.nan
        if math.isfinite(x) and math.isfinite(y):
            keys[position_id] = (0, y, x, index, position_id)
        else:
            # An uncalibrated point still has a stable authored order, but it is
            # deliberately ranked after points with usable coordinates.
            keys[position_id] = (1, index, position_id)
    return keys


def task_hand_priority(task: dict[str, Any]) -> int:
    """Prefer the first-mentioned operating hand: left, then right, then unknown."""
    text = " ".join(str(task.get(field) or "") for field in ("prompt_en", "prompt_zh"))
    match = re.search(r"left\s+(?:arm|hand)|right\s+(?:arm|hand)|左手|左臂|右手|右臂", text, re.I)
    if match is None:
        return 2
    return 0 if match.group(0).lower().startswith(("left", "左")) else 1


def placement_matches_reference(placement: dict[str, Any], reference: str) -> bool:
    reference = str(reference).strip()
    if not reference:
        return False
    aliases = {
        str(placement.get(field) or "").strip()
        for field in ("object_id", "name", "name_zh", "name_en")
    }
    aliases.discard("")
    return any(reference == alias or reference in alias or alias in reference for alias in aliases)


def task_spatial_sort_key(
    task: dict[str, Any],
    scene: dict[str, Any],
    position_keys: dict[str, tuple[Any, ...]],
    task_index: int,
) -> tuple[Any, ...]:
    """Return the scene task's hand-first, then row-major object ordering key."""
    references = [
        str(value).strip() for value in task.get("operation_object_ids") or [] if str(value).strip()
    ]
    references.extend(
        item.strip()
        for item in str(task.get("operation_object") or task.get("operation_objects") or "").split(
            "/"
        )
        if item.strip()
    )
    placements = scene.get("placements") or []
    matched_keys: list[tuple[Any, ...]] = []
    for reference in references:
        matching_positions: list[tuple[Any, ...]] = []
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            object_id = str(placement.get("object_id") or "").strip()
            if reference != object_id and not placement_matches_reference(placement, reference):
                continue
            matching_positions.extend(
                position_keys[position_id]
                for position_id in placement.get("group_position_ids")
                or placement.get("position_ids")
                or [placement.get("position_id")]
                if position_id in position_keys
            )
        if matching_positions:
            matched_keys.append(min(matching_positions))
    if matched_keys:
        return (0, task_hand_priority(task), tuple(matched_keys), task_index)
    return (0, task_hand_priority(task), (), task_index)


def plan_slots(
    scene_plan: dict[str, Any],
    entries: Sequence[Sequence[Any]] | None = None,
    bindings: dict[str, str] | None = None,
) -> list[PlanSlot]:
    """Expand a plan into slots in the order the collect panel and QC grid show.

    ``entries`` is the mounted task set's prompt list; when given, tasks outside
    it are left out and its indexes break spatial ties. Without it every task in
    the plan is kept and the plan's own task order breaks ties.
    """
    prompt_indices = {str(entry[0]): index for index, entry in enumerate(entries or ())}
    bindings = bindings or {}
    tasks = list(scene_plan.get("tasks") or [])
    position_keys = position_sort_keys(scene_plan)
    slots: list[PlanSlot] = []
    for scene in scene_plan.get("scenes") or []:
        scene_id = str(scene.get("scene_id") or "").strip()
        if not scene_id:
            continue
        scene_tasks = [
            (index, task)
            for index, task in enumerate(tasks)
            if scene_id in list(task.get("scene_ids") or [])
        ]
        scene_tasks.sort(
            key=lambda item: task_spatial_sort_key(item[1], scene, position_keys, item[0])
        )
        for plan_index, task in scene_tasks:
            if bindings and str(task.get("task_id") or "") not in bindings:
                continue
            prompt = str(
                bindings.get(str(task.get("task_id") or "")) or task.get("prompt_en") or ""
            )
            prompt = prompt.strip()
            if entries is None:
                task_index = plan_index
            else:
                task_index = prompt_indices.get(prompt)
                if task_index is None:
                    continue
            scene_ids = list(task.get("scene_ids") or [])
            scene_index = scene_ids.index(scene_id)
            counts = list(task.get("scene_epsiodes_count") or [])
            round_total = int(counts[scene_index]) if scene_index < len(counts) else 0
            task_id = str(task.get("task_id") or prompt).strip()
            label = scene_label(scene)
            for round_index in range(max(0, round_total)):
                slots.append(
                    PlanSlot(
                        slot_id=f"{task_id}:{scene_id}:{round_index}",
                        ordinal=len(slots),
                        task_index=task_index,
                        task_id=task_id,
                        task=prompt,
                        task_zh=str(task.get("prompt_zh") or "").strip(),
                        scene_id=scene_id,
                        scene_label=label,
                        round_index=round_index,
                        round_total=round_total,
                    )
                )
    return slots
