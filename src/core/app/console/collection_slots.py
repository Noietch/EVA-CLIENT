"""Fixed collection-slot planning and persisted skip state."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import ConfigDict

_STATE_LOCK = threading.RLock()
_STATE_FILE = "collection_slots.json"


@dataclass(frozen=True)
class CollectionSlot:
    """One immutable capture target in Scene -> Task -> Round order."""

    slot_id: str
    ordinal: int
    dataset: str
    task_index: int
    task_id: str
    task: str
    task_zh: str
    scene_id: str
    scene_label: str
    round_index: int
    round_total: int
    unbounded: bool = False


@dataclass(frozen=True)
class CollectionSlotState:
    """Small persisted operator override layered on top of the fixed plan."""

    deferred: list[str]
    selected_slot_id: str = ""
    selected_episode_index: int | None = None
    selected_manually: bool = False


def _scene_label(scene: dict[str, Any]) -> str:
    labels: list[str] = []
    for placement in scene.get("placements") or []:
        name = str(placement.get("name") or "").strip()
        positions = placement.get("group_position_ids") or [placement.get("position_id")]
        position_label = "/".join(str(value) for value in positions if value)
        label = f"{name} · {position_label}" if name and position_label else name
        if label and label not in labels:
            labels.append(label)
    return " / ".join(labels) or str(scene.get("scene_id") or "Scene")


def build_collection_slots(
    config: ConfigDict,
    scene_plan: dict[str, Any],
    dataset: str,
) -> list[CollectionSlot]:
    """Expand a dataset plan into stable slots in operator workflow order."""
    entries = list(config.collection.tasks.get(dataset) or [])
    prompt_indices: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, (list, tuple)) and entry:
            prompt_indices.setdefault(str(entry[0]), index)

    tasks = list(scene_plan.get("tasks") or [])
    slots: list[CollectionSlot] = []
    for scene in scene_plan.get("scenes") or []:
        scene_id = str(scene.get("scene_id") or "").strip()
        if not scene_id:
            continue
        for task in tasks:
            prompt = str(task.get("prompt_en") or "").strip()
            task_index = prompt_indices.get(prompt)
            scene_ids = list(task.get("scene_ids") or [])
            if task_index is None or scene_id not in scene_ids:
                continue
            scene_index = scene_ids.index(scene_id)
            counts = list(task.get("scene_epsiodes_count") or [])
            round_total = int(counts[scene_index]) if scene_index < len(counts) else 0
            task_id = str(task.get("task_id") or prompt).strip()
            for round_index in range(max(0, round_total)):
                slot_id = f"{task_id}:{scene_id}:{round_index}"
                slots.append(
                    CollectionSlot(
                        slot_id=slot_id,
                        ordinal=len(slots),
                        dataset=dataset,
                        task_index=task_index,
                        task_id=task_id,
                        task=prompt,
                        task_zh=str(task.get("prompt_zh") or "").strip(),
                        scene_id=scene_id,
                        scene_label=_scene_label(scene),
                        round_index=round_index,
                        round_total=round_total,
                    )
                )
    if slots:
        return slots

    # Inline collection configs do not have scene.csv/tasks.csv, but they still
    # use the same slot picker. A -1 target means the task remains available for
    # repeated captures; finite targets expand into one slot per episode.
    for task_index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        prompt = str(entry[0]).strip()
        if not prompt:
            continue
        target = int(entry[1])
        unbounded = target == -1
        round_total = target if target > 0 else 1
        task_id = f"INLINE-{task_index}"
        for round_index in range(round_total):
            slots.append(
                CollectionSlot(
                    slot_id=f"{task_id}:{round_index}",
                    ordinal=len(slots),
                    dataset=dataset,
                    task_index=task_index,
                    task_id=task_id,
                    task=prompt,
                    task_zh="",
                    scene_id="DEFAULT",
                    scene_label="DEFAULT",
                    round_index=round_index,
                    round_total=round_total,
                    unbounded=unbounded,
                )
            )
    return slots


def load_slot_state(dataset_dir: Path | None) -> CollectionSlotState:
    if dataset_dir is None:
        return CollectionSlotState([])
    path = dataset_dir / "meta" / _STATE_FILE
    with _STATE_LOCK:
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return CollectionSlotState([])
    deferred = payload.get("deferred") if isinstance(payload, dict) else None
    selected_slot_id = payload.get("selected_slot_id") if isinstance(payload, dict) else ""
    selected_episode_index = (
        payload.get("selected_episode_index") if isinstance(payload, dict) else None
    )
    return CollectionSlotState(
        [str(slot_id) for slot_id in deferred] if isinstance(deferred, list) else [],
        str(selected_slot_id or ""),
        selected_episode_index if type(selected_episode_index) is int else None,
        bool(payload.get("selected_manually", selected_episode_index is not None))
        if isinstance(payload, dict)
        else False,
    )


def save_slot_state(dataset_dir: Path, state: CollectionSlotState) -> None:
    path = dataset_dir / "meta" / _STATE_FILE
    with _STATE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "deferred": state.deferred,
                    "selected_slot_id": state.selected_slot_id,
                    "selected_episode_index": state.selected_episode_index,
                    "selected_manually": state.selected_manually,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        temporary.replace(path)


def _episode_outcome(episode: dict[str, Any]) -> str:
    if str(episode.get("status") or "") != "saved":
        return "pending"
    if (
        str(episode.get("quality") or "green").lower() == "red"
        or str(episode.get("qc_verdict") or "").lower() == "fail"
    ):
        return "rejected"
    return "usable"


def _episode_indices(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    by_slot: dict[str, dict[str, Any]] = {}
    by_legacy_target: dict[tuple[str, str, int], dict[str, Any]] = {}
    for episode in episodes:
        slot_id = str(episode.get("slot_id") or "")
        if slot_id:
            current = by_slot.get(slot_id)
            if current is None or int(episode.get("episode_index", -1)) > int(
                current.get("episode_index", -1)
            ):
                by_slot[slot_id] = episode
            continue
        try:
            key = (
                str(episode.get("task") or episode.get("prompt") or ""),
                str(episode.get("scene_id") or ""),
                int(episode["scene_round"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        current = by_legacy_target.get(key)
        if current is None or int(episode.get("episode_index", -1)) > int(
            current.get("episode_index", -1)
        ):
            by_legacy_target[key] = episode
    return by_slot, by_legacy_target


def collection_slot_status(
    slots: list[CollectionSlot],
    episodes: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    slot_state: CollectionSlotState,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, int]]:
    """Resolve completion, repair, deferred, and current states for the plan."""
    valid_ids = {slot.slot_id for slot in slots}
    deferred = [slot_id for slot_id in slot_state.deferred if slot_id in valid_ids]
    deferred_set = set(deferred)
    episode_by_slot, legacy_episode_by_target = _episode_indices(
        episodes + [item for item in queue if item.get("status") == "saved"]
    )
    saving_ids = {
        str(item.get("slot_id") or "")
        for item in queue
        if item.get("status") in {"queued", "saving"}
    }
    failed_ids = {
        str(item.get("slot_id") or "") for item in queue if item.get("status") == "failed"
    }
    rows: list[dict[str, Any]] = []
    unresolved_regular: list[dict[str, Any]] = []
    unresolved_deferred: dict[str, dict[str, Any]] = {}
    counts = {"complete": 0, "rejected": 0, "deferred": 0, "pending": 0}

    for slot in slots:
        episode = episode_by_slot.get(slot.slot_id) or legacy_episode_by_target.get(
            (slot.task, slot.scene_id, slot.round_index)
        )
        outcome = (
            "pending" if slot.unbounded else (_episode_outcome(episode) if episode else "pending")
        )
        if slot.slot_id in saving_ids:
            state = "saving"
            counts["pending"] += 1
        elif outcome == "usable":
            state = "complete"
            counts["complete"] += 1
        elif slot.slot_id in deferred_set:
            state = "deferred"
            counts["deferred"] += 1
        elif slot.slot_id in failed_ids or outcome == "rejected":
            state = "rejected"
            counts["rejected"] += 1
        else:
            state = "pending"
            counts["pending"] += 1
        row = {**vars(slot), "state": state, "episode": episode}
        rows.append(row)
        if state not in {"complete", "saving"}:
            if state == "deferred":
                unresolved_deferred[slot.slot_id] = row
            else:
                unresolved_regular.append(row)

    unresolved = {
        row["slot_id"]: row for row in [*unresolved_regular, *unresolved_deferred.values()]
    }
    active = unresolved.get(slot_state.selected_slot_id)
    if active is not None and active["state"] in {"rejected", "deferred"}:
        if not (
            slot_state.selected_manually
            and (active["episode"] or {}).get("episode_index") == slot_state.selected_episode_index
        ):
            active = None
    # A manual retake stays selected only until a newer attempt exists.
    if (
        active is None
        and slot_state.selected_manually
        and slot_state.selected_episode_index is not None
    ):
        active = next(
            (
                row
                for row in rows
                if row["slot_id"] == slot_state.selected_slot_id
                and row["state"] == "complete"
                and (row["episode"] or {}).get("episode_index") == slot_state.selected_episode_index
            ),
            None,
        )
    if active is None:
        # Failed captures stay red until the operator explicitly selects a retake.
        active = next((row for row in unresolved_regular if row["state"] == "pending"), None)
    if active is not None:
        active["repair"] = active["state"] in {"complete", "deferred", "rejected"}
        active["state"] = "active"
    counts["total"] = len(slots)
    return rows, active, counts


def defer_active_slot(
    dataset_dir: Path,
    active_slot_id: str,
    state: CollectionSlotState,
) -> CollectionSlotState:
    """Move the current slot to the tail of the deferred repair queue."""
    ordered = [slot_id for slot_id in state.deferred if slot_id != active_slot_id]
    ordered.append(active_slot_id)
    updated = CollectionSlotState(ordered)
    save_slot_state(dataset_dir, updated)
    return updated


def select_collection_slot(
    dataset_dir: Path,
    slot_id: str,
    state: CollectionSlotState,
    *,
    episode_index: int | None = None,
    manual: bool = False,
) -> CollectionSlotState:
    """Select a capture target, optionally retaining a completed episode for retake."""
    updated = CollectionSlotState(state.deferred, slot_id, episode_index, manual)
    save_slot_state(dataset_dir, updated)
    return updated
