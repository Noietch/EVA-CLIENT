"""Join task-plan batches to collected LeRobot episodes for QC review."""

from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from core.utils.lerobot import LeRobotDatasetIO
from robots.utils import UrdfScene
from tools.datasets.assets import ObjectCatalog
from tools.datasets.store import ConflictError, RecordNotFoundError, TaskSetStore

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PLAN_CACHE_SECONDS = 2.0
PLAN_PAYLOAD_KEYS = ("batch_id", "info", "layout", "dataset_dir", "collection")
STATE_CACHE_VERSION = 2
QC_REASONS = {"image_quality", "trajectory_quality", "task_mismatch", "other"}


class PlanCatalog:
    """Expose existing editor CRUD plus task-centric collection review."""

    def __init__(
        self,
        plans_root: str | Path,
        assets_root: str | Path,
        collection_root: str | Path,
    ) -> None:
        self.plans_root = Path(plans_root).expanduser().resolve()
        self.collection_root = Path(collection_root).expanduser().resolve()
        self.plans_root.mkdir(parents=True, exist_ok=True)
        self.collection_root.mkdir(parents=True, exist_ok=True)
        self.assets = ObjectCatalog(assets_root, [])
        self._robot_scenes: dict[str, tuple[Any, UrdfScene]] = {}
        self._robot_meta_cache: dict[str, dict[str, Any]] = {}
        self._mesh_cache: dict[tuple[str, str], bytes] = {}
        self._lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._plan_cache: dict[str, tuple[float, str, dict[str, Any]]] = {}
        self._summary_cache: dict[str, tuple[float, str, dict[str, Any]]] = {}
        self._db_path = self.collection_root / ".dataset_cache.sqlite3"
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db_lock = threading.RLock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS state_cache ("
            "cache_key TEXT PRIMARY KEY, updated REAL NOT NULL, payload TEXT NOT NULL, "
            "cache_version INTEGER NOT NULL DEFAULT 1, source_token TEXT NOT NULL DEFAULT ''"
            ")"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(state_cache)").fetchall()}
        if "cache_version" not in columns:
            self._db.execute(
                "ALTER TABLE state_cache ADD COLUMN cache_version INTEGER NOT NULL DEFAULT 1"
            )
        if "source_token" not in columns:
            self._db.execute(
                "ALTER TABLE state_cache ADD COLUMN source_token TEXT NOT NULL DEFAULT ''"
            )
        self._db.commit()

    def state(self, batch: str = "", robot_type: str = "") -> dict[str, Any]:
        batches = self.batch_ids()
        robot_type = str(robot_type).strip()
        source_token = self._source_token(batches)
        cache_key = f"v{STATE_CACHE_VERSION}:{robot_type}::{batch}:{source_token}"
        with self._db_lock:
            cached = self._db.execute(
                "SELECT updated, payload FROM state_cache "
                "WHERE cache_key = ? AND cache_version = ?",
                (cache_key, STATE_CACHE_VERSION),
            ).fetchone()
        if cached and time.time() - float(cached[0]) < 300:
            return json.loads(cached[1])

        summaries = [self._batch_summary(value) for value in batches]
        self._attach_yam_benchmarks(summaries)
        robot_types = sorted(
            {
                str(summary["robot_type"]).strip()
                for summary in summaries
                if str(summary["robot_type"]).strip()
            }
        )
        if batch == "__all__":
            selected = [
                summary["batch_id"]
                for summary in summaries
                if not robot_type or summary["robot_type"] == robot_type
            ]
        elif batch:
            batch_id = self._batch_id(batch)
            if not any(summary["batch_id"] == batch_id for summary in summaries):
                raise RecordNotFoundError(batch_id)
            selected = [
                batch_id
                for summary in summaries
                if summary["batch_id"] == batch_id
                and (not robot_type or summary["robot_type"] == robot_type)
            ]
        else:
            # The landing page is a dashboard. Return summaries only and
            # defer the expensive episode join until a user opens a batch. A
            # robot filter is an explicit catalog request and still returns
            # its task rows for the editor's filtered view.
            selected = (
                [
                    summary["batch_id"]
                    for summary in summaries
                    if summary["robot_type"] == robot_type
                ]
                if robot_type
                else []
            )
        plan_states = [self._plan_state(value) for value in selected]
        tasks = [task for plan in plan_states for task in plan["tasks"]]
        scenes = [scene for plan in plan_states for scene in plan["scenes"]]
        issues = [issue for plan in plan_states for issue in plan["issues"]]
        plans = [{key: plan[key] for key in PLAN_PAYLOAD_KEYS} for plan in plan_states]
        result = {
            "batch_filter": batch,
            # Dashboard consumers must always have an unfiltered view.  The
            # legacy ``batches`` field remains scoped to the robot selector so
            # QC batch controls do not change their meaning.
            "all_batches": summaries,
            "batches": [
                summary
                for summary in summaries
                if not robot_type or summary["robot_type"] == robot_type
            ],
            "robot_types": robot_types,
            "plans": plans,
            "tasks": tasks,
            "scenes": scenes,
            "objects": self.assets.records(),
            "issues": issues,
            "plans_root": str(self.plans_root),
            "assets_root": str(self.assets.root),
            "collection_root": str(self.collection_root),
        }
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO state_cache "
                "(cache_key, updated, payload, cache_version, source_token) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cache_key,
                    time.time(),
                    json.dumps(result, ensure_ascii=False),
                    STATE_CACHE_VERSION,
                    source_token,
                ),
            )
            self._db.commit()
        return result

    def batch_ids(self) -> list[str]:
        return sorted(
            path.name
            for path in self.plans_root.iterdir()
            if path.is_dir() and (path / "tasks.csv").is_file()
        )

    @staticmethod
    def _benchmark_action(batch_id: str) -> str:
        """Normalize a batch name to its action suffix for YAM alignment."""
        value = str(batch_id).strip()
        value = re.sub(r"^larybench2_[0-9]{8}_", "", value, flags=re.IGNORECASE)
        for prefix in ("dual_yam_", "arx_x5_", "agilex_piper_", "agilex_"):
            if value.lower().startswith(prefix):
                return value[len(prefix) :]
        return value

    @classmethod
    def _attach_yam_benchmarks(cls, summaries: list[dict[str, Any]]) -> None:
        """Annotate dashboard quantities with the matching YAM task baseline.

        Task plans remain the source of truth for QC and editing.  These derived
        fields only keep the cross-robot dashboard comparable when one robot has
        an extra or stale variant directory.
        """
        yam = {
            cls._benchmark_action(summary["batch_id"]): summary
            for summary in summaries
            if str(summary.get("robot_type", "")) == "dual_yam"
        }
        for summary in summaries:
            baseline = yam.get(cls._benchmark_action(summary["batch_id"]))
            if baseline is None and not yam:
                # Keep standalone/custom collections useful when no YAM plan
                # is present to establish the cross-robot benchmark.
                baseline = summary
            if baseline is None:
                summary.update(
                    {
                        "benchmark_batch": "",
                        "benchmark_task_count": 0,
                        "benchmark_target_episodes": 0,
                    }
                )
                continue
            summary.update(
                {
                    "benchmark_batch": baseline["batch_id"],
                    "benchmark_task_count": baseline["tasks"],
                    "benchmark_target_episodes": baseline["target_episodes"],
                }
            )

    def upsert_plan(
        self, batch: str, kind: str, current_id: str | None, payload: Any
    ) -> dict[str, Any]:
        store = self._store(batch)
        store.upsert(kind, current_id, payload)
        self._invalidate_plan_cache(batch)
        return self.state(batch)

    def delete_plan(self, batch: str, kind: str, record_id: str) -> dict[str, Any]:
        self._store(batch).delete(kind, record_id)
        self._invalidate_plan_cache(batch)
        return self.state(batch)

    def update_info(self, batch: str, payload: Any) -> dict[str, Any]:
        self._store(batch).update_info(payload)
        self._invalidate_plan_cache(batch)
        return self.state(batch)

    def import_plan(self, batch: str, content: bytes) -> dict[str, Any]:
        batch_id = self._batch_id(batch)
        TaskSetStore(self.plans_root / batch_id, self.assets.path).import_zip(content)
        self._invalidate_plan_cache(batch_id)
        return self.state(batch_id)

    def export_plan(self, batch: str) -> io.BytesIO:
        return self._store(batch).export_zip()

    def upsert_asset(self, current_id: str | None, payload: Any) -> None:
        self.assets.upsert(current_id, payload)
        self._invalidate_plan_cache()

    def delete_asset(self, object_id: str) -> None:
        references = []
        for batch in self.batch_ids():
            store = self._store(batch)
            state = store.state()
            for scene in state["scenes"]:
                if any(item["object_id"] == object_id for item in scene["placements"]):
                    references.append(f"{batch}/{scene['scene_id']}")
            for task in state["tasks"]:
                if object_id in task["operation_object_ids"]:
                    references.append(f"{batch}/{task['task_id']}")
        if references:
            raise ConflictError(f"{object_id} is referenced by {', '.join(references[:6])}")
        self.assets.delete(object_id)
        self._invalidate_plan_cache()

    def upload_photos(self, object_id: str, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
        result = self.assets.save_photos(object_id, uploads)
        self._invalidate_plan_cache()
        return result

    def review(self, batch: str, slot_id: str) -> dict[str, Any]:
        plan = self._plan_state(batch)
        slots = (slot for task in plan["tasks"] for slot in task["slots"])
        slot = next((item for item in slots if item["slot_id"] == slot_id), None)
        if slot is None:
            raise RecordNotFoundError(slot_id)
        payload = {
            "batch_id": batch,
            "robot_type": plan["info"].get("robot_type", ""),
            "dataset_dir": plan["collection"]["dataset_dir"],
            "slot": slot,
            "series": None,
            "videos": [],
        }
        episode = slot.get("episode")
        if episode is None:
            return payload
        dataset_dir = Path(plan["collection"]["dataset_dir"])
        series = self._series(dataset_dir, int(episode["episode_index"]))
        video_keys = series.pop("video_keys")
        payload["series"] = series
        payload["videos"] = [{"key": key, "label": key.rsplit(".", 1)[-1]} for key in video_keys]
        payload["fps"] = self._json_file(dataset_dir / "meta" / "info.json").get("fps", 30)
        return payload

    def qc_rows(self, batch: str) -> list[dict[str, Any]]:
        """Return pending or failed slots as a compact EVA-CLIENT handoff list."""
        plan = self._plan_state(batch)
        robot_type = str(plan["info"].get("robot_type", ""))
        rows: list[dict[str, Any]] = []
        for task in plan["tasks"]:
            for slot in task["slots"]:
                if slot.get("qc_state", slot["state"]) not in {"pending", "failed"}:
                    continue
                rows.append(
                    {
                        "batch_id": batch,
                        "robot_type": robot_type,
                        "task_id": task["task_id"],
                        "scene_id": slot["scene_id"],
                        "slot_id": slot["slot_id"],
                        "round_index": slot["round_index"],
                        "round_total": slot["round_total"],
                        "status": slot.get("qc_state", slot["state"]),
                        "episode_index": ""
                        if slot.get("episode") is None
                        else slot["episode"].get("episode_index", ""),
                        "prompt_en": task.get("prompt_en", ""),
                        "prompt_zh": task.get("prompt_zh", ""),
                        "qc_note": ""
                        if slot.get("episode") is None
                        else slot["episode"].get("qc_note", ""),
                        "qc_reason": ""
                        if slot.get("episode") is None
                        else slot["episode"].get("qc_reason", ""),
                    }
                )
        return rows

    def export_qc_plan(self, batch: str) -> io.BytesIO:
        """Build a standard task set containing only pending or failed slots."""
        plan = self._plan_state(batch)
        store = self._store(batch)
        selected_tasks: list[dict[str, Any]] = []
        selected_scene_ids: set[str] = set()
        for task in plan["tasks"]:
            slots_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for slot in task["slots"]:
                if slot.get("qc_state", slot["state"]) in {"pending", "failed"}:
                    slots_by_scene[str(slot["scene_id"])].append(slot)
            if not slots_by_scene:
                continue
            scene_ids = [
                scene_id for scene_id in task["scene_ids"] if str(scene_id) in slots_by_scene
            ]
            filtered = {
                key: task[key]
                for key in (
                    "task_id",
                    "action",
                    "category",
                    "operation_object_ids",
                    "prompt_en",
                    "prompt_zh",
                )
            }
            filtered["scene_ids"] = scene_ids
            filtered["scene_epsiodes_count"] = [
                len(slots_by_scene[str(scene_id)]) for scene_id in scene_ids
            ]
            filtered["total_epsiodes_count"] = sum(filtered["scene_epsiodes_count"])
            selected_tasks.append(filtered)
            selected_scene_ids.update(scene_ids)

        selected_scenes = [
            {key: scene[key] for key in ("scene_id", "placements")}
            for scene in plan["scenes"]
            if str(scene["scene_id"]) in selected_scene_ids
        ]
        info = store._current_info({"info": plan["info"], "tasks": selected_tasks})
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("info.yaml", store._yaml_text(info))
            archive.writestr("layout.yaml", store._yaml_text(plan["layout"]))
            archive.writestr("scene.csv", store._csv_text("scenes", selected_scenes))
            archive.writestr("tasks.csv", store._csv_text("tasks", selected_tasks))
        stream.seek(0)
        return stream

    def video_path(self, batch: str, episode_index: int, video_key: str) -> Path:
        plan = self._plan_state(batch)
        dataset_dir = Path(plan["collection"]["dataset_dir"])
        inferred = LeRobotDatasetIO(dataset_dir).infer_keys()
        if video_key not in inferred["image"]["candidates"]:
            raise ValueError("Unknown episode video key")
        paths = LeRobotDatasetIO(dataset_dir).episode_video_paths(
            episode_index, {video_key: video_key}
        )
        if video_key not in paths:
            raise FileNotFoundError(video_key)
        return paths[video_key]

    def mark_qc(
        self,
        batch: str,
        episode_index: int,
        verdict: str,
        note: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        if verdict not in {"", "pass", "fail"}:
            raise ValueError("QC verdict must be pass, fail, or empty")
        if reason not in {"", *QC_REASONS}:
            raise ValueError("Unknown QC reason")
        if reason == "other" and not note.strip():
            raise ValueError("QC reason 'other' requires a note")
        plan = self._plan_state(batch)
        dataset_dir = Path(plan["collection"]["dataset_dir"])
        with self._lock:
            updated = LeRobotDatasetIO(dataset_dir).mark_qc(episode_index, verdict, note, reason)
        if not updated:
            raise RecordNotFoundError(f"Episode {episode_index}")
        self._invalidate_plan_cache(batch)
        return self.state(batch)

    def robot_meta(self, robot_type: str) -> dict[str, Any]:
        with self._lock:
            cached = self._robot_meta_cache.get(robot_type)
            if cached is None:
                robot, scene = self._robot_scene(robot_type)
                cached = {
                    "available": True,
                    "robot_type": robot_type,
                    "arms": scene.arm_names,
                    "meshes": scene.static_meshes(),
                    "initial": scene.transforms(robot.initial_qpos),
                }
                self._robot_meta_cache[robot_type] = cached
            return cached

    def mesh_bytes(self, robot_type: str, name: str) -> bytes:
        key = (robot_type, name)
        with self._lock:
            content = self._mesh_cache.get(key)
            if content is None:
                _, scene = self._robot_scene(robot_type)
                content = scene.mesh_bytes(name)
                if content is None:
                    raise FileNotFoundError(name)
                self._mesh_cache[key] = content
            return content

    def transform_blob(
        self, batch: str, episode_index: int, start: int, count: int
    ) -> tuple[bytes, int, int]:
        plan = self._plan_state(batch)
        dataset_dir = Path(plan["collection"]["dataset_dir"])
        series = self._series(dataset_dir, episode_index)
        qpos = np.asarray(series["state"], dtype=np.float32)
        total = len(qpos)
        start = min(max(0, start), total)
        end = min(total, start + min(max(1, count), 120))
        _, scene = self._robot_scene(str(plan["info"].get("robot_type", "")))
        return scene.all_transforms_blob(qpos[start:end]), start, total

    def _plan_state(self, batch: str) -> dict[str, Any]:
        batch_id = self._batch_id(batch)
        now = time.monotonic()
        source_token = self._source_token([batch_id])
        with self._state_lock:
            cached = self._plan_cache.get(batch_id)
            if (
                cached is not None
                and now - cached[0] < PLAN_CACHE_SECONDS
                and cached[1] == source_token
            ):
                return cached[2]
            state = self._store(batch_id).state()
            collection = self._decorate(batch_id, state)
            state["batch_id"] = batch_id
            state["dataset_dir"] = str(self._batch_root(batch_id))
            state["tasks"] = collection.pop("tasks")
            state["scenes"] = [{**scene, "batch_id": batch_id} for scene in state["scenes"]]
            state["collection"] = collection
            state["issues"] = [{**issue, "batch_id": batch_id} for issue in state["issues"]]
            if collection["robot_type"] and collection["robot_type"] != state["info"]["robot_type"]:
                state["issues"].append(
                    {
                        "level": "error",
                        "entity": batch_id,
                        "message": "Collected robot_type does not match plan metadata",
                        "batch_id": batch_id,
                    }
                )
            self._plan_cache[batch_id] = (now, source_token, state)
            return state

    def _invalidate_plan_cache(self, batch: str = "") -> None:
        with self._state_lock:
            if batch:
                batch_id = self._batch_id(batch)
                self._plan_cache.pop(batch_id, None)
                self._summary_cache.pop(batch_id, None)
            else:
                self._plan_cache.clear()
                self._summary_cache.clear()
        # Dashboard responses aggregate every batch, so any mutation can
        # invalidate a cached filtered response as well as the current batch.
        with self._db_lock:
            self._db.execute("DELETE FROM state_cache")
            self._db.commit()

    def _source_token(self, batches: list[str]) -> str:
        """Return a cheap mtime/size token for cache invalidation."""
        parts: list[str] = [self._file_token(self.assets.path)]
        for batch in batches:
            root = self.plans_root / batch
            info_path = root / "info.yaml"
            for path in (info_path, root / "layout.yaml", root / "scene.csv", root / "tasks.csv"):
                parts.append(self._file_token(path))
            info: dict[str, Any] = {}
            if info_path.is_file():
                payload = yaml.safe_load(info_path.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict):
                    info = payload
            dataset_dir = self._source_dataset_dir(batch, info)
            for path in (
                dataset_dir / "meta" / "info.json",
                dataset_dir / "meta" / "episodes.jsonl",
                dataset_dir / "meta" / "tasks.jsonl",
            ):
                parts.append(self._file_token(path))
        return "|".join(parts)

    @staticmethod
    def _file_token(path: Path) -> str:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return f"{path}:missing"
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"

    def _source_dataset_dir(self, batch: str, info: dict[str, Any]) -> Path:
        configured = str(info.get("collection_dir", "")).strip()
        if configured:
            path = Path(configured).expanduser()
            resolved = (path if path.is_absolute() else self.collection_root / path).resolve()
            if resolved != self.collection_root and self.collection_root not in resolved.parents:
                raise ValueError("collection_dir must stay inside the configured collection root")
            return resolved
        dataset_name = str(info.get("dataset_name") or batch)
        robot_type = str(info.get("robot_type", ""))
        candidates = [
            self.collection_root / dataset_name / "raw",
            self.collection_root / batch / "raw",
            self.collection_root / robot_type / dataset_name / "raw",
            self.collection_root / robot_type / batch / "raw",
            self.collection_root / "datasets" / dataset_name,
            self.collection_root / "datasets" / dataset_name / "raw",
            self.collection_root / "datasets" / robot_type / dataset_name,
            self.collection_root / "datasets" / robot_type / dataset_name / "raw",
        ]
        datasets_root = self.collection_root / "datasets"
        if datasets_root.is_dir():
            candidates.extend(
                group / dataset_name for group in datasets_root.iterdir() if group.is_dir()
            )
        return next(
            (path.resolve() for path in candidates if (path / "meta" / "episodes.jsonl").is_file()),
            candidates[0].resolve(),
        )

    def _decorate(self, batch: str, state: dict[str, Any]) -> dict[str, Any]:
        dataset_dir = self._dataset_dir(batch, state["info"])
        episodes = self._episode_rows(dataset_dir)
        by_slot: dict[str, list[dict[str, Any]]] = {}
        by_target: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        by_task: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            slot_id = str(episode.get("slot_id", ""))
            if slot_id:
                by_slot.setdefault(slot_id, []).append(episode)
            target = (
                str(episode.get("task_id", "")),
                str(episode.get("scene_id", "")),
                int(episode.get("scene_round", -1)),
            )
            by_target.setdefault(target, []).append(episode)
            task_id = str(episode.get("task_id", ""))
            if task_id:
                by_task.setdefault(task_id, []).append(episode)
        # Expand task counts into exact collection slots
        tasks = []
        matched: set[int] = set()
        counts = {"complete": 0, "pending": 0, "repair": 0, "total": 0}
        qc_counts = {"pending": 0, "unreviewed": 0, "passed": 0, "failed": 0}
        duplicates = []
        for task in state["tasks"]:
            decorated = {**task, "batch_id": batch, "slots": []}
            for scene_index, scene_id in enumerate(task["scene_ids"]):
                total = task["scene_epsiodes_count"][scene_index]
                for round_index in range(total):
                    slot_id = f"{task['task_id']}:{scene_id}:{round_index}"
                    candidates = by_slot.get(slot_id) or by_target.get(
                        (task["task_id"], scene_id, round_index), []
                    )
                    if not candidates:
                        # Some legacy robot exports kept the same task IDs but
                        # used a different scene naming convention.
                        candidates = next(
                            (
                                [episode]
                                for episode in sorted(
                                    by_task.get(task["task_id"], []),
                                    key=lambda row: int(row.get("episode_index", -1)),
                                )
                                if id(episode) not in matched
                            ),
                            [],
                        )
                    candidates = sorted(
                        candidates, key=lambda row: int(row.get("episode_index", -1))
                    )
                    if len(candidates) > 1:
                        duplicates.append(slot_id)
                    matched.update(id(row) for row in candidates)
                    episode = candidates[-1] if candidates else None
                    slot = self._slot(slot_id, scene_id, round_index, total, episode)
                    decorated["slots"].append(slot)
                    qc_state = slot.get("qc_state", slot["state"])
                    legacy_state = (
                        "repair"
                        if qc_state == "failed"
                        else "pending"
                        if qc_state == "pending"
                        else "complete"
                    )
                    counts[legacy_state] += 1
                    qc_counts[qc_state] += 1
                    counts["total"] += 1
            decorated["counts"] = {
                "complete": sum(
                    slot.get("qc_state", slot["state"]) in {"unreviewed", "passed"}
                    for slot in decorated["slots"]
                ),
                "pending": sum(
                    slot.get("qc_state", slot["state"]) == "pending" for slot in decorated["slots"]
                ),
                "repair": sum(
                    slot.get("qc_state", slot["state"]) == "failed" for slot in decorated["slots"]
                ),
            }
            tasks.append(decorated)
        info = self._json_file(dataset_dir / "meta" / "info.json")
        return {
            "dataset_dir": str(dataset_dir),
            "available": (dataset_dir / "meta" / "episodes.jsonl").is_file(),
            "robot_type": str(info.get("robot_type", "")),
            "counts": counts,
            "qc_counts": qc_counts,
            "duplicates": duplicates,
            "orphaned": [self._episode_summary(row) for row in episodes if id(row) not in matched],
            "tasks": tasks,
        }

    def _dataset_dir(self, batch: str, info: dict[str, Any]) -> Path:
        configured = str(info.get("collection_dir", "")).strip()
        if configured:
            path = Path(configured).expanduser()
            resolved = (path if path.is_absolute() else self.collection_root / path).resolve()
            if resolved != self.collection_root and self.collection_root not in resolved.parents:
                raise ValueError("collection_dir must stay inside the configured collection root")
            return resolved
        dataset_name = str(info.get("dataset_name") or batch)
        robot_type = str(info.get("robot_type", ""))
        candidates = [
            self.collection_root / dataset_name / "raw",
            self.collection_root / batch / "raw",
            self.collection_root / robot_type / dataset_name / "raw",
            self.collection_root / robot_type / batch / "raw",
            # Canonical datasets synced below data_collection/datasets may be
            # grouped by source/robot and commonly have no ``raw`` level.
            self.collection_root / "datasets" / dataset_name,
            self.collection_root / "datasets" / dataset_name / "raw",
            self.collection_root / "datasets" / robot_type / dataset_name,
            self.collection_root / "datasets" / robot_type / dataset_name / "raw",
        ]
        if self.collection_root.is_dir():
            candidates.extend(
                path / dataset_name / "raw" for path in self.collection_root.iterdir()
            )
            datasets_root = self.collection_root / "datasets"
            if datasets_root.is_dir():
                # Support one additional grouping level (for example
                # datasets/real_robot/dual_yam/<dataset>).
                candidates.extend(
                    path
                    for group in datasets_root.iterdir()
                    if group.is_dir()
                    for path in group.rglob(dataset_name)
                    if path.is_dir()
                )
        return next(
            (path.resolve() for path in candidates if (path / "meta" / "episodes.jsonl").is_file()),
            candidates[0].resolve(),
        )

    def _series(self, dataset_dir: Path, episode_index: int) -> dict[str, Any]:
        io_store = LeRobotDatasetIO(dataset_dir)
        inferred = io_store.infer_keys()
        state_key = inferred["state"]["default"]
        action_key = inferred["action"]["default"]
        if not state_key or not action_key:
            raise ValueError("Episode does not declare state and action features")
        path = io_store.episode_parquet(episode_index)
        schema = pq.read_schema(path)
        columns = [state_key, action_key]
        if "timestamp" in schema.names:
            columns.append("timestamp")
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            return {
                "timestamp": [],
                "state": [],
                "action": [],
                "state_names": [],
                "action_names": [],
                "video_keys": inferred["image"]["candidates"],
            }
        info = self._json_file(dataset_dir / "meta" / "info.json")
        fps = float(info.get("fps", 30) or 30)
        timestamp = (
            [float(value) for value in table.column("timestamp").to_pylist()]
            if "timestamp" in columns
            else [index / fps for index in range(table.num_rows)]
        )
        return {
            "timestamp": timestamp,
            "state": table.column(state_key).to_pylist(),
            "action": table.column(action_key).to_pylist(),
            "state_names": self._feature_names(info, state_key, table.column(state_key)[0]),
            "action_names": self._feature_names(info, action_key, table.column(action_key)[0]),
            "video_keys": inferred["image"]["candidates"],
        }

    def _robot_scene(self, robot_type: str) -> tuple[Any, UrdfScene]:
        if not robot_type:
            raise ValueError("Plan metadata must define robot_type")
        with self._lock:
            cached = self._robot_scenes.get(robot_type)
            if cached is None:
                robot = ROBOT_REGISTRY.build(robot_type)
                cached = (robot, UrdfScene(robot))
                self._robot_scenes[robot_type] = cached
            return cached

    def _batch_summary(self, batch: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
        now = time.monotonic()
        source_token = self._source_token([batch])
        if state is None:
            cached = self._summary_cache.get(batch)
            if (
                cached is not None
                and now - cached[0] < PLAN_CACHE_SECONDS
                and cached[1] == source_token
            ):
                return cached[2]
        if state is None:
            state = self._store(batch).state()
        robot_type = str(state["info"].get("robot_type", ""))
        cameras = []
        # Plans may describe robots that this client does not implement. Camera
        # metadata is optional; such plans must still be browsable and editable.
        if robot_type and robot_type in ROBOT_REGISTRY:
            robot = ROBOT_REGISTRY.build(robot_type)
            cameras = [
                {
                    "name": camera.name,
                    "observation_key": camera.observation_key,
                    "attached_to": camera.attached_to,
                }
                for camera in robot.observation_schema.cameras
            ]
        target = sum(task["total_epsiodes_count"] for task in state["tasks"])
        collection_dir = self._dataset_dir(batch, state["info"])
        episode_rows = self._episode_rows(collection_dir)
        task_ids = {str(task["task_id"]) for task in state["tasks"]}
        slot_ids = {
            f"{task['task_id']}:{scene_id}:{round_index}"
            for task in state["tasks"]
            for scene_id, count in zip(
                task["scene_ids"], task["scene_epsiodes_count"], strict=False
            )
            for round_index in range(count)
        }
        collected_rows = sorted(
            (
                row
                for row in episode_rows
                if str(row.get("task_id", "")) in task_ids
                or str(row.get("slot_id", "")) in slot_ids
            ),
            key=lambda row: int(row.get("episode_index", -1)),
        )[:target]
        dataset_info = self._json_file(collection_dir / "meta" / "info.json")
        fps = float(dataset_info.get("fps", 30) or 30)
        frames = sum(int(row.get("length", 0) or 0) for row in collected_rows)
        duration_seconds = sum(
            float(row.get("duration_seconds", 0) or 0) or int(row.get("length", 0) or 0) / fps
            for row in collected_rows
        )
        daily: dict[str, int] = defaultdict(int)
        for row in collected_rows:
            started_at = str(row.get("started_at", ""))
            day = started_at[:10] if len(started_at) >= 10 else ""
            if day:
                daily[day] += 1
        status_counts = {"unreviewed": 0, "passed": 0, "failed": 0}
        for row in collected_rows:
            verdict = str(row.get("qc_verdict", "")).lower()
            quality = str(row.get("quality", "green")).lower()
            if quality == "red" or verdict == "fail":
                status_counts["failed"] += 1
            elif verdict == "pass":
                status_counts["passed"] += 1
            else:
                status_counts["unreviewed"] += 1
        collected = sum(status_counts.values())
        summary = {
            "batch_id": batch,
            "dataset_name": state["info"].get("dataset_name", batch),
            "robot_type": robot_type,
            "cameras": cameras,
            "tasks": len(state["tasks"]),
            "episodes": target,
            "target_episodes": target,
            "collected": collected,
            "pending": max(0, target - collected),
            "available": (collection_dir / "meta" / "episodes.jsonl").is_file(),
            "frames": frames,
            "duration_seconds": round(duration_seconds, 3),
            "daily": [{"date": day, "episodes": count} for day, count in sorted(daily.items())],
            **status_counts,
        }
        self._summary_cache[batch] = (now, source_token, summary)
        return summary

    def _batch_root(self, batch: str) -> Path:
        path = self.plans_root / self._batch_id(batch)
        if not path.is_dir():
            raise RecordNotFoundError(batch)
        return path

    def _store(self, batch: str) -> TaskSetStore:
        return TaskSetStore(self._batch_root(batch), self.assets.path)

    @staticmethod
    def _batch_id(batch: str) -> str:
        value = str(batch).strip()
        if not value or not ID_PATTERN.fullmatch(value):
            raise ValueError("batch must use letters, numbers, '.', '_' or '-'")
        return value

    @staticmethod
    def _slot(
        slot_id: str,
        scene_id: str,
        round_index: int,
        round_total: int,
        episode: dict[str, Any] | None,
    ) -> dict[str, Any]:
        summary = PlanCatalog._episode_summary(episode) if episode else None
        state = "pending"
        if summary is not None:
            verdict = summary["qc_verdict"].lower()
            rejected = summary["quality"].lower() == "red" or verdict == "fail"
            state = "failed" if rejected else "passed" if verdict == "pass" else "unreviewed"
        legacy_state = (
            "repair" if state == "failed" else "pending" if state == "pending" else "complete"
        )
        return {
            "slot_id": slot_id,
            "scene_id": scene_id,
            "round_index": round_index,
            "round_total": round_total,
            # Keep the legacy state field for existing API consumers while
            # exposing the four-state QC model to the reviewer UI.
            "state": legacy_state,
            "qc_state": state,
            "episode": summary,
        }

    @staticmethod
    def _episode_summary(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode_index": int(row.get("episode_index", -1)),
            "length": int(row.get("length", 0)),
            "quality": str(row.get("quality", "green")),
            "qc_verdict": str(row.get("qc_verdict", "")),
            "qc_note": str(row.get("qc_note", "")),
            "qc_reason": str(row.get("qc_reason", "")),
            "slot_id": str(row.get("slot_id", "")),
        }

    @staticmethod
    def _episode_rows(dataset_dir: Path) -> list[dict[str, Any]]:
        path = dataset_dir / "meta" / "episodes.jsonl"
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in filter(str.strip, handle):
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    @staticmethod
    def _feature_names(info: dict[str, Any], key: str, sample: Any) -> list[str]:
        names = (info.get("features", {}).get(key, {}) or {}).get("names")
        if isinstance(names, list):
            return [str(name) for name in names]
        values = sample.as_py() if hasattr(sample, "as_py") else []
        return [f"{key.rsplit('.', 1)[-1]} {index + 1}" for index in range(len(values))]

    @staticmethod
    def _json_file(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
