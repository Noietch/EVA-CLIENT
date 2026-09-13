"""Join task-plan batches to collected LeRobot episodes for QC review."""

from __future__ import annotations

import io
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from core.utils.lerobot import LeRobotDatasetIO
from robots.utils import UrdfScene
from tools.datasets.assets import ObjectCatalog
from tools.datasets.store import ConflictError, RecordNotFoundError, TaskSetStore

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PLAN_CACHE_SECONDS = 2.0
PLAN_PAYLOAD_KEYS = ("batch_id", "info", "layout", "dataset_dir", "collection")


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
        self.assets = ObjectCatalog(assets_root, [])
        self._robot_scenes: dict[str, tuple[Any, UrdfScene]] = {}
        self._robot_meta_cache: dict[str, dict[str, Any]] = {}
        self._mesh_cache: dict[tuple[str, str], bytes] = {}
        self._lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._plan_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def state(self, batch: str = "", robot_type: str = "") -> dict[str, Any]:
        batches = self.batch_ids()
        summaries = [self._batch_summary(value) for value in batches]
        robot_types = sorted(
            {
                str(summary["robot_type"]).strip()
                for summary in summaries
                if str(summary["robot_type"]).strip()
            }
        )
        robot_type = str(robot_type).strip()
        if batch:
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
            selected = [
                summary["batch_id"]
                for summary in summaries
                if not robot_type or summary["robot_type"] == robot_type
            ]
        plan_states = [self._plan_state(value) for value in selected]
        tasks = [task for plan in plan_states for task in plan["tasks"]]
        scenes = [scene for plan in plan_states for scene in plan["scenes"]]
        issues = [issue for plan in plan_states for issue in plan["issues"]]
        plans_by_batch = {plan["batch_id"]: plan for plan in plan_states}
        plans = [{key: plan[key] for key in PLAN_PAYLOAD_KEYS} for plan in plan_states]
        return {
            "batch_filter": batch,
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

    def batch_ids(self) -> list[str]:
        return sorted(
            path.name
            for path in self.plans_root.iterdir()
            if path.is_dir() and (path / "tasks.csv").is_file()
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

    def mark_qc(self, batch: str, episode_index: int, verdict: str, note: str) -> dict[str, Any]:
        if verdict not in {"", "pass", "fail"}:
            raise ValueError("QC verdict must be pass, fail, or empty")
        plan = self._plan_state(batch)
        dataset_dir = Path(plan["collection"]["dataset_dir"])
        with self._lock:
            updated = LeRobotDatasetIO(dataset_dir).mark_qc(episode_index, verdict, note)
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
        with self._state_lock:
            cached = self._plan_cache.get(batch_id)
            if cached is not None and now - cached[0] < PLAN_CACHE_SECONDS:
                return cached[1]
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
            self._plan_cache[batch_id] = (now, state)
            return state

    def _invalidate_plan_cache(self, batch: str = "") -> None:
        with self._state_lock:
            if batch:
                self._plan_cache.pop(self._batch_id(batch), None)
            else:
                self._plan_cache.clear()

    def _decorate(self, batch: str, state: dict[str, Any]) -> dict[str, Any]:
        dataset_dir = self._dataset_dir(batch, state["info"])
        episodes = self._episode_rows(dataset_dir)
        by_slot: dict[str, list[dict[str, Any]]] = {}
        by_target: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
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
        # Expand task counts into exact collection slots
        tasks = []
        matched: set[int] = set()
        counts = {"complete": 0, "repair": 0, "pending": 0, "total": 0}
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
                    candidates = sorted(
                        candidates, key=lambda row: int(row.get("episode_index", -1))
                    )
                    if len(candidates) > 1:
                        duplicates.append(slot_id)
                    matched.update(id(row) for row in candidates)
                    episode = candidates[-1] if candidates else None
                    slot = self._slot(slot_id, scene_id, round_index, total, episode)
                    decorated["slots"].append(slot)
                    counts[slot["state"]] += 1
                    counts["total"] += 1
            decorated["counts"] = {
                key: sum(slot["state"] == key for slot in decorated["slots"])
                for key in ("complete", "repair", "pending")
            }
            tasks.append(decorated)
        info = self._json_file(dataset_dir / "meta" / "info.json")
        return {
            "dataset_dir": str(dataset_dir),
            "available": (dataset_dir / "meta" / "episodes.jsonl").is_file(),
            "robot_type": str(info.get("robot_type", "")),
            "counts": counts,
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
        if state is None:
            state = self._store(batch).state()
        robot_type = str(state["info"].get("robot_type", ""))
        cameras = []
        if robot_type:
            robot = ROBOT_REGISTRY.build(robot_type)
            cameras = [
                {
                    "name": camera.name,
                    "observation_key": camera.observation_key,
                    "attached_to": camera.attached_to,
                }
                for camera in robot.observation_schema.cameras
            ]
        return {
            "batch_id": batch,
            "dataset_name": state["info"].get("dataset_name", batch),
            "robot_type": robot_type,
            "cameras": cameras,
            "tasks": len(state["tasks"]),
            "episodes": sum(task["total_epsiodes_count"] for task in state["tasks"]),
        }

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
            rejected = (
                summary["quality"].lower() == "red" or summary["qc_verdict"].lower() == "fail"
            )
            state = "repair" if rejected else "complete"
        return {
            "slot_id": slot_id,
            "scene_id": scene_id,
            "round_index": round_index,
            "round_total": round_total,
            "state": state,
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
