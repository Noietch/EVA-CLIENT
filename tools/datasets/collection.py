"""Join task-plan batches to collected LeRobot episodes for QC review."""

from __future__ import annotations

import io
import json
import logging
import re
import sqlite3
import threading
import time
import zipfile
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from core.utils.lerobot import LeRobotDatasetIO
from core.utils.qc import qc_state
from core.utils.scene_plan import plan_slots, scene_plan_from_dir
from robots.utils import UrdfScene
from tools.datasets.assets import ObjectCatalog
from tools.datasets.store import ConflictError, RecordNotFoundError, TaskSetStore

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
logger = logging.getLogger(__name__)
PLAN_CACHE_SECONDS = 60.0
SERIES_CACHE_MAX = 8
TRANSFORM_CACHE_MAX = 32
UNMATCHED_CACHE_SECONDS = 300.0
PLAN_PAYLOAD_KEYS = ("batch_id", "info", "layout", "dataset_dir", "collection")
# Bump when the state payload shape changes so cached pages are ignored.
STATE_CACHE_VERSION = 8
# Bump when the derived-row shape changes so existing database caches are ignored.
DATA_CACHE_VERSION = 3
# Derived rows stay useful while their source token holds; stale tokens are dead weight.
DATA_CACHE_TTL_SECONDS = 3 * 24 * 3600.0
CACHE_EVICTION_SECONDS = 3600.0
STATIC_FRAMES_REASON = "static_frames_excessive"
CAMERA_OFFLINE_REASON = "camera_offline"
SHORT_EPISODE_REASON = "trajectory_too_short"
# The machine never closes an episode: what it finds is handed to a reviewer.
MANUAL_REVIEW_REASON = "needs_manual_review"
QC_REASONS = {
    "image_quality",
    "trajectory_quality",
    "task_mismatch",
    STATIC_FRAMES_REASON,
    CAMERA_OFFLINE_REASON,
    SHORT_EPISODE_REASON,
    MANUAL_REVIEW_REASON,
    "other",
}
# A demonstration naturally holds still for a few frames while the operator
# repositions; only a longer pause means the recording sat idle.
STATIC_RUN_MIN_FRAMES = 5
STATIC_QC_ANALYSIS_LIMIT = 64
STATE_CACHE_SECONDS = 300.0


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
        self._raw_state_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._unmatched_cache: dict[str, tuple[float, str, list[dict[str, Any]]]] = {}
        self._dataset_info_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._dataset_stems_cache: dict[str, tuple[str, dict[int, str]]] = {}
        self._episode_rows_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        self._inferred_keys_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._source_token_cache: dict[tuple[str, ...], tuple[float, str]] = {}
        self._series_cache: OrderedDict[tuple[str, int, int, int], dict[str, Any]] = OrderedDict()
        self._transform_cache: OrderedDict[
            tuple[str, int, int, int, int], tuple[bytes, int, int]
        ] = OrderedDict()
        self._db_path = self.collection_root / ".dataset_cache.sqlite3"
        self._db_lock = threading.RLock()
        self._db_cache_disabled = False
        self._db_evicted_at = 0.0
        self._db = self._open_cache_db()

    def _open_cache_db(self) -> sqlite3.Connection:
        """Open the derived-data cache, rebuilding the image when it is unreadable."""
        db = self._connect_cache_db()
        try:
            db.execute("SELECT cache_key FROM data_cache LIMIT 1").fetchone()
            db.execute("SELECT cache_key FROM state_cache LIMIT 1").fetchone()
        except sqlite3.DatabaseError as error:
            logger.warning("dataset cache is unreadable (%s); rebuilding %s", error, self._db_path)
            db.close()
            self._discard_cache_db()
            db = self._connect_cache_db()
        return db

    def _connect_cache_db(self) -> sqlite3.Connection:
        """Connect with the pragmas two dataset services sharing one file need."""
        db = sqlite3.connect(self._db_path, check_same_thread=False, timeout=15.0)
        db.execute("PRAGMA busy_timeout=15000")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS state_cache ("
            "cache_key TEXT PRIMARY KEY, updated REAL NOT NULL, payload TEXT NOT NULL, "
            "cache_version INTEGER NOT NULL DEFAULT 1, source_token TEXT NOT NULL DEFAULT ''"
            ")"
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(state_cache)").fetchall()}
        if "cache_version" not in columns:
            db.execute(
                "ALTER TABLE state_cache ADD COLUMN cache_version INTEGER NOT NULL DEFAULT 1"
            )
        if "source_token" not in columns:
            db.execute("ALTER TABLE state_cache ADD COLUMN source_token TEXT NOT NULL DEFAULT ''")
        db.execute(
            "CREATE TABLE IF NOT EXISTS data_cache ("
            "cache_key TEXT PRIMARY KEY, updated REAL NOT NULL, payload TEXT NOT NULL, "
            "cache_version INTEGER NOT NULL DEFAULT 1, source_token TEXT NOT NULL DEFAULT ''"
            ")"
        )
        db.commit()
        return db

    def _discard_cache_db(self) -> None:
        """Drop a cache image; everything in it is recomputable."""
        for suffix in ("", "-wal", "-shm"):
            path = self._db_path.with_name(self._db_path.name + suffix)
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("cannot remove cache file %s: %s", path, error)

    def _cache_failed(self, error: sqlite3.Error) -> None:
        """Caching is optional: a broken cache costs a recompute, never an answer."""
        if isinstance(error, sqlite3.DatabaseError) and not isinstance(
            error, sqlite3.OperationalError
        ):
            logger.warning("dataset cache is corrupt (%s); rebuilding %s", error, self._db_path)
            with self._db_lock:
                try:
                    self._db.close()
                except sqlite3.Error:
                    pass
                self._discard_cache_db()
                try:
                    self._db = self._connect_cache_db()
                except sqlite3.Error as retry_error:
                    error = retry_error
                else:
                    return
        logger.warning("dataset cache disabled: %s", error)
        self._db_cache_disabled = True

    def _evict_stale_cache_rows(self) -> None:
        """Keep the cache file from growing without bound as tokens change."""
        now = time.time()
        if now - self._db_evicted_at < CACHE_EVICTION_SECONDS:
            return
        self._db_evicted_at = now
        self._db.execute(
            "DELETE FROM data_cache WHERE updated < ?", (now - DATA_CACHE_TTL_SECONDS,)
        )
        self._db.execute("DELETE FROM state_cache WHERE updated < ?", (now - STATE_CACHE_SECONDS,))

    def _db_cache_get(self, key: str, source_token: str, ttl: float = 300.0) -> Any | None:
        if self._db_cache_disabled:
            return None
        try:
            with self._db_lock:
                row = self._db.execute(
                    "SELECT updated, payload FROM data_cache "
                    "WHERE cache_key = ? AND cache_version = ? AND source_token = ?",
                    (key, DATA_CACHE_VERSION, source_token),
                ).fetchone()
            if row and time.time() - float(row[0]) < ttl:
                return json.loads(row[1])
        except sqlite3.Error as error:
            self._cache_failed(error)
        except json.JSONDecodeError:
            return None
        return None

    def _db_cache_put(self, key: str, source_token: str, payload: Any) -> None:
        if self._db_cache_disabled:
            return
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            with self._db_lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO data_cache "
                    "(cache_key, updated, payload, cache_version, source_token) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, time.time(), encoded, DATA_CACHE_VERSION, source_token),
                )
                self._evict_stale_cache_rows()
                self._db.commit()
        except sqlite3.Error as error:
            self._cache_failed(error)
        except (TypeError, ValueError):
            return

    def state(self, batch: str = "", robot_type: str = "") -> dict[str, Any]:
        batches = self.batch_ids()
        robot_type = str(robot_type).strip()
        source_token = self._source_token(batches)
        cache_key = f"v{STATE_CACHE_VERSION}:{robot_type}::{batch}:{source_token}"
        cached = None
        if not self._db_cache_disabled:
            try:
                with self._db_lock:
                    cached = self._db.execute(
                        "SELECT updated, payload FROM state_cache "
                        "WHERE cache_key = ? AND cache_version = ?",
                        (cache_key, STATE_CACHE_VERSION),
                    ).fetchone()
            except sqlite3.Error as error:
                self._cache_failed(error)
        if cached and time.time() - float(cached[0]) < STATE_CACHE_SECONDS:
            return json.loads(cached[1])

        summaries = [self._batch_summary(value) for value in batches]
        robot_names = sorted(
            {
                str(item.get("robot_type", ""))
                for item in summaries
                if item.get("robot_type") and item.get("available")
            }
        )
        summaries.extend(self._unmatched_summary(robot) for robot in robot_names)
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
        if batch and batch != "__all__" and plan_states and self._unmatched_robot(batch_id) is None:
            detailed_summary = self._batch_summary(batch_id, plan_states[0], analyze_static_qc=True)
            for index, summary in enumerate(summaries):
                if summary["batch_id"] == batch_id:
                    summaries[index] = detailed_summary
                    break
        tasks = [task for plan in plan_states for task in plan["tasks"]]
        slot_order = [
            {"batch_id": plan["batch_id"], "slot_id": slot_id}
            for plan in plan_states
            for slot_id in plan.get("slot_order") or []
        ]
        scenes = [scene for plan in plan_states for scene in plan["scenes"]]
        issues = [issue for plan in plan_states for issue in plan["issues"]]
        plans = [{key: plan[key] for key in PLAN_PAYLOAD_KEYS} for plan in plan_states]
        result = {
            "batch_filter": batch,
            # Every consumer filters this list by the selected robot itself.
            "all_batches": summaries,
            "robot_types": robot_types,
            "plans": plans,
            "tasks": tasks,
            "slot_order": slot_order,
            "scenes": scenes,
            "objects": self.assets.records(),
            "issues": issues,
            "plans_root": str(self.plans_root),
            "assets_root": str(self.assets.root),
            "collection_root": str(self.collection_root),
        }
        if not self._db_cache_disabled:
            now = time.time()
            try:
                with self._db_lock:
                    self._db.execute(
                        "INSERT OR REPLACE INTO state_cache "
                        "(cache_key, updated, payload, cache_version, source_token) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            cache_key,
                            now,
                            json.dumps(result, ensure_ascii=False),
                            STATE_CACHE_VERSION,
                            source_token,
                        ),
                    )
                    # A token change strands its predecessor; drop rows past
                    # the read TTL instead of one payload per review click.
                    self._db.execute(
                        "DELETE FROM state_cache WHERE updated < ?",
                        (now - STATE_CACHE_SECONDS,),
                    )
                    self._db.commit()
            except sqlite3.Error as error:
                self._cache_failed(error)
        return result

    def batch_ids(self) -> list[str]:
        return sorted(
            path.name
            for path in self.plans_root.iterdir()
            if path.is_dir() and (path / "tasks.csv").is_file()
        )

    @staticmethod
    def _unmatched_batch_id(robot_type: str) -> str:
        return f"{robot_type}_unmatched"

    @classmethod
    def _unmatched_robot(cls, batch: str) -> str | None:
        value = str(batch).strip()
        if not value.endswith("_unmatched"):
            return None
        robot = value[: -len("_unmatched")].strip()
        return robot or None

    def _unmatched_records(self, robot_type: str) -> list[dict[str, Any]]:
        source_token = self._source_token(self.batch_ids())
        now = time.monotonic()
        cached = self._unmatched_cache.get(robot_type)
        if (
            cached is not None
            and now - cached[0] < UNMATCHED_CACHE_SECONDS
            and cached[1] == source_token
        ):
            return cached[2]
        db_records = self._db_cache_get(
            f"unmatched:{robot_type}", source_token, UNMATCHED_CACHE_SECONDS
        )
        if isinstance(db_records, list):
            records = [dict(row) for row in db_records if isinstance(row, dict)]
            self._unmatched_cache[robot_type] = (now, source_token, records)
            return records
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for source_batch in self.batch_ids():
            state = self._raw_state(source_batch)
            if str(state["info"].get("robot_type", "")) != robot_type:
                continue
            dataset_dir = self._dataset_dir(source_batch, state["info"])
            expected_slots = {
                f"{task['task_id']}:{scene_id}:{round_index}"
                for task in state["tasks"]
                for scene_id, total in zip(
                    task["scene_ids"], task["scene_epsiodes_count"], strict=False
                )
                for round_index in range(total)
            }
            expected_targets = {
                (str(task["task_id"]), str(scene_id), round_index)
                for task in state["tasks"]
                for scene_id, total in zip(
                    task["scene_ids"], task["scene_epsiodes_count"], strict=False
                )
                for round_index in range(total)
            }
            for row in self._annotate_static_qc(
                dataset_dir, self.episode_rows(dataset_dir), analyze=False
            ):
                marker = str(row.get("taskset_match", ""))
                target = (
                    str(row.get("task_id", "")),
                    str(row.get("scene_id", "")),
                    int(row.get("scene_round", -1)),
                )
                if marker == "matched" or (
                    not marker
                    and (
                        str(row.get("slot_id", "")) in expected_slots or target in expected_targets
                    )
                ):
                    continue
                source_index = int(row.get("episode_index", -1))
                key = (str(dataset_dir), source_index)
                if key in seen:
                    continue
                seen.add(key)
                record = dict(row)
                record["_source_batch"] = source_batch
                record["_source_dataset_dir"] = str(dataset_dir)
                record["_source_episode_index"] = source_index
                records.append(record)
        records.sort(
            key=lambda row: (
                str(row["_source_dataset_dir"]),
                int(row["_source_episode_index"]),
            )
        )
        for virtual_index, record in enumerate(records):
            record["episode_index"] = virtual_index
        self._unmatched_cache[robot_type] = (now, source_token, records)
        self._db_cache_put(f"unmatched:{robot_type}", source_token, records)
        return records

    def _unmatched_summary(self, robot_type: str) -> dict[str, Any]:
        records = self._unmatched_records(robot_type)
        cameras = []
        if robot_type in ROBOT_REGISTRY:
            robot = ROBOT_REGISTRY.build(robot_type)
            cameras = [
                {
                    "name": camera.name,
                    "observation_key": camera.observation_key,
                    "attached_to": camera.attached_to,
                }
                for camera in robot.observation_schema.cameras
            ]
        frames = sum(int(row.get("length", 0) or 0) for row in records)
        status_counts = {"unreviewed": 0, "passed": 0, "failed": 0}
        for row in records:
            status_counts[qc_state(row.get("qc_verdict"), row.get("quality"))] += 1
        batch = self._unmatched_batch_id(robot_type)
        return {
            "batch_id": batch,
            "dataset_name": batch,
            "batch_kind": "unmatched",
            "robot_type": robot_type,
            "cameras": cameras,
            "tasks": len(records),
            "episodes": len(records),
            "target_episodes": len(records),
            "collected": len(records),
            "pending": 0,
            "available": bool(records),
            "frames": frames,
            "duration_seconds": round(
                sum(float(row.get("duration_seconds", 0) or 0) for row in records), 3
            ),
            "daily": [],
            **status_counts,
        }

    def _unmatched_state(self, batch: str) -> dict[str, Any]:
        robot_type = self._unmatched_robot(batch)
        if robot_type is None:
            raise RecordNotFoundError(batch)
        records = self._unmatched_records(robot_type)
        tasks = []
        for index, record in enumerate(records):
            task_id = f"UNMATCHED-{index:06d}"
            scene_id = task_id
            episode = dict(record)
            episode["slot_id"] = f"{task_id}:{scene_id}:0"
            slot = self._slot(episode["slot_id"], scene_id, 0, 1, episode)
            prompt = (record.get("tasks") or [record.get("source_prompt", "")])[0]
            tasks.append(
                {
                    "task_id": task_id,
                    "action": "unmatched",
                    "category": "unmatched",
                    "operation_object_ids": [],
                    "prompt_en": str(prompt),
                    "prompt_zh": str(prompt),
                    "scene_ids": [scene_id],
                    "scene_epsiodes_count": [1],
                    "total_epsiodes_count": 1,
                    "batch_id": batch,
                    "slots": [slot],
                    "counts": self._slot_counts([slot]),
                }
            )
        info = {
            "dataset_name": batch,
            "robot_type": robot_type,
            "task_count": len(tasks),
            "target_episodes": len(records),
            "read_only": True,
        }
        slot_counts = self._slot_counts([slot for task in tasks for slot in task["slots"]])
        collection = {
            "dataset_dir": str(self.collection_root),
            "available": bool(records),
            "robot_type": robot_type,
            "counts": {**slot_counts, "total": len(records)},
            "duplicates": [],
            "orphaned": [],
            "tasks": tasks,
        }
        return {
            "batch_id": batch,
            "info": info,
            "layout": {},
            "tasks": tasks,
            "scenes": [],
            "issues": [],
            "collection": collection,
            "dataset_dir": str(self.collection_root),
        }

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
        self.invalidate(batch)
        return self.state(batch)

    def delete_plan(self, batch: str, kind: str, record_id: str) -> dict[str, Any]:
        self._store(batch).delete(kind, record_id)
        self.invalidate(batch)
        return self.state(batch)

    def update_info(self, batch: str, payload: Any) -> dict[str, Any]:
        self._store(batch).update_info(payload)
        self.invalidate(batch)
        return self.state(batch)

    def import_plan(self, batch: str, content: bytes) -> dict[str, Any]:
        batch_id = self._batch_id(batch)
        TaskSetStore(self.plans_root / batch_id, self.assets.path).import_zip(content)
        self.invalidate(batch_id)
        return self.state(batch_id)

    def export_plan(self, batch: str) -> io.BytesIO:
        return self._store(batch).export_zip()

    def upsert_asset(self, current_id: str | None, payload: Any) -> None:
        self.assets.upsert(current_id, payload)
        self.invalidate()

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
        self.invalidate()

    def upload_photos(self, object_id: str, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
        result = self.assets.save_photos(object_id, uploads)
        self.invalidate()
        return result

    def import_assets_csv(self, content: bytes, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
        summary = self.assets.import_csv(content, uploads)
        self.invalidate()
        state = self.state()
        state["import_summary"] = summary
        return state

    def compare(
        self,
        task_id: str,
        scene_id: str = "",
        round_index: int = 0,
        current_batch: str = "",
    ) -> dict[str, Any]:
        """Return only the matching slot from each robot for one task comparison."""
        task_id = str(task_id).strip()
        scene_id = str(scene_id).strip()
        try:
            round_index = int(round_index)
        except (TypeError, ValueError):
            round_index = 0
        summaries = [self._batch_summary(batch) for batch in self.batch_ids()]
        summaries.sort(key=lambda item: 0 if item["batch_id"] == current_batch else 1)
        groups: list[dict[str, Any]] = []
        seen_robots: set[str] = set()
        for summary in summaries:
            batch = str(summary["batch_id"])
            robot = str(summary.get("robot_type") or batch)
            if robot in seen_robots:
                continue
            try:
                state = self._plan_state(batch)
            except (FileNotFoundError, ValueError):
                continue
            task = next(
                (item for item in state["tasks"] if str(item.get("task_id")) == task_id),
                None,
            )
            if task is None:
                continue
            slot = next(
                (
                    item
                    for item in task.get("slots", [])
                    if int(item.get("round_index", -1)) == round_index
                    and (not scene_id or str(item.get("scene_id")) == scene_id)
                ),
                None,
            )
            if slot is None:
                continue
            episode = slot.get("episode")
            episode_scene = str(episode.get("scene_id", "")) if episode else ""
            # A comparison is valid only when the episode carries the exact
            # canonical taskset scene requested by the caller.
            if episode_scene and scene_id and episode_scene != scene_id:
                continue
            try:
                payload = self.review(batch, str(slot["slot_id"]))
            except (FileNotFoundError, ValueError):
                payload = {
                    "batch_id": batch,
                    "robot_type": robot,
                    "slot": slot,
                    "series": None,
                    "videos": [],
                }
            payload["robot_type"] = robot
            groups.append(
                {
                    "batch_id": batch,
                    "robot_type": robot,
                    "task": {
                        "task_id": task.get("task_id", task_id),
                        "prompt_en": task.get("prompt_en", ""),
                        "prompt_zh": task.get("prompt_zh", ""),
                    },
                    "slot": slot,
                    "payload": payload,
                }
            )
            seen_robots.add(robot)
        return {
            "task_id": task_id,
            "scene_id": scene_id,
            "round_index": round_index,
            "groups": groups,
        }

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
            "video_batch": batch,
            "video_episode_index": None,
        }
        episode = slot.get("episode")
        if episode is None:
            return payload
        source_batch = str(episode.get("_source_batch", ""))
        source_index = int(episode.get("_source_episode_index", episode["episode_index"]))
        payload["video_batch"] = source_batch or batch
        payload["video_episode_index"] = source_index
        dataset_dir = (
            Path(self._plan_state(source_batch)["collection"]["dataset_dir"])
            if source_batch
            else Path(plan["collection"]["dataset_dir"])
        )
        series = dict(self._series(dataset_dir, source_index))
        video_keys = series.pop("video_keys")
        payload["series"] = series
        payload["videos"] = [{"key": key, "label": key.rsplit(".", 1)[-1]} for key in video_keys]
        payload["fps"] = self._dataset_info(dataset_dir).get("fps", 30)
        return payload

    def qc_rows(self, batch: str) -> list[dict[str, Any]]:
        """Return pending or failed slots as a compact EVA-CLIENT handoff list."""
        plan = self._plan_state(batch)
        robot_type = str(plan["info"].get("robot_type", ""))
        rows: list[dict[str, Any]] = []
        for task in plan["tasks"]:
            for slot in task["slots"]:
                if slot["qc_state"] not in {"pending", "failed"}:
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
                        "status": slot["qc_state"],
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
                if slot["qc_state"] in {"pending", "failed"}:
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
        episode = next(
            (
                slot.get("episode")
                for task in plan["tasks"]
                for slot in task["slots"]
                if slot.get("episode")
                and int(slot["episode"].get("episode_index", -1)) == episode_index
            ),
            None,
        )
        source_batch = str(episode.get("_source_batch", "")) if episode else ""
        source_index = (
            int(episode.get("_source_episode_index", episode_index)) if episode else episode_index
        )
        dataset_dir = (
            Path(self._plan_state(source_batch)["collection"]["dataset_dir"])
            if source_batch
            else Path(plan["collection"]["dataset_dir"])
        )
        return self._cached_video_path(dataset_dir, source_index, video_key)

    def mark_qc(
        self,
        batch: str,
        episode_index: int,
        verdict: str,
        note: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        if self._unmatched_robot(batch) is not None:
            raise ValueError("unmatched batches are read-only")
        if verdict not in {"", "pass", "fail", "unreviewed"}:
            raise ValueError("QC verdict must be pass, fail, unreviewed, or empty")
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
        self.invalidate(batch)
        updated_plan = self._plan_state(batch)
        return {
            "batch_id": batch,
            "episode_index": episode_index,
            "verdict": verdict,
            "note": note,
            "reason": reason,
            "tasks": updated_plan["tasks"],
        }

    def trim_episode(
        self,
        batch: str,
        episode_index: int,
        start_frame: int,
        end_frame: int,
    ) -> dict[str, Any]:
        """Trim one recorded episode in place and keep all LeRobot streams aligned."""
        if self._unmatched_robot(batch) is not None:
            raise ValueError("unmatched batches are read-only")
        plan = self._plan_state(batch)
        dataset_dir = Path(plan["collection"]["dataset_dir"])
        info = self._dataset_info(dataset_dir)
        fps = float(info.get("fps", 30) or 30)
        store = LeRobotDatasetIO(dataset_dir)
        parquet_path = store.episode_parquet(episode_index)
        if not parquet_path.is_file():
            raise RecordNotFoundError(f"Episode {episode_index}")
        episode_rows = self.episode_rows(dataset_dir)
        if not any(int(row.get("episode_index", -1)) == episode_index for row in episode_rows):
            raise RecordNotFoundError(f"Episode {episode_index}")
        table = pq.read_table(str(parquet_path))
        start, end = self._trim_bounds(start_frame, end_frame, table.num_rows)
        trimmed = table.slice(start, end - start)
        trimmed = self._reset_frame_columns(trimmed, fps)

        video_paths = store.episode_video_paths(
            episode_index,
            {key: key for key in store.infer_keys()["image"]["candidates"]},
        )
        temporary_videos: dict[Path, Path] = {}
        temporary_parquet = parquet_path.with_name(parquet_path.name + ".trim.tmp")
        try:
            for path in video_paths.values():
                temporary_videos[path] = self._trim_video(path, start, end, fps)
            pq.write_table(trimmed, str(temporary_parquet))
            for path, temporary in temporary_videos.items():
                temporary.replace(path)
            temporary_parquet.replace(parquet_path)
            length = end - start
            self._update_episode_metadata(dataset_dir, episode_index, length, fps, start, end)
        finally:
            temporary_parquet.unlink(missing_ok=True)
            for temporary in temporary_videos.values():
                temporary.unlink(missing_ok=True)
        self.invalidate(batch)
        return {
            "batch_id": batch,
            "episode_index": episode_index,
            "start_frame": start,
            "end_frame": end,
            "length": end - start,
        }

    @staticmethod
    def _trim_bounds(start_frame: int, end_frame: int, total: int) -> tuple[int, int]:
        start = int(start_frame)
        end = int(end_frame)
        if start < 0 or end > total or start >= end:
            raise ValueError(f"Trim range must satisfy 0 <= start < end <= {total}")
        return start, end

    @staticmethod
    def _reset_frame_columns(table: pa.Table, fps: float) -> pa.Table:
        length = table.num_rows
        for name in ("frame_index", "index"):
            if name in table.column_names:
                values = pa.array(range(length), type=table.column(name).type)
                table = table.set_column(table.column_names.index(name), name, values)
        if "timestamp" in table.column_names and length:
            timestamps = [float(value) for value in table.column("timestamp").to_pylist()]
            first = timestamps[0]
            normalized = [value - first for value in timestamps]
            if not all(np.isfinite(value) for value in normalized):
                normalized = [index / fps for index in range(length)]
            table = table.set_column(
                table.column_names.index("timestamp"),
                "timestamp",
                pa.array(normalized, type=table.column("timestamp").type),
            )
        return table

    @staticmethod
    def _trim_video(path: Path, start: int, end: int, fps: float) -> Path:
        temporary = path.with_name(path.stem + ".trim.mp4")
        reader = imageio.get_reader(str(path))
        writer = imageio.get_writer(
            str(temporary),
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        )
        observed = 0
        try:
            for index, frame in enumerate(reader):
                observed = index + 1
                if index >= end:
                    break
                if index >= start:
                    writer.append_data(np.ascontiguousarray(frame))
        finally:
            reader.close()
            writer.close()
        if observed < end:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Video {path.name} has {observed} frames; expected at least {end}")
        return temporary

    def _update_episode_metadata(
        self,
        dataset_dir: Path,
        episode_index: int,
        length: int,
        fps: float,
        start: int,
        end: int,
    ) -> None:
        episodes_path = dataset_dir / "meta" / "episodes.jsonl"
        rows = self.episode_rows(dataset_dir)
        found = False
        for row in rows:
            if int(row.get("episode_index", -1)) != episode_index:
                continue
            row["length"] = length
            row["duration_seconds"] = round(length / fps, 6)
            row["trim_start_frame"] = start
            row["trim_end_frame"] = end
            found = True
            break
        if not found:
            raise RecordNotFoundError(f"Episode {episode_index}")
        temporary = episodes_path.with_name(episodes_path.name + ".trim.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(episodes_path)
        info_path = dataset_dir / "meta" / "info.json"
        info = self._json_file(info_path)
        info["total_frames"] = sum(int(row.get("length", 0) or 0) for row in rows)
        info_path.write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

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
        episode = next(
            (
                slot.get("episode")
                for task in plan["tasks"]
                for slot in task["slots"]
                if slot.get("episode")
                and int(slot["episode"].get("episode_index", -1)) == episode_index
            ),
            None,
        )
        source_batch = str(episode.get("_source_batch", "")) if episode else ""
        source_index = (
            int(episode.get("_source_episode_index", episode_index)) if episode else episode_index
        )
        dataset_dir = (
            Path(self._plan_state(source_batch)["collection"]["dataset_dir"])
            if source_batch
            else Path(plan["collection"]["dataset_dir"])
        )
        parquet = LeRobotDatasetIO(dataset_dir).episode_parquet(source_index)
        stat = parquet.stat()
        cache_key = (
            str(parquet),
            int(episode_index),
            int(start),
            int(count),
            int(stat.st_mtime_ns),
        )
        with self._lock:
            cached = self._transform_cache.get(cache_key)
            if cached is not None:
                self._transform_cache.move_to_end(cache_key)
                return cached
        series = self._series(dataset_dir, source_index)
        qpos = np.asarray(series["state"], dtype=np.float32)
        total = len(qpos)
        start = min(max(0, start), total)
        end = min(total, start + min(max(1, count), 120))
        _, scene = self._robot_scene(str(plan["info"].get("robot_type", "")))
        result = scene.all_transforms_blob(qpos[start:end]), start, total
        with self._lock:
            self._transform_cache[cache_key] = result
            self._transform_cache.move_to_end(cache_key)
            while len(self._transform_cache) > TRANSFORM_CACHE_MAX:
                self._transform_cache.popitem(last=False)
        return result

    def _plan_state(self, batch: str) -> dict[str, Any]:
        batch_id = self._batch_id(batch)
        if self._unmatched_robot(batch_id) is not None:
            return self._unmatched_state(batch_id)
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
            state = dict(self._raw_state(batch_id))
            collection = self._decorate(batch_id, state)
            state["batch_id"] = batch_id
            state["dataset_dir"] = str(self._batch_root(batch_id))
            state["tasks"] = collection.pop("tasks")
            state["slot_order"] = collection.pop("slot_order")
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

    def invalidate(self, batch: str = "") -> None:
        with self._state_lock:
            if batch:
                batch_id = self._batch_id(batch)
                self._plan_cache.pop(batch_id, None)
                self._summary_cache.pop(batch_id, None)
                self._raw_state_cache.pop(batch_id, None)
                self._unmatched_cache.clear()
                self._source_token_cache.clear()
            else:
                self._plan_cache.clear()
                self._summary_cache.clear()
                self._raw_state_cache.clear()
                self._unmatched_cache.clear()
                self._dataset_info_cache.clear()
                self._dataset_stems_cache.clear()
                self._episode_rows_cache.clear()
                self._inferred_keys_cache.clear()
                self._source_token_cache.clear()
        # Persistent state entries include source mtimes/sizes in their key,
        # so a changed plan or episode naturally bypasses old entries. Avoid
        # deleting the whole database on every QC click.

    def _source_token(self, batches: list[str]) -> str:
        """Return a cheap mtime/size token for cache invalidation."""
        cache_key = tuple(batches)
        now = time.monotonic()
        cached = self._source_token_cache.get(cache_key)
        if cached is not None and now - cached[0] < 1.0:
            return cached[1]
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
                dataset_dir / "meta" / "qc.jsonl",
            ):
                parts.append(self._file_token(path))
        token = "|".join(parts)
        self._source_token_cache[cache_key] = (now, token)
        return token

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
        # One dataset, one directory: the plan names it, otherwise it is the
        # canonical location under the shared data root.
        dataset_name = str(info.get("dataset_name") or batch)
        robot_type = str(info.get("robot_type", ""))
        located = self.collection_root / "datasets" / "real_robot" / robot_type / dataset_name
        return located.resolve()

    def _decorate(self, batch: str, state: dict[str, Any]) -> dict[str, Any]:
        dataset_dir = self._dataset_dir(batch, state["info"])
        episodes = self._annotate_static_qc(dataset_dir, self.episode_rows(dataset_dir))
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
        counts = {"pending": 0, "unreviewed": 0, "passed": 0, "failed": 0, "total": 0}
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
                    counts[slot["qc_state"]] += 1
                    counts["total"] += 1
            decorated["counts"] = self._slot_counts(decorated["slots"])
            tasks.append(decorated)
        info = self._dataset_info(dataset_dir)
        return {
            "dataset_dir": str(dataset_dir),
            "available": (dataset_dir / "meta" / "episodes.jsonl").is_file(),
            "robot_type": str(info.get("robot_type", "")),
            "counts": counts,
            "duplicates": duplicates,
            "orphaned": [self._episode_summary(row) for row in episodes if id(row) not in matched],
            "tasks": tasks,
            "slot_order": self._slot_order(batch, tasks),
        }

    def _slot_order(self, batch: str, tasks: list[dict[str, Any]]) -> list[str]:
        """Order the batch's slots the way the console's collect grid does.

        The console owns the scene -> task -> round order operators work in, so
        the plan is expanded here from the same task-set directory and every
        slot carries its place in that order; the review grid numbers the tiles
        from it.
        """
        slots = {slot["slot_id"]: slot for task in tasks for slot in task.get("slots", [])}
        order = [
            slot.slot_id
            for slot in plan_slots(scene_plan_from_dir(self._batch_root(batch)))
            if slot.slot_id in slots
        ]
        # Slots no longer described by the plan keep an authored place at the end.
        order.extend(slot_id for slot_id in slots if slot_id not in set(order))
        for ordinal, slot_id in enumerate(order):
            slots[slot_id]["ordinal"] = ordinal
        return order

    def _dataset_dir(self, batch: str, info: dict[str, Any]) -> Path:
        """The dataset's one directory: ``collection_dir`` or the shared layout."""
        return self._source_dataset_dir(batch, info)

    def frame_analysis(self, dataset_dir: Path, episode_index: int) -> dict[str, Any]:
        """Recorded frame labelling of one episode, shared by review and auto QC."""
        return self._series(dataset_dir, episode_index)["frame_label_analysis"]

    def _series(self, dataset_dir: Path, episode_index: int) -> dict[str, Any]:
        io_store = LeRobotDatasetIO(dataset_dir)
        inferred = self._inferred_keys(dataset_dir)
        state_key = inferred["state"]["default"]
        action_key = inferred["action"]["default"]
        if not state_key or not action_key:
            raise ValueError("Episode does not declare state and action features")
        path = io_store.episode_parquet(episode_index)
        stat = path.stat()
        cache_key = (str(path), int(episode_index), int(stat.st_mtime_ns), int(stat.st_size))
        db_key = "series:" + ":".join(str(value) for value in cache_key)
        series_token = f"{stat.st_mtime_ns}:{stat.st_size}"
        with self._lock:
            cached = self._series_cache.get(cache_key)
            if cached is not None:
                self._series_cache.move_to_end(cache_key)
                return cached
        db_series = self._db_cache_get(db_key, series_token, 3600.0)
        if isinstance(db_series, dict):
            with self._lock:
                self._series_cache[cache_key] = db_series
                self._series_cache.move_to_end(cache_key)
                while len(self._series_cache) > SERIES_CACHE_MAX:
                    self._series_cache.popitem(last=False)
            return db_series
        schema = pq.read_schema(path)
        columns = [state_key, action_key]
        if "timestamp" in schema.names:
            columns.append("timestamp")
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            analysis = self._frame_label_analysis([])
            series = {
                "timestamp": [],
                "state": [],
                "action": [],
                "state_names": [],
                "action_names": [],
                "frame_labels": [],
                "frame_label_analysis": analysis,
                "frame_label_segments": analysis["segments"],
                "static_segments": analysis["static_segments"],
                "middle_static_frames": analysis["middle_static_frames"],
                "machine_label": analysis["machine_label"],
                "trim_start_frame": analysis["trim_start_frame"],
                "trim_end_frame": analysis["trim_end_frame"],
                "video_keys": inferred["image"]["candidates"],
            }
            with self._lock:
                self._series_cache[cache_key] = series
                self._series_cache.move_to_end(cache_key)
                while len(self._series_cache) > SERIES_CACHE_MAX:
                    self._series_cache.popitem(last=False)
            self._db_cache_put(db_key, series_token, series)
            return series
        info = self._dataset_info(dataset_dir)
        fps = float(info.get("fps", 30) or 30)
        timestamp = (
            [float(value) for value in table.column("timestamp").to_pylist()]
            if "timestamp" in columns
            else [index / fps for index in range(table.num_rows)]
        )
        action = table.column(action_key).to_pylist()
        frame_labels = self._action_frame_labels(action)
        analysis = self._frame_label_analysis(frame_labels)
        series = {
            "timestamp": timestamp,
            "state": table.column(state_key).to_pylist(),
            "action": action,
            "state_names": self._feature_names(info, state_key, table.column(state_key)[0]),
            "action_names": self._feature_names(info, action_key, table.column(action_key)[0]),
            "frame_labels": frame_labels,
            "frame_label_analysis": analysis,
            "frame_label_segments": analysis["segments"],
            "static_segments": analysis["static_segments"],
            "middle_static_frames": analysis["middle_static_frames"],
            "machine_label": analysis["machine_label"],
            "trim_start_frame": analysis["trim_start_frame"],
            "trim_end_frame": analysis["trim_end_frame"],
            "video_keys": inferred["image"]["candidates"],
        }
        with self._lock:
            self._series_cache[cache_key] = series
            self._series_cache.move_to_end(cache_key)
            while len(self._series_cache) > SERIES_CACHE_MAX:
                self._series_cache.popitem(last=False)
        self._db_cache_put(db_key, series_token, series)
        return series

    @staticmethod
    def _action_frame_labels(actions: list[Any]) -> list[str]:
        if not actions:
            return []
        labels = ["static"]
        previous = np.asarray(actions[0], dtype=np.float64)
        for value in actions[1:]:
            current = np.asarray(value, dtype=np.float64)
            delta = np.max(np.abs(current - previous))
            labels.append("static" if delta <= 1e-4 else "non-static")
            previous = current
        return labels

    @staticmethod
    def _frame_label_analysis(labels: list[str]) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        start = 0
        while start < len(labels):
            label = labels[start] or "unlabeled"
            end = start + 1
            while end < len(labels) and (labels[end] or "unlabeled") == label:
                end += 1
            segments.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "start_frame": start,
                    "end_frame": end - 1,
                    "length": end - start,
                }
            )
            start = end
        static = [segment for segment in segments if segment["label"] == "static"]
        leading = static[0] if static and static[0]["start"] == 0 else None
        trailing = static[-1] if static and static[-1]["end"] == len(labels) else None
        edge_ids = {id(segment) for segment in (leading, trailing) if segment is not None}
        middle = [
            segment
            for segment in static
            if id(segment) not in edge_ids and int(segment["length"]) > STATIC_RUN_MIN_FRAMES
        ]
        only_static = leading is not None and leading is trailing
        trim_start = 0 if only_static else int(leading["end"]) if leading is not None else 0
        trim_end = (
            len(labels)
            if only_static
            else int(trailing["start"])
            if trailing is not None
            else len(labels)
        )
        middle_frames = sum(int(segment["length"]) for segment in middle)
        return {
            "segments": segments,
            "static_segments": static,
            "leading_static": leading,
            "trailing_static": trailing,
            "middle_static_segments": middle,
            "middle_static_frames": middle_frames,
            "static_frames_excessive": bool(middle),
            "machine_label": STATIC_FRAMES_REASON if middle else "",
            "trim_start_frame": trim_start,
            "trim_end_frame": trim_end,
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

    def _raw_state(self, batch: str) -> dict[str, Any]:
        batch_id = self._batch_id(batch)
        source_token = self._source_token([batch_id])
        with self._state_lock:
            cached = self._raw_state_cache.get(batch_id)
            if cached is not None and cached[0] == source_token:
                return cached[1]
            state = self._store(batch_id).state()
            self._raw_state_cache[batch_id] = (source_token, state)
            return state

    def _batch_summary(
        self,
        batch: str,
        state: dict[str, Any] | None = None,
        analyze_static_qc: bool = False,
    ) -> dict[str, Any]:
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
            state = self._raw_state(batch)
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
        episode_rows = self.episode_rows(collection_dir)
        if analyze_static_qc:
            episode_rows = self._annotate_static_qc(collection_dir, episode_rows)
        slot_ids = set()
        slot_targets: dict[tuple[str, str, int], str] = {}
        for task in state["tasks"]:
            for scene_id, count in zip(
                task["scene_ids"], task["scene_epsiodes_count"], strict=False
            ):
                for round_index in range(count):
                    slot_id = f"{task['task_id']}:{scene_id}:{round_index}"
                    slot_ids.add(slot_id)
                    slot_targets[(str(task["task_id"]), str(scene_id), round_index)] = slot_id
        matched_rows: dict[str, dict[str, Any]] = {}
        for row in episode_rows:
            row_slot = str(row.get("slot_id", ""))
            if row_slot in slot_ids:
                slot_id = row_slot
            else:
                target_key = (
                    str(row.get("task_id", "")),
                    str(row.get("scene_id", "")),
                    int(row.get("scene_round", -1)),
                )
                slot_id = slot_targets.get(target_key)
            if slot_id is None:
                continue
            previous = matched_rows.get(slot_id)
            if previous is None or int(row.get("episode_index", -1)) > int(
                previous.get("episode_index", -1)
            ):
                matched_rows[slot_id] = row
        collected_rows = sorted(
            matched_rows.values(), key=lambda row: int(row.get("episode_index", -1))
        )
        dataset_info = self._dataset_info(collection_dir)
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
            status_counts[qc_state(row.get("qc_verdict"), row.get("quality"))] += 1
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
        state = (
            "pending" if summary is None else qc_state(summary["qc_verdict"], summary["quality"])
        )
        return {
            "slot_id": slot_id,
            "scene_id": scene_id,
            "round_index": round_index,
            "round_total": round_total,
            "qc_state": state,
            "episode": summary,
        }

    @staticmethod
    def _slot_counts(slots: list[dict[str, Any]]) -> dict[str, int]:
        """How many of these slots sit in each QC state."""
        counts = {"pending": 0, "unreviewed": 0, "passed": 0, "failed": 0}
        for slot in slots:
            counts[slot["qc_state"]] += 1
        return counts

    @staticmethod
    def _episode_summary(row: dict[str, Any]) -> dict[str, Any]:
        analysis = row.get("frame_label_analysis", {})
        status = str(row.get("status", "")).lower()
        return {
            "episode_index": int(row.get("episode_index", -1)),
            "length": int(row.get("length", 0)),
            "quality": str(row.get("quality", "green")),
            "qc_verdict": str(row.get("qc_verdict") or ("fail" if status == "failed" else "")),
            "qc_note": str(row.get("qc_note") or row.get("notes") or ""),
            "qc_reason": str(row.get("qc_reason") or row.get("reason") or ""),
            "qc_auto": bool(row.get("qc_auto", False)),
            "qc_auto_note": str(row.get("qc_auto_note") or ""),
            "qc_auto_reason": str(row.get("qc_auto_reason") or ""),
            "frame_label_analysis": analysis,
            "middle_static_frames": int(analysis.get("middle_static_frames", 0) or 0),
            "machine_label": str(analysis.get("machine_label", "")),
            "slot_id": str(row.get("slot_id", "")),
            "task_id": str(row.get("task_id", "")),
            "scene_id": str(row.get("scene_id", "")),
            "scene_round": int(row.get("scene_round", -1)),
            "taskset_match": str(row.get("taskset_match", "")),
            "taskset_match_method": str(row.get("taskset_match_method", "")),
            "source_scene_id": str(row.get("source_scene_id", "")),
            "_source_batch": str(row.get("_source_batch", "")),
            "_source_episode_index": int(
                row.get("_source_episode_index", row.get("episode_index", -1))
            ),
        }

    def _dataset_info(self, dataset_dir: Path) -> dict[str, Any]:
        """Read one dataset info file once per mtime/size token."""
        path = dataset_dir / "meta" / "info.json"
        token = self._file_token(path)
        key = str(path)
        cached = self._dataset_info_cache.get(key)
        if cached is not None and cached[0] == token:
            return cached[1]
        db_info = self._db_cache_get(f"info:{key}", token)
        if isinstance(db_info, dict):
            self._dataset_info_cache[key] = (token, db_info)
            return db_info
        info = self._json_file(path)
        self._dataset_info_cache[key] = (token, info)
        self._db_cache_put(f"info:{key}", token, info)
        return info

    def _episode_stems(self, dataset_dir: Path) -> dict[int, str]:
        """Cache optional episode file stems used by video URL resolution."""
        path = dataset_dir / "meta" / "episodes.jsonl"
        token = self._file_token(path)
        key = str(path)
        cached = self._dataset_stems_cache.get(key)
        if cached is not None and cached[0] == token:
            return cached[1]
        db_stems = self._db_cache_get(f"stems:{key}", token)
        if isinstance(db_stems, dict):
            stems = {int(index): str(value) for index, value in db_stems.items()}
            self._dataset_stems_cache[key] = (token, stems)
            return stems
        stems: dict[int, str] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in filter(str.strip, handle):
                    row = json.loads(line)
                    index = int(row.get("episode_index", -1))
                    if index >= 0 and row.get("file_stem"):
                        stems[index] = str(row["file_stem"])
        self._dataset_stems_cache[key] = (token, stems)
        self._db_cache_put(f"stems:{key}", token, stems)
        return stems

    def _inferred_keys(self, dataset_dir: Path) -> dict[str, Any]:
        """Cache feature-role inference; it only depends on info.json."""
        path = dataset_dir / "meta" / "info.json"
        token = self._file_token(path)
        key = str(path)
        cached = self._inferred_keys_cache.get(key)
        if cached is not None and cached[0] == token:
            return cached[1]
        db_inferred = self._db_cache_get(f"inferred:{key}", token)
        if isinstance(db_inferred, dict):
            self._inferred_keys_cache[key] = (token, db_inferred)
            return db_inferred
        inferred = LeRobotDatasetIO(dataset_dir).infer_keys()
        self._inferred_keys_cache[key] = (token, inferred)
        self._db_cache_put(f"inferred:{key}", token, inferred)
        return inferred

    def camera_videos(
        self, dataset_dir: Path, episode_index: int, cameras: Iterable[str]
    ) -> dict[str, Path]:
        """Recorded video file of each named camera in one episode."""
        candidates = self._inferred_keys(dataset_dir)["image"]["candidates"]
        keys = {
            camera: next(
                (name for name in candidates if name == camera or name.endswith("." + camera)),
                None,
            )
            for camera in cameras
        }
        return {
            camera: self._cached_video_path(dataset_dir, episode_index, key)
            for camera, key in keys.items()
            if key
        }

    def _cached_video_path(self, dataset_dir: Path, episode_index: int, video_key: str) -> Path:
        info = self._dataset_info(dataset_dir)
        inferred = self._inferred_keys(dataset_dir)
        if video_key not in inferred["image"]["candidates"]:
            raise ValueError("Unknown episode video key")
        chunks_size = int(info.get("chunks_size", 1000) or 1000)
        episode_chunk = episode_index // chunks_size
        stem = self._episode_stems(dataset_dir).get(episode_index) or f"episode_{episode_index:06d}"
        template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        path = dataset_dir / str(template).format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
            episode_file=stem,
        )
        if not path.is_file():
            raise FileNotFoundError(video_key)
        return path

    def episode_rows(self, dataset_dir: Path) -> list[dict[str, Any]]:
        path = dataset_dir / "meta" / "episodes.jsonl"
        qc_path = dataset_dir / "meta" / "qc.jsonl"
        token = self._file_token(path) + "|" + self._file_token(qc_path)
        key = str(path)
        cached = self._episode_rows_cache.get(key)
        if cached is not None and cached[0] == token:
            return cached[1]
        db_rows = self._db_cache_get(f"episodes:{key}", token)
        if isinstance(db_rows, list):
            rows = [dict(row) for row in db_rows if isinstance(row, dict)]
            self._episode_rows_cache[key] = (token, rows)
            return rows
        if not path.is_file():
            rows: list[dict[str, Any]] = []
        else:
            rows = []
            with path.open(encoding="utf-8") as handle:
                for line in filter(str.strip, handle):
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        qc_rows = []
        if qc_path.is_file():
            qc_rows = [
                json.loads(line) for line in qc_path.read_text().splitlines() if line.strip()
            ]
        qc_by_episode = {int(row["episode_index"]): row for row in qc_rows}
        # QC files written against another enumeration keep their slot column as
        # the only trustworthy link; recordings own their identity fields.
        qc_by_slot = {
            str(row["slot_id"]): row for row in qc_rows if str(row.get("slot_id", "")).strip()
        }
        for row in rows:
            source = qc_by_slot.get(str(row.get("slot_id", ""))) or qc_by_episode.get(
                int(row.get("episode_index", -1)), {}
            )
            row.update({key: value for key, value in source.items() if key.startswith("qc_")})
        self._episode_rows_cache[key] = (token, rows)
        self._db_cache_put(f"episodes:{key}", token, rows)
        return rows

    def _annotate_static_qc(
        self, dataset_dir: Path, rows: list[dict[str, Any]], analyze: bool | None = None
    ) -> list[dict[str, Any]]:
        """Attach static-frame QC without blocking large catalog loads.

        Small collections retain the existing automatic classification. For
        large collections, reading every episode parquet file is deferred until
        the selected review series is requested.
        """
        if analyze is None:
            analyze = len(rows) <= STATIC_QC_ANALYSIS_LIMIT
        if not analyze:
            return rows
        for row in rows:
            episode_index = int(row.get("episode_index", -1))
            if episode_index < 0:
                continue
            verdict = str(row.get("qc_verdict") or "").strip()
            if verdict:
                continue
            try:
                analysis = self._series(dataset_dir, episode_index)["frame_label_analysis"]
            except (FileNotFoundError, ValueError):
                continue
            row["frame_label_analysis"] = analysis
            if analysis["static_frames_excessive"] and not verdict:
                row["qc_verdict"] = "fail"
                row["qc_reason"] = STATIC_FRAMES_REASON
                row["qc_auto"] = True
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
