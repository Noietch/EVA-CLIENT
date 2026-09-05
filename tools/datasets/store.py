"""Read, validate, and atomically persist an EVA task plan with shared objects."""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import re
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

FILES = ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv")
MAX_ARCHIVE_FILES, MAX_ARCHIVE_BYTES = 64, 32 * 1024 * 1024
FIELDS = {
    "objects": ["object_id", "object_name", "object_name_zh", "color"],
    "scenes": ["scene_id", "placements"],
    "tasks": (
        "task_id action category operation_object_ids prompt_en prompt_zh "
        "total_epsiodes_count scene_ids scene_epsiodes_count"
    ).split(),
}
ID_FIELDS = {"objects": "object_id", "scenes": "scene_id", "tasks": "task_id"}
DEFAULT_LAYOUT = {
    "coordinate_frame": "table_top_left",
    "unit": "mm",
    "orientation": "x_right_y_down",
    "calibration_status": "unverified",
    "bounds": {"width": 500, "height": 500},
    "sampling_points": [
        {"position_id": f"P{row * 3 + column + 1}", "x": column * 250, "y": row * 250}
        for row in range(3)
        for column in range(3)
    ],
}


class ConflictError(ValueError): ...


class RecordNotFoundError(LookupError): ...


class TaskSetStore:
    """Own canonical task-set parsing, validation, and atomic persistence."""

    def __init__(self, root: str | Path, objects_path: str | Path | None = None):
        self.root = Path(root).expanduser().resolve()
        self.objects_path = Path(objects_path).expanduser().resolve() if objects_path else None
        self.lock = threading.RLock()

    def state(self) -> dict[str, Any]:
        with self.lock, self._write_lock():
            state = self._read_state()
            state["info"] = self._current_info(state)
            state["issues"] = self.validate(state)
            state["dataset_dir"] = str(self.root)
            return state

    def upsert(self, kind: str, current_id: str | None, payload: Any) -> dict[str, Any]:
        if kind not in {"scenes", "tasks"}:
            raise ValueError("Task plans support scenes and tasks")
        id_field = ID_FIELDS[kind]
        record = self._normalize_record(kind, payload)
        record_id = record[id_field]
        if current_id is not None and record_id != current_id:
            raise ValueError(f"{id_field} cannot be changed")

        with self.lock, self._write_lock():
            state = self._read_state()
            records = state[kind]
            index = next((i for i, item in enumerate(records) if item[id_field] == record_id), -1)
            if current_id is None and index >= 0:
                raise ConflictError(f"{record_id} already exists")
            if current_id is not None and index < 0:
                raise RecordNotFoundError(current_id)
            if index < 0:
                records.append(record)
            else:
                records[index] = record

            errors = [issue for issue in self.validate(state) if issue["level"] == "error"]
            if errors:
                raise ValueError("; ".join(issue["message"] for issue in errors[:3]))
            self._write_state(state, kind)
        return self.state()

    def delete(self, kind: str, record_id: str) -> dict[str, Any]:
        if kind not in {"scenes", "tasks"}:
            raise ValueError("Task plans support scenes and tasks")
        id_field = ID_FIELDS[kind]
        with self.lock, self._write_lock():
            state = self._read_state()
            records = state[kind]
            if not any(item[id_field] == record_id for item in records):
                raise RecordNotFoundError(record_id)
            references = self._references(state, kind, record_id)
            if references:
                raise ConflictError(f"{record_id} is referenced by {', '.join(references[:6])}")
            state[kind] = [item for item in records if item[id_field] != record_id]
            self._write_state(state, kind)
        return self.state()

    def update_info(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("info must be an object")
        with self.lock, self._write_lock():
            state = self._read_state()
            state["info"].update(
                {
                    "dataset_name": str(payload.get("dataset_name", "")).strip(),
                    "robot_type": str(payload.get("robot_type", "")).strip(),
                    "collection_dir": str(payload.get("collection_dir", "")).strip(),
                }
            )
            self._write_state(state)
        return self.state()

    def export_zip(self) -> io.BytesIO:
        with self.lock, self._write_lock():
            state = self._read_state()
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("info.yaml", self._yaml_text(self._current_info(state)))
                archive.writestr("layout.yaml", self._yaml_text(state["layout"]))
                for kind in ("scenes", "tasks"):
                    stem = "scene" if kind == "scenes" else kind
                    archive.writestr(f"{stem}.csv", self._csv_text(kind, state[kind]))
            stream.seek(0)
            return stream

    def import_zip(self, content: bytes) -> dict[str, Any]:
        if not content:
            raise ValueError("Choose a non-empty ZIP file")
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ARCHIVE_FILES:
                    raise ValueError(f"Archive contains more than {MAX_ARCHIVE_FILES} files")
                supported = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                if any(info.flag_bits & 1 for info in infos):
                    raise ValueError("Encrypted ZIP archives are not supported")
                if any(info.compress_type not in supported for info in infos):
                    raise ValueError("ZIP archive uses an unsupported compression method")
                if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive expands beyond 32 MiB")
                members: dict[str, str] = {}
                for name in archive.namelist():
                    base = PurePosixPath(name).name
                    if base in FILES:
                        if base in members:
                            raise ValueError(f"Archive contains more than one {base}")
                        members[base] = name
                missing = [name for name in FILES if name not in members]
                if missing:
                    raise ValueError(f"Archive is missing {', '.join(missing)}")
                documents = {name: archive.read(member) for name, member in members.items()}
        except zipfile.BadZipFile as exc:
            raise ValueError("Uploaded file is not a valid ZIP archive") from exc

        try:
            state = self._decode_documents(documents)
        except (UnicodeError, yaml.YAMLError, csv.Error) as exc:
            raise ValueError(f"Archive contains invalid task-set data: {exc}") from exc
        errors = [issue for issue in self.validate(state) if issue["level"] == "error"]
        if errors:
            raise ValueError("; ".join(issue["message"] for issue in errors[:5]))
        with self.lock, self._write_lock():
            documents = {
                "info.yaml": self._yaml_text(self._current_info(state)),
                "layout.yaml": self._yaml_text(state["layout"]),
                "scene.csv": self._csv_text("scenes", state["scenes"]),
                "tasks.csv": self._csv_text("tasks", state["tasks"]),
            }
            self._commit_documents(documents)
        return self.state()

    def validate(self, state: dict[str, Any] | None = None) -> list[dict[str, str]]:
        if state is None:
            with self.lock, self._write_lock():
                state = self._read_state()
        issues: list[dict[str, str]] = []
        object_ids = self._unique_ids(state["objects"], "object_id", "object", issues)
        scene_ids = self._unique_ids(state["scenes"], "scene_id", "scene", issues)
        self._unique_ids(state["tasks"], "task_id", "task", issues)
        points = state["layout"].get("sampling_points", [])
        if not isinstance(points, list):
            issues.append(self._issue("error", "layout", "sampling_points must be a list"))
            points = []
        position_ids = self._unique_ids(points, "position_id", "position", issues)

        for scene in state["scenes"]:
            scene_id = scene["scene_id"]
            if not scene["placements"]:
                issues.append(self._issue("warning", scene_id, "Scene has no placements"))
            for placement in scene["placements"]:
                if placement["object_id"] not in object_ids:
                    issues.append(
                        self._issue("error", scene_id, f"Unknown object {placement['object_id']}")
                    )
                unknown = set(placement["position_ids"]) - position_ids
                if unknown:
                    issues.append(
                        self._issue(
                            "error", scene_id, f"Unknown positions {', '.join(sorted(unknown))}"
                        )
                    )

        for task in state["tasks"]:
            task_id = task["task_id"]
            unknown_objects = set(task["operation_object_ids"]) - object_ids
            unknown_scenes = set(task["scene_ids"]) - scene_ids
            if unknown_objects:
                issues.append(
                    self._issue(
                        "error", task_id, f"Unknown objects {', '.join(sorted(unknown_objects))}"
                    )
                )
            if unknown_scenes:
                issues.append(
                    self._issue(
                        "error", task_id, f"Unknown scenes {', '.join(sorted(unknown_scenes))}"
                    )
                )
            counts = task["scene_epsiodes_count"]
            if not task["scene_ids"]:
                issues.append(self._issue("error", task_id, "Task must reference a scene"))
            if len(set(task["scene_ids"])) != len(task["scene_ids"]):
                issues.append(self._issue("error", task_id, "Task contains duplicate scenes"))
            if len(task["scene_ids"]) != len(counts):
                issues.append(self._issue("error", task_id, "Scene and episode counts differ"))
            if sum(counts) != task["total_epsiodes_count"]:
                issues.append(
                    self._issue("error", task_id, "Total episodes does not match scene counts")
                )
            if not task["prompt_en"]:
                issues.append(self._issue("error", task_id, "Task requires an English prompt"))
        return issues

    def _read_state(self) -> dict[str, Any]:
        info = self._read_yaml("info.yaml")
        layout = self._read_yaml("layout.yaml") or json.loads(json.dumps(DEFAULT_LAYOUT))
        return {
            "info": info,
            "layout": layout,
            "objects": [
                self._normalize_record("objects", row)
                for row in self._read_csv_path(self._objects_file(info))
            ],
            "scenes": [
                self._normalize_record("scenes", row) for row in self._read_csv("scene.csv")
            ],
            "tasks": [self._normalize_record("tasks", row) for row in self._read_csv("tasks.csv")],
        }

    def _decode_documents(self, documents: dict[str, bytes]) -> dict[str, Any]:
        def yaml_doc(name: str) -> dict[str, Any]:
            value = yaml.safe_load(documents[name].decode("utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError(f"{name} must contain a mapping")
            return value

        def csv_rows(name: str) -> list[dict[str, str]]:
            text = documents[name].decode("utf-8-sig")
            return [dict(row) for row in csv.DictReader(io.StringIO(text))]

        info = yaml_doc("info.yaml")
        configured_info = self._read_yaml("info.yaml")
        info["objects_file"] = self._objects_reference(configured_info)
        return {
            "info": info,
            "layout": yaml_doc("layout.yaml"),
            "objects": [
                self._normalize_record("objects", row)
                for row in self._read_csv_path(self._objects_file(configured_info))
            ],
            "scenes": [self._normalize_record("scenes", row) for row in csv_rows("scene.csv")],
            "tasks": [self._normalize_record("tasks", row) for row in csv_rows("tasks.csv")],
        }

    def _normalize_record(self, kind: str, payload: Any) -> dict[str, Any]:
        if kind not in ID_FIELDS or not isinstance(payload, dict):
            raise ValueError("Invalid record type")
        id_field = ID_FIELDS[kind]
        record_id = str(payload.get(id_field, "")).strip()
        if not record_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", record_id):
            raise ValueError(f"{id_field} must use letters, numbers, '.', '_' or '-'")
        if kind == "objects":
            return {
                "object_id": record_id,
                "object_name": str(payload.get("object_name", "")).strip(),
                "object_name_zh": str(payload.get("object_name_zh", "")).strip(),
                "color": str(payload.get("color", "")).strip(),
            }
        if kind == "scenes":
            placements = payload.get("placements", [])
            if isinstance(placements, str):
                placements = json.loads(placements or "[]")
            if not isinstance(placements, list):
                raise ValueError("placements must be a list")
            return {
                "scene_id": record_id,
                "placements": [self._placement(item) for item in placements],
            }

        scene_ids = self._string_list(payload.get("scene_ids", []))
        counts = self._int_list(payload.get("scene_epsiodes_count", []))
        total = payload.get("total_epsiodes_count", sum(counts))
        return {
            "task_id": record_id,
            "action": str(payload.get("action", "")).strip(),
            "category": str(payload.get("category", "")).strip(),
            "operation_object_ids": self._string_list(payload.get("operation_object_ids", [])),
            "prompt_en": str(payload.get("prompt_en", "")).strip(),
            "prompt_zh": str(payload.get("prompt_zh", "")).strip(),
            "total_epsiodes_count": int(total or 0),
            "scene_ids": scene_ids,
            "scene_epsiodes_count": counts,
        }

    def _placement(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Each placement must be an object")
        position_ids = self._string_list(payload.get("position_ids", []))
        object_id = str(payload.get("object_id", "")).strip()
        randomized = payload.get("random", False)
        if not object_id or not position_ids or not isinstance(randomized, bool):
            raise ValueError("Each placement needs object_id, position_ids, and boolean random")
        return {"position_ids": position_ids, "object_id": object_id, "random": randomized}

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        values = value.split(";") if isinstance(value, str) else value
        if not isinstance(values, list):
            raise ValueError("Expected a list")
        return [str(item).strip() for item in values if str(item).strip()]

    @staticmethod
    def _int_list(value: Any) -> list[int]:
        values = value.split(";") if isinstance(value, str) else value
        if not isinstance(values, list):
            raise ValueError("Expected a list of episode counts")
        try:
            counts = [int(item) for item in values if str(item).strip()]
        except (TypeError, ValueError) as exc:
            raise ValueError("Episode counts must be integers") from exc
        if any(count <= 0 for count in counts):
            raise ValueError("Episode counts must be positive")
        return counts

    def _references(self, state: dict[str, Any], kind: str, record_id: str) -> list[str]:
        if kind == "scenes":
            return [task["task_id"] for task in state["tasks"] if record_id in task["scene_ids"]]
        return []

    def _write_state(self, state: dict[str, Any], changed_kind: str | None = None) -> None:
        documents = {}
        if changed_kind:
            filename = "scene.csv" if changed_kind == "scenes" else f"{changed_kind}.csv"
            documents[filename] = self._csv_text(changed_kind, state[changed_kind])
        if not (self.root / "layout.yaml").is_file():
            documents["layout.yaml"] = self._yaml_text(state["layout"])
        for kind in ("scenes", "tasks"):
            filename = "scene.csv" if kind == "scenes" else f"{kind}.csv"
            if not (self.root / filename).is_file():
                documents[filename] = self._csv_text(kind, state[kind])
        documents["info.yaml"] = self._yaml_text(self._current_info(state))
        self._commit_documents(documents)

    def _current_info(self, state: dict[str, Any]) -> dict[str, Any]:
        info = dict(state.get("info", {}))
        info.update(
            {
                "dataset_name": str(info.get("dataset_name", self.root.name)),
                "robot_type": str(info.get("robot_type", "")),
                "task_count": len(state["tasks"]),
                "target_episodes": sum(task["total_epsiodes_count"] for task in state["tasks"]),
                "tasks_file": "tasks.csv",
                "objects_file": self._objects_reference(info),
                "scenes_file": "scene.csv",
                "layout_file": "layout.yaml",
            }
        )
        return info

    def _csv_text(self, kind: str, records: list[dict[str, Any]]) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS[kind], lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            if kind == "scenes":
                row["placements"] = json.dumps(
                    row["placements"], ensure_ascii=False, separators=(",", ":")
                )
            if kind == "tasks":
                for field in ("operation_object_ids", "scene_ids", "scene_epsiodes_count"):
                    row[field] = ";".join(str(value) for value in row[field])
            writer.writerow({field: row.get(field, "") for field in FIELDS[kind]})
        return "\ufeff" + stream.getvalue()

    @staticmethod
    def _yaml_text(payload: dict[str, Any]) -> str:
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

    def _read_csv(self, filename: str) -> list[dict[str, str]]:
        return self._read_csv_path(self.root / filename)

    @staticmethod
    def _read_csv_path(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _objects_file(self, info: dict[str, Any]) -> Path:
        if self.objects_path is not None:
            return self.objects_path
        reference = Path(str(info.get("objects_file", "objects.csv")))
        return (reference if reference.is_absolute() else self.root / reference).resolve()

    def _objects_reference(self, info: dict[str, Any]) -> str:
        if self.objects_path is None:
            return str(info.get("objects_file", "objects.csv"))
        return Path(os.path.relpath(self.objects_path, self.root)).as_posix()

    def _read_yaml(self, filename: str) -> dict[str, Any]:
        path = self.root / filename
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{filename} must contain a mapping")
        return payload

    def _atomic_write(self, filename: str, content: str) -> None:
        self._atomic_write_bytes(filename, content.encode("utf-8"))

    def _atomic_write_bytes(self, filename: str, content: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root, prefix=f".{filename}.{os.getpid()}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.root / filename)
        finally:
            temporary.unlink(missing_ok=True)

    def _commit_documents(self, documents: dict[str, str]) -> None:
        originals = {
            name: (self.root / name).read_bytes() if (self.root / name).is_file() else None
            for name in documents
        }
        written: list[str] = []
        try:
            for name, content in documents.items():
                self._atomic_write(name, content)
                written.append(name)
        except OSError:
            for name in reversed(written):
                original = originals[name]
                if original is None:
                    (self.root / name).unlink(missing_ok=True)
                else:
                    self._atomic_write_bytes(name, original)
            raise

    @contextmanager
    def _write_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    @staticmethod
    def _unique_ids(
        records: list[dict[str, Any]], field: str, label: str, issues: list[dict[str, str]]
    ) -> set[str]:
        seen: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                issues.append(
                    TaskSetStore._issue("error", label, f"{label.title()} #{index + 1} is invalid")
                )
                continue
            value = str(record.get(field, "")).strip()
            if not value:
                issues.append(TaskSetStore._issue("error", label, f"{label.title()} ID is empty"))
            elif value in seen:
                issues.append(TaskSetStore._issue("error", value, f"Duplicate {label} ID"))
            seen.add(value)
        return seen

    @staticmethod
    def _issue(level: str, entity: str, message: str) -> dict[str, str]:
        return {"level": level, "entity": entity, "message": message}
