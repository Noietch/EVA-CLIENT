"""Persist the shared collection-object catalog and its photographed views."""

from __future__ import annotations

import csv
import io
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from tools.datasets.store import ConflictError, RecordNotFoundError

MEASUREMENT_FIELDS = ("length_cm", "width_cm", "height_cm", "mass_g")
ASSET_REFERENCE_FIELDS = ("image2assets_id", "eva_sim_id")
FIELDS = (
    "object_id",
    "object_name",
    "object_name_zh",
    "scan_status",
    "color",
    "photo_dir",
    *MEASUREMENT_FIELDS,
    "modeling_method",
    *ASSET_REFERENCE_FIELDS,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_PHOTOS_PER_UPLOAD = 16
MAX_PHOTO_BYTES = 16 * 1024 * 1024
CSV_PHOTO_COLUMNS = ("photos", "photo", "photo_files", "photo_filenames", "photo_paths")
COLOR_TRANSLATIONS = {
    "white": "白色",
    "black": "黑色",
    "red": "红色",
    "orange": "橙色",
    "yellow": "黄色",
    "green": "绿色",
    "blue": "蓝色",
    "purple": "紫色",
    "pink": "粉色",
    "brown": "棕色",
    "gray": "灰色",
    "grey": "灰色",
}
CSV_FIELD_ALIASES = {
    "物体编号": "object_id",
    "编号": "object_id",
    "英文名称": "object_name",
    "物体名称": "object_name",
    "中文名称": "object_name_zh",
    "颜色": "color",
    "长": "length_cm",
    "宽": "width_cm",
    "高": "height_cm",
    "质量": "mass_g",
    "照片": "photos",
    "照片文件": "photos",
}
OBJECT_CACHE_SECONDS = 2.0
PREVIEW_SPECS = {
    "thumb": (192, 68),
    "display": (1280, 78),
}


class ObjectCatalog:
    """Own one object table shared by every task-plan batch."""

    def __init__(self, root: str | Path, seed_objects: list[dict[str, Any]]):
        self.root = Path(root).expanduser().resolve()
        self.photo_root = self.root / "object_photos"
        self.preview_root = self.root / ".preview_cache"
        self.path = self.root / "objects.csv"
        self.lock = threading.RLock()
        self._records_cache: list[dict[str, Any]] | None = None
        self._records_cached_at = 0.0
        self.photo_root.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self._bootstrap(seed_objects)

    def records(self) -> list[dict[str, Any]]:
        with self.lock:
            now = time.monotonic()
            if (
                self._records_cache is not None
                and now - self._records_cached_at < OBJECT_CACHE_SECONDS
            ):
                return [dict(row, photos=list(row["photos"])) for row in self._records_cache]
            with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
            for row in rows:
                for key in (*MEASUREMENT_FIELDS, "modeling_method", *ASSET_REFERENCE_FIELDS):
                    row[key] = row.get(key) or ""
                directory = self.photo_root / row.get("photo_dir", "")
                row["photos"] = (
                    sorted(
                        path.name
                        for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    )
                    if directory.is_dir()
                    else []
                )
            self._records_cache = rows
            self._records_cached_at = now
            return [dict(row, photos=list(row["photos"])) for row in rows]

    def upsert(self, current_id: str | None, payload: Any) -> dict[str, Any]:
        if (
            current_id is None
            and isinstance(payload, dict)
            and not str(payload.get("object_id", "")).strip()
        ):
            payload = dict(payload)
            payload["object_id"] = self._next_object_id()
        record = self._normalize(payload)
        if current_id is not None and current_id != record["object_id"]:
            raise ValueError("object_id cannot be changed")
        with self.lock:
            rows = self.records()
            index = next(
                (i for i, row in enumerate(rows) if row["object_id"] == record["object_id"]), -1
            )
            if current_id is None and index >= 0:
                raise ConflictError(f"{record['object_id']} already exists")
            if current_id is not None and index < 0:
                raise RecordNotFoundError(current_id)
            documents = [{key: row.get(key, "") for key in FIELDS} for row in rows]
            if index < 0:
                documents.append(record)
            else:
                documents[index] = record
            self._write(documents)
        return next(row for row in self.records() if row["object_id"] == record["object_id"])

    def import_csv(self, content: bytes, uploads: list[tuple[str, bytes]]) -> dict[str, int]:
        if not content:
            raise ValueError("Choose a non-empty CSV file")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("CSV must be UTF-8 encoded") from error
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row")
        rows = list(reader)
        if not rows:
            raise ValueError("CSV must contain at least one object row")
        files = {Path(name).name: (name, data) for name, data in uploads if name}
        if len(files) != len(uploads):
            raise ValueError("Uploaded photo filenames must be unique")
        existing = self.records()
        existing_ids = {row["object_id"] for row in existing}
        seen_ids: set[str] = set()
        prepared: list[tuple[dict[str, str], list[tuple[str, bytes]]]] = []
        next_id = self._next_object_id(existing_ids)
        for row in rows:
            payload = {
                CSV_FIELD_ALIASES.get(str(key).strip(), str(key).strip()): ""
                if value is None
                else str(value)
                for key, value in row.items()
                if key
            }
            object_id = str(payload.get("object_id", "")).strip()
            if object_id and object_id in existing_ids:
                existing_row = next(item for item in existing if item["object_id"] == object_id)
                payload["photo_dir"] = existing_row.get("photo_dir", "")
            else:
                payload.pop("photo_dir", None)
            payload["scan_status"] = ""
            if not any(
                str(payload.get(key, "")).strip() for key in ("object_name", "object_name_zh")
            ):
                raise ValueError("Each CSV row needs object_name or object_name_zh")
            if not object_id:
                object_id = next_id
                next_id = self._next_object_id(existing_ids | seen_ids | {object_id}, next_id)
            payload["object_id"] = object_id
            record = self._normalize(payload)
            if record["object_id"] in seen_ids:
                raise ConflictError(f"{record['object_id']} appears more than once in CSV")
            seen_ids.add(record["object_id"])
            photo_value = next(
                (payload.get(key, "") for key in CSV_PHOTO_COLUMNS if payload.get(key)),
                "",
            )
            photo_names = [
                name.strip() for name in re.split(r"[;|\n]", str(photo_value)) if name.strip()
            ]
            photo_uploads = []
            for name in photo_names:
                file_entry = files.get(Path(name).name)
                if file_entry is None:
                    raise ValueError(f"CSV photo file not uploaded: {name}")
                photo_uploads.append(file_entry)
            self._validate_uploads(photo_uploads)
            prepared.append((record, photo_uploads))

        with self.lock:
            documents = [{key: row.get(key, "") for key in FIELDS} for row in existing]
            positions = {row["object_id"]: index for index, row in enumerate(documents)}
            for record, _ in prepared:
                index = positions.get(record["object_id"])
                if index is None:
                    positions[record["object_id"]] = len(documents)
                    documents.append(record)
                else:
                    documents[index] = record
            self._write(documents)
            for record, photo_uploads in prepared:
                if photo_uploads:
                    self.save_photos(record["object_id"], photo_uploads)
        return {
            "objects": len(prepared),
            "photos": sum(len(photo_uploads) for _, photo_uploads in prepared),
        }

    def delete(self, object_id: str) -> None:
        with self.lock:
            rows = self.records()
            if not any(row["object_id"] == object_id for row in rows):
                raise RecordNotFoundError(object_id)
            self._write(
                [
                    {key: row.get(key, "") for key in FIELDS}
                    for row in rows
                    if row["object_id"] != object_id
                ]
            )

    def save_photos(self, object_id: str, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
        asset = self._record(object_id)
        if not uploads or len(uploads) > MAX_PHOTOS_PER_UPLOAD:
            raise ValueError(f"Choose between 1 and {MAX_PHOTOS_PER_UPLOAD} photos")
        validated = self._validate_uploads(uploads)

        directory = self.photo_root / asset["photo_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        for filename, content in validated:
            target = directory / filename
            descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=".upload-")
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        with self.lock:
            self._records_cache = None
        return self._record(object_id)

    def _validate_uploads(self, uploads: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
        if len(uploads) > MAX_PHOTOS_PER_UPLOAD:
            raise ValueError(f"Choose at most {MAX_PHOTOS_PER_UPLOAD} photos per object")
        validated = []
        for original_name, content in uploads:
            suffix = Path(original_name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES or not content or len(content) > MAX_PHOTO_BYTES:
                raise ValueError("Photos must be non-empty JPG, PNG, or WebP files under 16 MiB")
            try:
                Image.open(io.BytesIO(content)).verify()
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid image: {original_name}") from error
            stem = re.sub(r"[^\w.-]+", "_", Path(original_name).stem, flags=re.UNICODE).strip("._")
            validated.append((f"{stem or 'photo'}{suffix}", content))
        return validated

    def _next_object_id(self, used: set[str] | None = None, start: str | None = None) -> str:
        used = used or {row["object_id"] for row in self.records()}
        match = re.search(r"(\d+)$", start or "")
        number = int(match.group(1)) if match else 1
        while f"AST-{number:04d}" in used:
            number += 1
        return f"AST-{number:04d}"

    def photo_path(self, object_id: str, filename: str) -> Path:
        asset = self._record(object_id)
        if Path(filename).name != filename:
            raise FileNotFoundError(filename)
        directory = (self.photo_root / asset["photo_dir"]).resolve()
        path = (directory / filename).resolve()
        if (
            directory not in path.parents
            or not path.is_file()
            or path.suffix.lower() not in IMAGE_SUFFIXES
        ):
            raise FileNotFoundError(filename)
        return path

    def preview_path(self, object_id: str, filename: str, variant: str) -> Path:
        """Return a cached WebP derived from the immutable original photo."""
        max_size, quality = PREVIEW_SPECS[variant]
        source = self.photo_path(object_id, filename)
        source_stat = source.stat()
        cache_dir = self.preview_root / variant / object_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{filename}.{source_stat.st_mtime_ns}.{source_stat.st_size}.webp"
        target = cache_dir / cache_name
        if target.is_file():
            return target

        with Image.open(source) as original:
            preview = ImageOps.exif_transpose(original)
            preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if preview.mode not in {"RGB", "RGBA"}:
                preview = preview.convert("RGB")
            descriptor, temporary_name = tempfile.mkstemp(
                dir=cache_dir, prefix=".preview-", suffix=".webp"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    preview.save(handle, format="WEBP", quality=quality, method=4)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    def _record(self, object_id: str) -> dict[str, Any]:
        record = next((row for row in self.records() if row["object_id"] == object_id), None)
        if record is None:
            raise RecordNotFoundError(object_id)
        return record

    def _bootstrap(self, seed_objects: list[dict[str, Any]]) -> None:
        rows: dict[str, dict[str, str]] = {}
        for payload in seed_objects:
            record = self._normalize(payload)
            rows.setdefault(record["object_id"], record)
        directories = sorted(path.name for path in self.photo_root.iterdir() if path.is_dir())
        unused = set(directories)
        for row in rows.values():
            keys = {self._photo_key(row["object_name"]), self._photo_key(row["object_name_zh"])}
            match = next((name for name in directories if self._photo_key(name) in keys), "")
            if match:
                row["photo_dir"] = match
            unused.discard(match)
        next_id = 1
        for name in sorted(unused):
            while f"ASSET-{next_id:03d}" in rows:
                next_id += 1
            object_id = f"ASSET-{next_id:03d}"
            rows[object_id] = {
                "object_id": object_id,
                "object_name": "",
                "object_name_zh": name,
                "color": "",
                "photo_dir": name,
            }
            next_id += 1
        self._write(list(rows.values()))

    @staticmethod
    def _normalize(payload: Any) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise ValueError("asset must be an object")
        object_id = str(payload.get("object_id", "")).strip()
        if not ID_PATTERN.fullmatch(object_id):
            raise ValueError("object_id must use letters, numbers, '.', '_' or '-'")
        name_en = str(payload.get("object_name", "")).strip()
        name_zh = str(payload.get("object_name_zh", "")).strip()
        if not name_en and not name_zh:
            raise ValueError("At least one object name is required")
        supplied_photo_dir = str(payload.get("photo_dir", "")).strip()
        photo_dir = supplied_photo_dir or re.sub(
            r"[/\\]+", "_", name_zh or name_en or object_id
        ).strip(" .")
        if Path(photo_dir).name != photo_dir or photo_dir in {".", ".."}:
            raise ValueError("photo_dir must be one directory name")
        measurements = {}
        for key in MEASUREMENT_FIELDS:
            value = payload.get(key)
            text = "" if value is None else str(value).strip()
            if text:
                try:
                    number = float(text)
                except ValueError:
                    raise ValueError(f"{key} 必须是大于 0 的有限数值，或留空") from None
                if not math.isfinite(number) or number <= 0:
                    raise ValueError(f"{key} 必须是大于 0 的有限数值，或留空")
            measurements[key] = text
        method = str(payload.get("modeling_method") or "").strip()
        if method not in {"", "A", "B", "C", "D"}:
            raise ValueError("建模方法必须为 A、B、C、D，或留空")
        asset_references = {
            key: str(payload.get(key) or "").strip() for key in ASSET_REFERENCE_FIELDS
        }
        return {
            **measurements,
            **asset_references,
            "modeling_method": method,
            "object_id": object_id,
            "object_name": name_en,
            "object_name_zh": name_zh,
            "scan_status": str(payload.get("scan_status", "")).strip(),
            "color": COLOR_TRANSLATIONS.get(
                str(payload.get("color", "")).strip().lower(),
                str(payload.get("color", "")).strip(),
            ),
            "photo_dir": photo_dir,
        }

    def _write(self, rows: list[dict[str, Any]]) -> None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
        descriptor, temporary_name = tempfile.mkstemp(dir=self.root, prefix=".objects-")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write("\ufeff" + stream.getvalue())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._records_cache = None
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _photo_key(value: str) -> str:
        text = str(value).lower()
        for source, target in (
            ("绿色", "绿"),
            ("蓝色", "蓝"),
            ("黄色", "黄"),
            ("灰色", "灰"),
            ("黑色", "黑"),
            ("白色", "白"),
            ("红色", "红"),
            ("橙色", "橙"),
        ):
            text = text.replace(source, target)
        return "".join(sorted(char for char in text if char.isalnum()))
