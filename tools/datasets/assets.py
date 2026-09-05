"""Persist the shared collection-object catalog and its photographed views."""

from __future__ import annotations

import csv
import io
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from tools.datasets.store import ConflictError, RecordNotFoundError

FIELDS = ("object_id", "object_name", "object_name_zh", "scan_status", "color", "photo_dir")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_PHOTOS_PER_UPLOAD = 16
MAX_PHOTO_BYTES = 16 * 1024 * 1024
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
        validated = []
        for original_name, content in uploads:
            suffix = Path(original_name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES or not content or len(content) > MAX_PHOTO_BYTES:
                raise ValueError("Photos must be non-empty JPG, PNG, or WebP files under 16 MiB")
            try:
                Image.open(io.BytesIO(content)).verify()
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid image: {original_name}") from error

            # Keep a readable Unicode filename while blocking path traversal
            stem = re.sub(r"[^\w.-]+", "_", Path(original_name).stem, flags=re.UNICODE).strip("._")
            validated.append((f"{stem or 'photo'}{suffix}", content))

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
        return {
            "object_id": object_id,
            "object_name": name_en,
            "object_name_zh": name_zh,
            "scan_status": str(payload.get("scan_status", "")).strip(),
            "color": str(payload.get("color", "")).strip(),
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
