from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .base import (
    LANGUAGE_ANNOTATION_KEY,
    BaseDataset,
    EpisodeData,
    InferKeysResult,
    VideoRef,
    build_episode_metadata,
    build_infer_keys,
    coerce_column,
    pack_value,
    unpack_value,
)

_DEFAULT_V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_FILES_PER_CHUNK = 1000
_DATA_FILE_SIZE_BYTES = 100 * 1024 * 1024
_VIDEO_FILE_SIZE_BYTES = 500 * 1024 * 1024
_FILE_PATTERN = re.compile(r"file-(\d+)\.")
_CHUNK_PATTERN = re.compile(r"chunk-(\d+)$")
_EPISODE_METADATA_KEY = b"eva.episode"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_path(root: Path, template: str, values: dict[str, Any]) -> Path:
    return root / template.format(**values)


def _episode_file_stem(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def _row_value(row: dict[str, Any], *candidates: str) -> Any:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _first_task_from_row(row: dict[str, Any]) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    task = row.get("task")
    if isinstance(task, str):
        return task
    return ""


def _episode_rows_from_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        rows[int(row["episode_index"])] = row
    return rows


def _tasks_from_jsonl(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    if not path.exists():
        return tasks
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        tasks[int(row["task_index"])] = str(row.get("task", ""))
    return tasks


def _tasks_from_parquet(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    table = pq.read_table(path)
    payload = table.to_pydict()
    if "task_index" not in payload:
        return {}
    label_column = None
    for candidate in ("task", "__index_level_0__"):
        if candidate in payload:
            label_column = payload[candidate]
            break
    if label_column is None:
        return {}
    return {
        int(task_index): str(task)
        for task_index, task in zip(payload["task_index"], label_column, strict=True)
    }


def _coerce_table_columns(
    table: Any, image_keys: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    columns: dict[str, Any] = {}
    images: dict[str, Any] = {}
    payload = table.to_pydict()
    for key, values in payload.items():
        if key in image_keys:
            images[key] = coerce_column(values)
            continue
        columns[key] = coerce_column(values)
    return columns, images


def _arrow_table(columns: dict[str, Any]) -> pa.Table:
    return pa.table(
        {
            key: value.tolist() if isinstance(value, np.ndarray) and value.ndim > 1 else value
            for key, value in columns.items()
        }
    )


class LeRobotV21Dataset(BaseDataset):
    format = "lerobot_v21"

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        video_keys: Iterable[str] = (),
    ) -> None:
        super().__init__(path, fps=fps, video_keys=video_keys)
        self._meta_dir = self.path / "meta"
        info_path = self._meta_dir / "info.json"
        self._info = _read_json(info_path) if info_path.exists() else {}

    @cached_property
    def _tasks_by_index(self) -> dict[int, str]:
        return _tasks_from_jsonl(self._meta_dir / "tasks.jsonl")

    @cached_property
    def _episode_rows(self) -> dict[int, dict[str, Any]]:
        return _episode_rows_from_jsonl(self._meta_dir / "episodes.jsonl")

    @property
    def _features(self) -> dict[str, Any]:
        return self._info.get("features", {})

    def count_episodes(self) -> int:
        physical = len(list(self.path.glob("data/chunk-*/episode_*.parquet")))
        return physical or int(self._info.get("total_episodes", 0))

    def episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for parquet_path in sorted(self.path.glob("data/chunk-*/episode_*.parquet")):
            episode_index = int(parquet_path.stem.removeprefix("episode_"))
            parquet_file = pq.ParquetFile(parquet_path)
            metadata = parquet_file.schema_arrow.metadata or {}
            stored = metadata.get(_EPISODE_METADATA_KEY)
            row = dict(unpack_value(stored)) if stored is not None else {}
            row.setdefault("episode_index", episode_index)
            row.setdefault("length", parquet_file.metadata.num_rows)
            catalog_row = self._episode_rows.get(episode_index, {})
            if not row.get("tasks") and not catalog_row:
                columns = (
                    ["task_index"]
                    if "task_index" in parquet_file.schema_arrow.names
                    else []
                )
                table = parquet_file.read(columns=columns)
                task = self._resolve_task(episode_index, table)
                row["tasks"] = [task] if task else []
            rows.append({**row, **catalog_row})
        return rows

    def infer_keys(self) -> InferKeysResult:
        image_keys = [key for key, spec in self._features.items() if _is_image_feature(key, spec)]
        return build_infer_keys(image_keys=image_keys, column_keys=list(self._features))

    def load(self, episode_index: int) -> EpisodeData:
        parquet_path = self._episode_parquet_path(episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
        table = pq.read_table(parquet_path)
        image_keys = {
            key
            for key, spec in self._features.items()
            if _is_image_feature(key, spec) and not _is_video_feature(spec)
        }
        columns, images = _coerce_table_columns(table, image_keys=image_keys)
        return EpisodeData(
            columns=columns,
            task=self._resolve_task(episode_index, table),
            fps=_read_fps(self._info),
            videos=self._episode_videos(episode_index),
            images=images,
        )

    def write(
        self,
        columns: dict[str, Any],
        row: dict[str, Any],
        videos: Mapping[str, Iterable[np.ndarray]],
    ) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        chunk = episode_index // int(self._info.get("chunks_size", 1000))
        path = (
            self.path
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table = _arrow_table(columns)
        metadata = dict(table.schema.metadata or {})
        metadata[_EPISODE_METADATA_KEY] = pack_value(
            build_episode_metadata(row, self.require_fps())
        )
        pq.write_table(table.replace_schema_metadata(metadata), path)
        for key, frames in videos.items():
            video_path = (
                self.path
                / "videos"
                / f"chunk-{chunk:03d}"
                / key
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(video_path),
                fps=self.require_fps(),
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=[
                    "-preset",
                    "ultrafast",
                    "-g",
                    str(max(1, round(self.require_fps()))),
                    "-movflags",
                    "+faststart",
                ],
            )
            try:
                for frame in frames:
                    writer.append_data(np.ascontiguousarray(frame))
            finally:
                writer.close()
        return dict(row)

    def episode_videos(self, episode_index: int) -> dict[str, VideoRef]:
        return self._episode_videos(episode_index)

    def _resolve_task(self, episode_index: int, table: Any) -> str:
        task = _first_task_from_row(self._episode_rows.get(episode_index, {}))
        if task:
            return task
        if "task_index" not in table.column_names:
            return ""
        task_indices = table.column("task_index").to_pylist()
        return self._tasks_by_index.get(int(task_indices[0]), "") if task_indices else ""

    def _episode_parquet_path(self, episode_index: int) -> Path:
        chunk_size = int(self._info.get("chunks_size", 1000))
        template = self._info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        return self.path / template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            episode_file=self._episode_file_stem(episode_index),
        )

    def _episode_videos(self, episode_index: int) -> dict[str, VideoRef]:
        video_keys = {key: key for key, spec in self._features.items() if _is_video_feature(spec)}
        chunk_size = int(self._info.get("chunks_size", 1000))
        template = self._info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        videos = {}
        for key in video_keys:
            path = self.path / template.format(
                episode_chunk=episode_index // chunk_size,
                video_key=key,
                episode_index=episode_index,
                episode_file=self._episode_file_stem(episode_index),
            )
            if path.exists():
                videos[key] = VideoRef(path=path)
        if videos:
            return videos
        video_root = self.path / "videos" / f"chunk-{episode_index // chunk_size:03d}"
        episode_file = self._episode_file_stem(episode_index)
        return {
            path.parent.name: VideoRef(path=path)
            for path in sorted(video_root.glob(f"*/{episode_file}.mp4"))
        }

    def _episode_file_stem(self, episode_index: int) -> str:
        stem = self._episode_rows.get(episode_index, {}).get("file_stem")
        return str(stem) if stem else _episode_file_stem(episode_index)

    def patch_episode(self, episode_index: int, values: Mapping[str, Any]) -> bool:
        rows = dict(self._episode_rows)
        if episode_index not in rows:
            rows = {
                int(row["episode_index"]): row
                for row in self.episode_rows()
                if "episode_index" in row
            }
        if episode_index not in rows:
            return False
        rows[episode_index] = {**rows[episode_index], **values}
        path = self._meta_dir / "episodes.jsonl"
        with path.open("w") as stream:
            for index in sorted(rows):
                stream.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
        vars(self).pop("_episode_rows", None)
        return True

    def read_annotation(self, episode_index: int) -> str:
        row = self._episode_rows.get(episode_index, {})
        return str(row.get(LANGUAGE_ANNOTATION_KEY, "") or "")

    def annotate_language(self, episode_index: int, annotation: str) -> bool:
        if not self.patch_episode(
            episode_index,
            {LANGUAGE_ANNOTATION_KEY: annotation},
        ):
            return False
        self._write_annotation_parquet_column(episode_index, annotation)
        self._declare_annotation_feature()
        return True

    def _write_annotation_parquet_column(self, episode_index: int, annotation: str) -> None:
        parquet_path = self._episode_parquet_path(episode_index)
        if not parquet_path.exists():
            return
        table = pq.read_table(parquet_path)
        values = pa.array([annotation] * table.num_rows, type=pa.string())
        if LANGUAGE_ANNOTATION_KEY in table.column_names:
            table = table.set_column(
                table.column_names.index(LANGUAGE_ANNOTATION_KEY),
                LANGUAGE_ANNOTATION_KEY,
                values,
            )
        else:
            table = table.append_column(LANGUAGE_ANNOTATION_KEY, values)
        pq.write_table(table, parquet_path)

    def _declare_annotation_feature(self) -> None:
        features = self._info.setdefault("features", {})
        if LANGUAGE_ANNOTATION_KEY in features:
            return
        features[LANGUAGE_ANNOTATION_KEY] = {"dtype": "string", "shape": [1], "names": None}
        (self._meta_dir / "info.json").write_text(
            json.dumps(self._info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class LeRobotV3Dataset(BaseDataset):
    format = "lerobot_v3"

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        video_keys: Iterable[str] = (),
    ) -> None:
        super().__init__(path, fps=fps, video_keys=video_keys)
        self._meta_dir = self.path / "meta"
        info_path = self._meta_dir / "info.json"
        self._info = _read_json(info_path) if info_path.exists() else {}
        version = str(self._info.get("codebase_version", ""))
        if fps is not None and version and not version.startswith("v3"):
            raise ValueError(f"Cannot append LeRobot v3 shards to {version}: {self.path}")
        self._shard = self._next_shard()
        self._data_writer: pq.ParquetWriter | None = None
        self._episode_writer: pq.ParquetWriter | None = None
        self._video_writers: dict[str, Any] = {}
        self._video_frames: dict[str, int] = {}
        self._source_schema: pa.Schema | None = None

    @cached_property
    def _tasks_by_index(self) -> dict[int, str]:
        parquet_tasks = _tasks_from_parquet(self._meta_dir / "tasks.parquet")
        if parquet_tasks:
            return parquet_tasks
        return _tasks_from_jsonl(self._meta_dir / "tasks.jsonl")

    @property
    def _features(self) -> dict[str, Any]:
        return self._info.get("features", {})

    def count_episodes(self) -> int:
        physical = sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in (self._meta_dir / "episodes").glob("chunk-*/file-*.parquet")
        )
        return physical or int(self._info.get("total_episodes", 0))

    def episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted((self._meta_dir / "episodes").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path)
            payload = table.to_pylist()
            rows.extend(payload)
        return rows

    def infer_keys(self) -> InferKeysResult:
        image_keys = [key for key, spec in self._features.items() if _is_image_feature(key, spec)]
        return build_infer_keys(image_keys=image_keys, column_keys=list(self._features))

    def load(self, episode_index: int) -> EpisodeData:
        row = self._episode_row(episode_index)
        table = self._read_episode_table(episode_index, row)
        image_keys = {
            key
            for key, spec in self._features.items()
            if _is_image_feature(key, spec) and not _is_video_feature(spec)
        }
        columns, images = _coerce_table_columns(table, image_keys=image_keys)
        return EpisodeData(
            columns=columns,
            task=self._resolve_task(row, table),
            fps=_read_fps(self._info),
            videos=self._episode_videos(row),
            images=images,
        )

    def write(
        self,
        columns: dict[str, Any],
        row: dict[str, Any],
        videos: Mapping[str, Iterable[np.ndarray]],
    ) -> dict[str, Any]:
        try:
            return self._write(columns, row, videos)
        except BaseException:
            if self._has_open_shard():
                self._rotate()
            raise

    def _write(
        self,
        columns: dict[str, Any],
        row: dict[str, Any],
        videos: Mapping[str, Iterable[np.ndarray]],
    ) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        table = _arrow_table(columns)
        if table.num_rows != length:
            raise ValueError(
                f"Episode {episode_index} has {table.num_rows} rows, expected {length}"
            )
        if self._source_schema is not None and not table.schema.equals(self._source_schema):
            self._rotate()
        chunk_index, file_index = divmod(self._shard, _FILES_PER_CHUNK)
        self._open_data_writer(table.schema, chunk_index, file_index)
        assert self._data_writer is not None
        self._data_writer.write_table(table)

        metadata: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": [str(task) for task in row.get("tasks", [])],
            "length": length,
            "data/chunk_index": chunk_index,
            "data/file_index": file_index,
            "dataset_from_index": int(columns["index"][0]),
            "dataset_to_index": int(columns["index"][-1]) + 1,
        }
        fps = self.require_fps()
        for key in self.video_keys:
            frames = videos.get(key)
            if frames is None:
                metadata.update(self._empty_video_metadata(key))
                continue
            start = self._video_frames.get(key, 0)
            writer = self._video_writer(key, chunk_index, file_index)
            written = 0
            for frame in frames:
                writer.append_data(np.ascontiguousarray(frame))
                written += 1
            self._video_frames[key] = start + written
            metadata.update(
                {
                    f"videos/{key}/chunk_index": chunk_index,
                    f"videos/{key}/file_index": file_index,
                    f"videos/{key}/from_timestamp": start / fps,
                    f"videos/{key}/to_timestamp": (start + written) / fps,
                }
            )
        self._write_episode_metadata(metadata, chunk_index, file_index)
        if self._shard_is_full(chunk_index, file_index):
            self._rotate()
        return metadata

    def episode_videos(self, episode_index: int) -> dict[str, VideoRef]:
        return self._episode_videos(self._episode_row(episode_index))

    def _episode_row(self, episode_index: int) -> dict[str, Any]:
        for path in sorted((self._meta_dir / "episodes").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, filters=[("episode_index", "=", int(episode_index))])
            if table.num_rows:
                return {key: values[0] for key, values in table.to_pydict().items()}
        raise IndexError(f"Episode index {episode_index} not found in {self.path}")

    def _resolve_task(self, row: dict[str, Any], table: Any) -> str:
        task = _first_task_from_row(row)
        if task:
            return task
        if "task_index" not in table.column_names:
            return ""
        task_indices = table.column("task_index").to_pylist()
        if not task_indices:
            return ""
        return self._tasks_by_index.get(int(task_indices[0]), "")

    def _read_episode_table(self, episode_index: int, row: dict[str, Any]) -> Any:
        data_path = self._data_path(row)
        table = pq.read_table(data_path, filters=[("episode_index", "=", int(episode_index))])
        if table.num_rows:
            return table
        table = pq.read_table(data_path)
        start = _int_or_default(
            _row_value(row, "dataset_from_index", "from_index", "data_from_index"), 0
        )
        stop = _int_or_default(
            _row_value(row, "dataset_to_index", "to_index", "data_to_index"), table.num_rows
        )
        return table.slice(start, max(0, stop - start))

    def _data_path(self, row: dict[str, Any]) -> Path:
        template = self._info.get("data_path", _DEFAULT_V3_DATA_PATH)
        values = {
            "chunk_index": _int_or_default(row.get("data/chunk_index"), 0),
            "file_index": _int_or_default(row.get("data/file_index"), 0),
            "episode_index": int(row["episode_index"]),
            "episode_file": _episode_file_stem(int(row["episode_index"])),
        }
        return _format_path(self.path, template, values)

    def _episode_videos(self, row: dict[str, Any]) -> dict[str, VideoRef]:
        template = self._info.get("video_path", _DEFAULT_V3_VIDEO_PATH)
        videos: dict[str, VideoRef] = {}
        for key, spec in self._features.items():
            if not _is_video_feature(spec):
                continue
            chunk_value = row.get(f"videos/{key}/chunk_index")
            file_value = row.get(f"videos/{key}/file_index")
            video_path = _format_path(
                self.path,
                template,
                {
                    "video_key": key,
                    "episode_index": int(row["episode_index"]),
                    "episode_file": _episode_file_stem(int(row["episode_index"])),
                    "chunk_index": _int_or_default(chunk_value, 0),
                    "file_index": _int_or_default(file_value, 0),
                },
            )
            if not video_path.exists():
                continue
            start_time = _float_or_default(
                row.get(f"videos/{key}/from_timestamp"),
                0.0,
            )
            end_time = _float_or_none(row.get(f"videos/{key}/to_timestamp"))
            videos[key] = VideoRef(path=video_path, start_time=start_time, end_time=end_time)
        return videos

    def finalize(self) -> None:
        self._close_current_shard()

    def seal_current_shard(self) -> None:
        if self._has_open_shard():
            self._rotate()

    def _open_data_writer(self, schema: pa.Schema, chunk_index: int, file_index: int) -> None:
        if self._data_writer is not None:
            return
        path = self._shard_data_path(chunk_index, file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._source_schema = schema
        self._data_writer = pq.ParquetWriter(
            path,
            schema,
            compression="snappy",
            use_dictionary=True,
        )

    def _write_episode_metadata(
        self,
        metadata: dict[str, Any],
        chunk_index: int,
        file_index: int,
    ) -> None:
        schema = self._episode_schema()
        writer = self._episode_writer
        if writer is None:
            path = self._episode_path(chunk_index, file_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(
                path,
                schema,
                compression="snappy",
                use_dictionary=True,
            )
            self._episode_writer = writer
        writer.write_table(pa.Table.from_pylist([metadata], schema=schema))

    def _episode_schema(self) -> pa.Schema:
        fields = [
            pa.field("episode_index", pa.int64()),
            pa.field("tasks", pa.list_(pa.string())),
            pa.field("length", pa.int64()),
            pa.field("data/chunk_index", pa.int64()),
            pa.field("data/file_index", pa.int64()),
            pa.field("dataset_from_index", pa.int64()),
            pa.field("dataset_to_index", pa.int64()),
        ]
        for key in self.video_keys:
            fields.extend(
                [
                    pa.field(f"videos/{key}/chunk_index", pa.int64()),
                    pa.field(f"videos/{key}/file_index", pa.int64()),
                    pa.field(f"videos/{key}/from_timestamp", pa.float64()),
                    pa.field(f"videos/{key}/to_timestamp", pa.float64()),
                ]
            )
        return pa.schema(fields)

    def _empty_video_metadata(self, key: str) -> dict[str, None]:
        return {
            f"videos/{key}/chunk_index": None,
            f"videos/{key}/file_index": None,
            f"videos/{key}/from_timestamp": None,
            f"videos/{key}/to_timestamp": None,
        }

    def _video_writer(self, key: str, chunk_index: int, file_index: int) -> Any:
        writer = self._video_writers.get(key)
        if writer is not None:
            return writer
        path = self._shard_video_path(key, chunk_index, file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(path),
            fps=self.require_fps(),
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=[
                "-preset",
                "ultrafast",
                "-g",
                str(max(1, round(self.require_fps()))),
                "-movflags",
                "+faststart",
            ],
        )
        self._video_writers[key] = writer
        return writer

    def _shard_is_full(self, chunk_index: int, file_index: int) -> bool:
        data_full = (
            self._shard_data_path(chunk_index, file_index).stat().st_size >= _DATA_FILE_SIZE_BYTES
        )
        return data_full or any(
            path.exists() and path.stat().st_size >= _VIDEO_FILE_SIZE_BYTES
            for path in (
                self._shard_video_path(key, chunk_index, file_index) for key in self._video_writers
            )
        )

    def _rotate(self) -> None:
        self._close_current_shard()
        self._shard += 1

    def _has_open_shard(self) -> bool:
        return (
            self._data_writer is not None
            or self._episode_writer is not None
            or bool(self._video_writers)
        )

    def _close_current_shard(self) -> None:
        for writer in self._video_writers.values():
            writer.close()
        self._video_writers = {}
        self._video_frames = {}
        if self._episode_writer is not None:
            self._episode_writer.close()
            self._episode_writer = None
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        self._source_schema = None

    def _next_shard(self) -> int:
        latest = -1
        for path in (self.path / "data").glob("chunk-*/file-*.parquet"):
            chunk = _CHUNK_PATTERN.fullmatch(path.parent.name)
            file = _FILE_PATTERN.match(path.name)
            if chunk and file:
                latest = max(
                    latest,
                    int(chunk.group(1)) * _FILES_PER_CHUNK + int(file.group(1)),
                )
        return latest + 1

    def _shard_data_path(self, chunk_index: int, file_index: int) -> Path:
        return self.path / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"

    def _episode_path(self, chunk_index: int, file_index: int) -> Path:
        return (
            self.path
            / "meta"
            / "episodes"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.parquet"
        )

    def _shard_video_path(self, key: str, chunk_index: int, file_index: int) -> Path:
        return (
            self.path / "videos" / key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        )


def _is_video_feature(spec: Any) -> bool:
    return isinstance(spec, dict) and spec.get("dtype") == "video"


def _is_image_feature(key: str, spec: Any) -> bool:
    if isinstance(spec, dict) and spec.get("dtype") in {"video", "image"}:
        return True
    return "image" in key.lower()


def _read_fps(info: dict[str, Any]) -> float | None:
    fps = info.get("fps")
    return float(fps) if fps is not None else None


def _int_or_default(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _float_or_default(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = ["LeRobotV21Dataset", "LeRobotV3Dataset"]
