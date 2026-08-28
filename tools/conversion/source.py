from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SourceEpisode:
    row: dict[str, Any]
    table: pa.Table
    videos: dict[str, Path]


class LeRobotV21Source:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.meta_dir = self.root / "meta"
        self.info = json.loads((self.meta_dir / "info.json").read_text())
        self.episode_rows = _read_jsonl(self.meta_dir / "episodes.jsonl")
        self.fps = float(self.info["fps"])
        self.chunk_size = int(self.info.get("chunks_size", 1000))
        self.data_path = str(
            self.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
        )
        self.video_path = str(
            self.info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        self.video_keys = tuple(
            key
            for key, feature in self.info.get("features", {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        )

    def episodes(self) -> Iterator[SourceEpisode]:
        for row in self.episode_rows:
            values = self._path_values(row)
            parquet_path = self.root / self.data_path.format(**values)
            if not parquet_path.is_file():
                raise FileNotFoundError(f"episode parquet not found: {parquet_path}")
            videos = {
                key: self.root / self.video_path.format(**values, video_key=key)
                for key in tuple(row.get("video_keys") or self.video_keys)
            }
            missing = [str(path) for path in videos.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"episode videos not found: {', '.join(missing)}")
            yield SourceEpisode(row=dict(row), table=pq.read_table(parquet_path), videos=videos)

    def inline_images(self, table: pa.Table) -> dict[str, np.ndarray]:
        image_keys = {
            key
            for key, feature in self.info.get("features", {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "image"
        }
        return {
            key: np.asarray(table[key].to_pylist())
            for key in image_keys
            if key in table.column_names
        }

    def columns_without_images(self, table: pa.Table) -> dict[str, Any]:
        images = set(self.inline_images(table))
        return {
            key: _coerce_column(values)
            for key, values in table.to_pydict().items()
            if key not in images
        }

    def _path_values(self, row: dict[str, Any]) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        return {
            "episode_chunk": episode_index // self.chunk_size,
            "episode_index": episode_index,
            "episode_file": str(row.get("file_stem") or f"episode_{episode_index:06d}"),
        }


def video_frames(path: Path) -> Iterator[np.ndarray]:
    reader = cast(Any, imageio.get_reader(str(path)))
    try:
        for frame in reader:
            yield np.asarray(frame)
    finally:
        reader.close()


def write_common_metadata(source: LeRobotV21Source, output: Path, data_format: str) -> None:
    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = dict(source.info)
    info["dataset_format"] = data_format
    info["codebase_version"] = {
        "lerobot_v3": "v3.0",
        "hdf5": "hdf5",
        "mcap": "mcap",
    }[data_format]
    if data_format == "lerobot_v3":
        info.update(
            data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            data_files_size_in_mb=100,
            video_files_size_in_mb=500,
        )
    else:
        suffix = "hdf5" if data_format == "hdf5" else "mcap"
        info.update(
            data_path=f"data/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.{suffix}",
            embedded_images=True,
            total_videos=0,
        )
        info.pop("video_path", None)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    _copy_optional(source.meta_dir / "stats.json", meta_dir / "stats.json")
    _copy_optional(source.meta_dir / "tasks.jsonl", meta_dir / "tasks.jsonl")
    _copy_optional(source.meta_dir / "episodes.jsonl", meta_dir / "episodes.jsonl")
    marker = json.loads((source.meta_dir / "quality_split.json").read_text())
    marker["dataset_format"] = data_format
    (meta_dir / "quality_split.json").write_text(json.dumps(marker, indent=2) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _copy_optional(source: Path, target: Path) -> None:
    if source.is_file():
        shutil.copy2(source, target)


def _coerce_column(values: list[Any]) -> np.ndarray | list[Any]:
    if not values:
        return np.asarray(values)
    array = np.asarray(values)
    if array.dtype != object:
        return array
    try:
        return np.stack([np.asarray(value) for value in values])
    except ValueError:
        return values


__all__ = [
    "LeRobotV21Source",
    "SourceEpisode",
    "video_frames",
    "write_common_metadata",
]
