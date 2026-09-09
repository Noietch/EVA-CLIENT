from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.recorder.video_encoding import dataset_h264_ffmpeg_params

from .source import LeRobotV21Source, SourceEpisode, video_frames, write_common_metadata

_DATA_FILE_LIMIT = 100 * 1024 * 1024


class _LeRobotV3Writer:
    def __init__(self, output_dir: Path, source: LeRobotV21Source) -> None:
        self.output_dir = output_dir
        self.source = source
        self.shard_index = 0
        self.shard_bytes = 0
        self.data_writer: pq.ParquetWriter | None = None
        self.data_schema: pa.Schema | None = None
        self.video_writers: dict[str, Any] = {}
        self.video_frames: dict[str, int] = {}
        self.episode_rows: list[dict[str, Any]] = []

    def write(self, episode: SourceEpisode) -> None:
        episode_index = int(episode.row["episode_index"])
        frame_count = episode.table.num_rows
        if frame_count == 0:
            raise ValueError(f"LeRobot v3 episode {episode_index} has no frames")
        shard_full = self.shard_bytes + episode.table.nbytes > _DATA_FILE_LIMIT
        schema_changed = self.data_schema is not None and not episode.table.schema.equals(
            self.data_schema
        )
        if self.data_writer is not None and (shard_full or schema_changed):
            self._close_shard()
            self.shard_index += 1
        data_path = self._data_path()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if self.data_writer is None:
            self.data_schema = episode.table.schema
            self.data_writer = pq.ParquetWriter(
                data_path,
                episode.table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        self.data_writer.write_table(episode.table)
        self.shard_bytes += episode.table.nbytes

        row = dict(episode.row)
        row.update(
            {
                "data/chunk_index": 0,
                "data/file_index": self.shard_index,
                "dataset_from_index": int(episode.table["index"][0].as_py()),
                "dataset_to_index": int(episode.table["index"][-1].as_py()) + 1,
            }
        )
        for key, video_path in episode.videos.items():
            start = self.video_frames.get(key, 0)
            writer = self._video_writer(key)
            written = 0
            for frame in video_frames(video_path):
                writer.append_data(np.ascontiguousarray(frame))
                written += 1
            if written != frame_count:
                raise ValueError(
                    f"LeRobot v3 episode {episode_index} video {key!r} has {written} frames; "
                    f"expected {frame_count}"
                )
            self.video_frames[key] = start + written
            row.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": self.shard_index,
                    f"videos/{key}/from_timestamp": start / self.source.fps,
                    f"videos/{key}/to_timestamp": (start + written) / self.source.fps,
                }
            )
        self.episode_rows.append(row)

    def close(self) -> None:
        self._close_shard()
        meta_dir = self.output_dir / "meta"
        if self.episode_rows:
            episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
            episodes_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(self.episode_rows), episodes_path)
        tasks = _read_jsonl(meta_dir / "tasks.jsonl")
        if tasks:
            pq.write_table(pa.Table.from_pylist(tasks), meta_dir / "tasks.parquet")

    def _close_shard(self) -> None:
        for writer in self.video_writers.values():
            writer.close()
        self.video_writers = {}
        self.video_frames = {}
        if self.data_writer is not None:
            self.data_writer.close()
            self.data_writer = None
            self.data_schema = None
        self.shard_bytes = 0

    def _video_writer(self, key: str) -> Any:
        writer = self.video_writers.get(key)
        if writer is not None:
            return writer
        path = self.output_dir / "videos" / key / "chunk-000" / f"file-{self.shard_index:03d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(path),
            fps=self.source.fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=dataset_h264_ffmpeg_params(threads=None),
        )
        self.video_writers[key] = writer
        return writer

    def _data_path(self) -> Path:
        return self.output_dir / "data" / "chunk-000" / f"file-{self.shard_index:03d}.parquet"


def export_lerobot_v3(
    source_dir: Path,
    output_dir: Path,
    progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> None:
    source = LeRobotV21Source(source_dir)
    write_common_metadata(source, output_dir, "lerobot_v3")
    writer = _LeRobotV3Writer(output_dir, source)
    try:
        for completed, episode in enumerate(source.episodes(), start=1):
            writer.write(episode)
            if progress_callback is not None:
                progress_callback(completed, episode.row)
    finally:
        writer.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


__all__ = ["export_lerobot_v3"]
