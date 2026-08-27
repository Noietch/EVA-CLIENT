"""Split a LeRobot dataset into training-safe and quarantined subsets."""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclasses.dataclass(frozen=True)
class QualitySplitSummary:
    source_dir: str
    accepted_dir: str
    rejected_dir: str
    source_episodes: int
    accepted_episodes: int
    rejected_episodes: int
    accepted_frames: int
    rejected_frames: int
    rejected_source_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class QualityExportProgress:
    episodes_completed: int
    episodes_total: int
    subset: str
    source_episode_index: int | None


class _StatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None
        self.total: np.ndarray | None = None
        self.total_squares: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] == 0:
            return
        minimum = np.min(array, axis=0)
        maximum = np.max(array, axis=0)
        total = np.sum(array, axis=0)
        total_squares = np.sum(array * array, axis=0)
        if self.minimum is None:
            self.minimum = minimum
            self.maximum = maximum
            self.total = total
            self.total_squares = total_squares
        else:
            assert self.maximum is not None
            assert self.total is not None
            assert self.total_squares is not None
            self.minimum = np.minimum(self.minimum, minimum)
            self.maximum = np.maximum(self.maximum, maximum)
            self.total += total
            self.total_squares += total_squares
        self.count += array.shape[0]

    def result(self) -> dict[str, Any]:
        if self.count == 0 or self.minimum is None:
            raise ValueError("cannot compute statistics from no frames")
        assert self.maximum is not None
        assert self.total is not None
        assert self.total_squares is not None
        mean = self.total / self.count
        variance = np.maximum(self.total_squares / self.count - mean * mean, 0.0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
        }

    @property
    def has_data(self) -> bool:
        return self.count > 0


def is_rejected_episode(row: dict[str, Any]) -> bool:
    """Match the web console's red-state rule for one episode."""
    return (
        str(row.get("quality", "green")).lower() == "red"
        or str(row.get("qc_verdict", "")).lower() == "fail"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _format_path(
    root: Path,
    template: str,
    episode_index: int,
    *,
    chunks_size: int,
    **values: object,
) -> Path:
    return root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        **values,
    )


def _export_subset(
    source_dir: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    subset: str,
    episodes_offset: int,
    episodes_total: int,
    progress_callback: Callable[[QualityExportProgress], None] | None,
) -> tuple[int, int]:
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    video_path = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    source_stats = json.loads((source_dir / "meta" / "stats.json").read_text())
    accumulators = {name: _StatsAccumulator() for name in source_stats}
    episode_stats_by_index = {
        int(row["episode_index"]): row
        for row in _read_jsonl(source_dir / "meta" / "episodes_stats.jsonl")
    }
    default_video_keys = tuple(
        name
        for name, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    )

    (output_dir / "meta").mkdir(parents=True)
    tasks_path = source_dir / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        shutil.copy2(tasks_path, output_dir / "meta" / "tasks.jsonl")

    output_rows: list[dict[str, Any]] = []
    output_stats_rows: list[dict[str, Any]] = []
    global_index = 0
    total_videos = 0
    source_indices: list[int] = []
    for new_index, source_row in enumerate(rows):
        source_index = int(source_row["episode_index"])
        source_indices.append(source_index)
        source_parquet = _format_path(source_dir, data_path, source_index, chunks_size=chunks_size)
        output_parquet = _format_path(output_dir, data_path, new_index, chunks_size=chunks_size)
        if not source_parquet.is_file():
            raise FileNotFoundError(f"episode parquet not found: {source_parquet}")
        table = pq.read_table(source_parquet)
        frame_count = table.num_rows
        for name, accumulator in accumulators.items():
            if name not in table.column_names:
                raise ValueError(f"statistics column {name!r} missing from {source_parquet}")
            accumulator.update(np.asarray(table[name].to_pylist(), dtype=np.float64))
        if "episode_index" not in table.column_names or "index" not in table.column_names:
            raise ValueError(f"episode parquet lacks index columns: {source_parquet}")
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(np.full(frame_count, new_index, dtype=np.int64)),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(np.arange(global_index, global_index + frame_count, dtype=np.int64)),
        )
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_parquet)
        global_index += frame_count

        video_keys = tuple(source_row.get("video_keys") or default_video_keys)
        copied_videos = 0
        for video_key in video_keys:
            source_video = _format_path(
                source_dir,
                video_path,
                source_index,
                chunks_size=chunks_size,
                video_key=video_key,
            )
            if not source_video.is_file():
                raise FileNotFoundError(f"episode video not found: {source_video}")
            output_video = _format_path(
                output_dir,
                video_path,
                new_index,
                chunks_size=chunks_size,
                video_key=video_key,
            )
            output_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, output_video)
            copied_videos += 1
        total_videos += copied_videos

        output_row = dict(source_row)
        output_row["episode_index"] = new_index
        output_row["length"] = frame_count
        output_row["total_videos"] = copied_videos
        output_rows.append(output_row)
        if source_index in episode_stats_by_index:
            stats_row = dict(episode_stats_by_index[source_index])
            stats_row["episode_index"] = new_index
            output_stats_rows.append(stats_row)
        if progress_callback is not None:
            progress_callback(
                QualityExportProgress(
                    episodes_completed=episodes_offset + new_index + 1,
                    episodes_total=episodes_total,
                    subset=subset,
                    source_episode_index=source_index,
                )
            )

    info.update(
        {
            "total_episodes": len(output_rows),
            "total_frames": global_index,
            "total_videos": total_videos,
            "total_chunks": max(1, (len(output_rows) + chunks_size - 1) // chunks_size),
            "splits": {"train": f"0:{len(output_rows)}"},
        }
    )
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output_dir / "meta" / "stats.json").write_text(
        json.dumps(
            {name: value.result() for name, value in accumulators.items() if value.has_data},
            indent=2,
        )
        + "\n"
    )
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", output_rows)
    _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", output_stats_rows)
    (output_dir / "meta" / "quality_split.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "subset": subset,
                "rule": "quality == red OR qc_verdict == fail",
                "source_episode_indices": source_indices,
            },
            indent=2,
        )
        + "\n"
    )
    return global_index, total_videos


def split_dataset_by_quality(
    source_dir: Path,
    accepted_dir: Path | None = None,
    rejected_dir: Path | None = None,
    *,
    replace_existing: bool = False,
    progress_callback: Callable[[QualityExportProgress], None] | None = None,
) -> QualitySplitSummary:
    """Export accepted and quarantined datasets using the console's red-state rule."""
    source_dir = Path(source_dir).resolve()
    accepted_dir = Path(
        accepted_dir or source_dir.with_name(source_dir.name + "_accepted")
    ).resolve()
    rejected_dir = Path(
        rejected_dir or source_dir.with_name(source_dir.name + "_rejected")
    ).resolve()
    if not (source_dir / "meta" / "episodes.jsonl").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source_dir}")
    if accepted_dir == rejected_dir or source_dir in {accepted_dir, rejected_dir}:
        raise ValueError("source, accepted, and rejected directories must be distinct")
    for output in (accepted_dir, rejected_dir):
        if output.exists() and not replace_existing:
            raise FileExistsError(f"output directory already exists: {output}")
        if output.exists() and not output.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_dir / "meta" / "episodes.jsonl")
    if not rows:
        raise ValueError("dataset has no episode metadata")
    indices = [int(row["episode_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("episode indices must be unique")
    accepted_rows = [row for row in rows if not is_rejected_episode(row)]
    rejected_rows = [row for row in rows if is_rejected_episode(row)]
    if not accepted_rows:
        raise ValueError("quality split found no accepted episodes")
    if progress_callback is not None:
        progress_callback(QualityExportProgress(0, len(rows), "", None))

    accepted_stage = Path(
        tempfile.mkdtemp(prefix=f".{accepted_dir.name}.", dir=accepted_dir.parent)
    )
    rejected_stage = Path(
        tempfile.mkdtemp(prefix=f".{rejected_dir.name}.", dir=rejected_dir.parent)
    )
    published: list[Path] = []
    backups: dict[Path, tuple[Path, Path]] = {}
    try:
        accepted_frames, _ = _export_subset(
            source_dir,
            accepted_stage,
            accepted_rows,
            subset="accepted",
            episodes_offset=0,
            episodes_total=len(rows),
            progress_callback=progress_callback,
        )
        rejected_frames, _ = _export_subset(
            source_dir,
            rejected_stage,
            rejected_rows,
            subset="rejected",
            episodes_offset=len(accepted_rows),
            episodes_total=len(rows),
            progress_callback=progress_callback,
        )
        for output, stage in (
            (accepted_dir, accepted_stage),
            (rejected_dir, rejected_stage),
        ):
            if output.exists():
                backup_root = Path(
                    tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent)
                )
                backup = backup_root / output.name
                output.replace(backup)
                backups[output] = (backup_root, backup)
            stage.replace(output)
            published.append(output)
    except Exception:
        shutil.rmtree(accepted_stage, ignore_errors=True)
        shutil.rmtree(rejected_stage, ignore_errors=True)
        for output in reversed(published):
            shutil.rmtree(output, ignore_errors=True)
        for output, (_, backup) in backups.items():
            if backup.exists() and not output.exists():
                backup.replace(output)
        raise
    finally:
        for backup_root, _ in backups.values():
            shutil.rmtree(backup_root, ignore_errors=True)

    return QualitySplitSummary(
        source_dir=str(source_dir),
        accepted_dir=str(accepted_dir),
        rejected_dir=str(rejected_dir),
        source_episodes=len(rows),
        accepted_episodes=len(accepted_rows),
        rejected_episodes=len(rejected_rows),
        accepted_frames=accepted_frames,
        rejected_frames=rejected_frames,
        rejected_source_indices=tuple(int(row["episode_index"]) for row in rejected_rows),
    )


__all__ = [
    "QualityExportProgress",
    "QualitySplitSummary",
    "is_rejected_episode",
    "split_dataset_by_quality",
]
