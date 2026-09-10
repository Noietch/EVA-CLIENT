from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.recorder.video_encoding import DATASET_VIDEO_GOP_SIZE, dataset_h264_ffmpeg_params

from ._publish import publish_output_pair
from .source import video_frames


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
        self._merge(minimum, maximum, total, total_squares, array.shape[0])

    def update_summary(self, summary: dict[str, Any]) -> None:
        count_array = np.asarray(summary.get("count", []), dtype=np.float64)
        if count_array.size == 0:
            raise ValueError("episode statistics entry is missing count")
        count = int(np.max(count_array))
        if count <= 0:
            return
        minimum = np.asarray(summary["min"], dtype=np.float64)
        maximum = np.asarray(summary["max"], dtype=np.float64)
        mean = np.asarray(summary["mean"], dtype=np.float64)
        std = np.asarray(summary["std"], dtype=np.float64)
        total = mean * count
        total_squares = (np.square(std) + np.square(mean)) * count
        self._merge(minimum, maximum, total, total_squares, count)

    def _merge(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
        total: np.ndarray,
        total_squares: np.ndarray,
        count: int,
    ) -> None:
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
        self.count += count

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
    qc_verdict = str(row.get("qc_verdict", "")).lower()
    if qc_verdict == "pass":
        return False
    if qc_verdict == "fail":
        return True
    return str(row.get("quality", "green")).lower() == "red"


def _latest_slot_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep exactly the latest attempt for each capture slot, regardless of QC."""
    def legacy_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
        scene_id = str(row.get("scene_id") or "").strip()
        try:
            round_index = int(row["scene_round"])
        except (KeyError, TypeError, ValueError):
            return []
        if not scene_id or round_index < 0:
            return []
        keys = []
        task_id = str(row.get("task_id") or "").strip()
        if task_id:
            keys.append(("slot", f"{task_id}:{scene_id}:{round_index}"))
        tasks = row.get("tasks")
        prompt = row.get("task") or row.get("prompt")
        if not prompt and isinstance(tasks, list) and len(tasks) == 1:
            prompt = tasks[0]
        if prompt:
            keys.append(("target", json.dumps([str(prompt), scene_id, round_index])))
        return keys

    # Resolve old target metadata to an explicit slot when the mapping is unique.
    # Do not merge distinct explicit slots merely because their prompts match.
    aliases: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        slot_id = str(row.get("slot_id") or "").strip()
        if slot_id:
            for key in legacy_keys(row):
                aliases.setdefault(key, set()).add(slot_id)

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        slot_id = str(row.get("slot_id") or "").strip()
        key = ("slot", slot_id) if slot_id else ("episode", str(row["episode_index"]))
        if not slot_id:
            keys = legacy_keys(row)
            if keys:
                key = keys[0]
                # A task id is stronger than a shared human-readable prompt.
                for candidate in keys[:1]:
                    matches = aliases.get(candidate, set())
                    if len(matches) == 1:
                        key = ("slot", next(iter(matches)))
                        break
        previous = selected.get(key)
        if previous is None or int(row["episode_index"]) > int(previous["episode_index"]):
            selected[key] = row
    return list(selected.values())


def split_dataset_by_quality(
    source_dir: Path,
    accepted_dir: Path | None = None,
    rejected_dir: Path | None = None,
    *,
    replace_existing: bool = False,
    normalize_videos: bool = True,
    progress_callback: Callable[[QualityExportProgress], None] | None = None,
    source_episode_indices: set[int] | None = None,
) -> QualitySplitSummary:
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
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    _require_lerobot_v21(info, source_dir)
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
    if source_episode_indices is None:
        rows = _latest_slot_episodes(rows)
    else:
        missing = source_episode_indices - set(indices)
        if missing:
            raise ValueError(f"selected episodes missing from dataset: {sorted(missing)}")
        rows = [row for row in rows if int(row["episode_index"]) in source_episode_indices]
    accepted_rows = [row for row in rows if not is_rejected_episode(row)]
    rejected_rows = [row for row in rows if is_rejected_episode(row)]
    if progress_callback is not None:
        progress_callback(QualityExportProgress(0, len(rows), "", None))

    accepted_stage = Path(
        tempfile.mkdtemp(prefix=f".{accepted_dir.name}.", dir=accepted_dir.parent)
    )
    rejected_stage = Path(
        tempfile.mkdtemp(prefix=f".{rejected_dir.name}.", dir=rejected_dir.parent)
    )
    try:
        accepted_frames = _export_subset(
            source_dir,
            accepted_stage,
            accepted_rows,
            subset="accepted",
            episodes_offset=0,
            episodes_total=len(rows),
            normalize_videos=normalize_videos,
            progress_callback=progress_callback,
        )
        rejected_frames = _export_subset(
            source_dir,
            rejected_stage,
            rejected_rows,
            subset="rejected",
            episodes_offset=len(accepted_rows),
            episodes_total=len(rows),
            normalize_videos=normalize_videos,
            progress_callback=progress_callback,
        )
        publish_output_pair(
            ((accepted_dir, accepted_stage), (rejected_dir, rejected_stage)),
            replace_existing=replace_existing,
        )
    except Exception:
        shutil.rmtree(accepted_stage, ignore_errors=True)
        shutil.rmtree(rejected_stage, ignore_errors=True)
        raise

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


def _export_subset(
    source_dir: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    subset: str,
    episodes_offset: int,
    episodes_total: int,
    normalize_videos: bool,
    progress_callback: Callable[[QualityExportProgress], None] | None,
) -> int:
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    _require_lerobot_v21(info, source_dir)
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
    fps = float(info["fps"])
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
    source_tasks = _read_jsonl(tasks_path)
    task_name_by_index = {
        int(row["task_index"]): str(row["task"])
        for row in source_tasks
        if "task_index" in row and "task" in row
    }
    subset_tasks: list[str] = []
    subset_task_index: dict[str, int] = {}

    output_rows: list[dict[str, Any]] = []
    output_stats_rows: list[dict[str, Any]] = []
    global_index = 0
    total_videos = 0
    source_indices: list[int] = []

    def _register_task(task_name: str) -> int:
        index = subset_task_index.get(task_name)
        if index is not None:
            return index
        index = len(subset_tasks)
        subset_task_index[task_name] = index
        subset_tasks.append(task_name)
        return index

    for new_index, source_row in enumerate(rows):
        source_index = int(source_row["episode_index"])
        source_indices.append(source_index)
        source_parquet = _format_path(source_dir, data_path, source_index, chunks_size=chunks_size)
        output_parquet = _format_path(output_dir, data_path, new_index, chunks_size=chunks_size)
        if not source_parquet.is_file():
            raise FileNotFoundError(f"episode parquet not found: {source_parquet}")
        table = pq.read_table(source_parquet)
        frame_count = table.num_rows
        episode_stats = episode_stats_by_index.get(source_index, {}).get("stats", {})
        for name, accumulator in accumulators.items():
            if name in episode_stats:
                accumulator.update_summary(episode_stats[name])
                continue
            if name in table.column_names:
                accumulator.update(np.asarray(table[name].to_pylist(), dtype=np.float64))
                continue
            raise ValueError(f"statistics column {name!r} missing from {source_parquet}")
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
        row_tasks = list(dict.fromkeys(str(task) for task in source_row.get("tasks", [])))
        for task_name in row_tasks:
            _register_task(task_name)
        if "task_index" in table.column_names:
            source_task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
            remapped_task_indices = []
            for source_task_index in source_task_indices:
                task_name = task_name_by_index.get(int(source_task_index))
                if task_name is None:
                    raise ValueError(
                        f"task index {int(source_task_index)} missing from {tasks_path}"
                    )
                remapped_task_indices.append(_register_task(task_name))
            table = table.set_column(
                table.schema.get_field_index("task_index"),
                "task_index",
                pa.array(np.asarray(remapped_task_indices, dtype=np.int64)),
            )
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_parquet)
        global_index += frame_count

        written_videos = 0
        for video_key in tuple(source_row.get("video_keys") or default_video_keys):
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
            if normalize_videos:
                _transcode_dataset_video(
                    source_video,
                    output_video,
                    fps=fps,
                    expected_frames=frame_count,
                )
            else:
                _link_or_copy_video(source_video, output_video)
            written_videos += 1
        total_videos += written_videos

        output_row = dict(source_row)
        output_row["episode_index"] = new_index
        output_row["length"] = frame_count
        output_row["total_videos"] = written_videos
        if row_tasks:
            output_row["tasks"] = row_tasks
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
            "total_tasks": len(subset_tasks),
            "total_chunks": (
                0 if not output_rows else (len(output_rows) + chunks_size - 1) // chunks_size
            ),
            "splits": {"train": f"0:{len(output_rows)}"},
        }
    )
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    _write_jsonl(
        output_dir / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(subset_tasks)],
    )
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
                "dataset_format": "lerobot_v21",
                "rule": "qc_verdict == pass overrides quality == red; qc_verdict == fail rejects",
                "episode_selection": "latest attempt per slot; unassigned episodes retained",
                "source_episode_indices": source_indices,
            },
            indent=2,
        )
        + "\n"
    )
    return global_index


def _transcode_dataset_video(
    source: Path,
    target: Path,
    *,
    fps: float,
    expected_frames: int,
) -> None:
    """Rewrite one LeRobot v2.1 video with the required dataset GOP."""
    if _dataset_video_is_compatible(source, fps=fps, expected_frames=expected_frames):
        _link_or_copy_video(source, target)
        return
    writer = imageio.get_writer(
        str(target),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=dataset_h264_ffmpeg_params(threads=None),
    )
    observed_frames = 0
    try:
        for frame in video_frames(source):
            writer.append_data(np.ascontiguousarray(frame))
            observed_frames += 1
    finally:
        writer.close()
    if observed_frames != expected_frames:
        raise ValueError(
            f"LeRobot v2.1 video {source} has {observed_frames} frames; expected {expected_frames}"
        )


def _dataset_video_is_compatible(
    source: Path,
    *,
    fps: float,
    expected_frames: int,
) -> bool:
    """Check frame count, rate, codec, and GOP from packets without decoding pixels."""
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            if stream.codec_context.name != "h264":
                return False
            rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
            if rate <= 0.0 or abs(rate - fps) > max(0.01, fps * 0.001):
                return False
            keyframes: list[int] = []
            packet_count = 0
            for packet in container.demux(stream):
                if packet.size <= 0:
                    continue
                if packet.is_keyframe:
                    keyframes.append(packet_count)
                packet_count += 1
    except (OSError, ValueError, av.error.FFmpegError):
        return False
    return packet_count == expected_frames and keyframes == list(
        range(0, expected_frames, DATASET_VIDEO_GOP_SIZE)
    )


def _link_or_copy_video(source: Path, target: Path) -> None:
    """Reuse source video bytes for an intermediate conversion dataset."""
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


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


def _require_lerobot_v21(info: dict[str, Any], source_dir: Path) -> None:
    dataset_format = str(info.get("dataset_format", "lerobot_v21") or "lerobot_v21")
    if str(info.get("codebase_version", "")) != "v2.1" or dataset_format != "lerobot_v21":
        raise ValueError(f"quality split requires a LeRobot v2.1 dataset: {source_dir}")


__all__ = [
    "QualityExportProgress",
    "QualitySplitSummary",
    "is_rejected_episode",
    "split_dataset_by_quality",
]
