from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import numpy as np
from openpi_client import msgpack_numpy

ColumnValue: TypeAlias = np.ndarray | list[object]
ImageBatch: TypeAlias = np.ndarray
DataFormat: TypeAlias = str
CANONICAL_DATA_FORMATS = ("lerobot_v21", "lerobot_v3", "mcap", "hdf5")
_ALIASES = {
    "lerobot": "lerobot_v21",
    "lerobot2": "lerobot_v21",
    "lerobot21": "lerobot_v21",
    "lerobotv2": "lerobot_v21",
    "lerobotv21": "lerobot_v21",
    "lerobotv210": "lerobot_v21",
    "lerobot3": "lerobot_v3",
    "lerobot30": "lerobot_v3",
    "lerobotv3": "lerobot_v3",
    "lerobotv30": "lerobot_v3",
    "mcap": "mcap",
    "hdf": "hdf5",
    "h5": "hdf5",
    "hdf5": "hdf5",
}
_MCAP_MAGIC = b"\x89MCAP0\r\n"
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_DATA_PATHS = {
    "lerobot_v21": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "lerobot_v3": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "hdf5": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.hdf5",
    "mcap": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mcap",
}
_V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
LANGUAGE_ANNOTATION_KEY = "language_annotation"


class InferredKeyRole(TypedDict):
    candidates: list[str]
    default: str | None


InferKeysResult: TypeAlias = dict[str, InferredKeyRole]


@dataclass(frozen=True, slots=True)
class VideoRef:
    path: Path
    start_time: float = 0.0
    end_time: float | None = None


@dataclass(slots=True)
class EpisodeData:
    columns: dict[str, ColumnValue]
    task: str = ""
    fps: float | None = None
    videos: dict[str, VideoRef] = field(default_factory=dict)
    images: dict[str, ImageBatch] = field(default_factory=dict)


_BOOKKEEPING_COLUMNS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
}
_STATE_PREFERENCES = (
    "observation.state",
    "observations.state.qpos",
    "observation.qpos",
    "state",
    "state_qpos",
    "qpos",
)
_ACTION_PREFERENCES = (
    "action",
    "actions",
    "action_qpos",
    "action_eef",
)


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def normalize_data_format(value: str) -> DataFormat:
    compact = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    normalized = _ALIASES.get(compact)
    if normalized is None:
        raise ValueError(
            f"Unsupported data format {value!r}. Expected one of: "
            f"{', '.join(CANONICAL_DATA_FORMATS)}."
        )
    return normalized


def detect_data_format(path: str | Path) -> DataFormat:
    target = normalize_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Cannot use dataset_format='auto' for a new path: {target}. "
            "Choose an explicit format for writing."
        )
    if target.is_file():
        if _matches_magic(target, _MCAP_MAGIC, tail=True):
            return "mcap"
        if _matches_hdf5(target):
            return "hdf5"
        raise ValueError(f"Could not detect a supported dataset format from file: {target}")

    info_path = target / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        declared = str(info.get("dataset_format", "")).strip()
        if declared:
            return normalize_data_format(declared)
        version = str(info.get("codebase_version", "")).lower()
        if version.startswith(("v3", "3")):
            return "lerobot_v3"
        if version.startswith(("v2", "2")):
            return "lerobot_v21"
        return "lerobot_v3" if (target / "meta" / "episodes").is_dir() else "lerobot_v21"
    patterns = {
        "lerobot_v21": ("data/chunk-*/episode_*.parquet",),
        "lerobot_v3": (
            "meta/episodes/chunk-*/file-*.parquet",
            "data/chunk-*/file-*.parquet",
        ),
        "mcap": ("episode_*.mcap", "data/chunk-*/episode_*.mcap"),
        "hdf5": (
            "episode_*.h5",
            "episode_*.hdf5",
            "data/chunk-*/episode_*.h5",
            "data/chunk-*/episode_*.hdf5",
        ),
    }
    for name, candidates in patterns.items():
        if any(next(target.glob(pattern), None) is not None for pattern in candidates):
            return name
    raise ValueError(f"Could not detect a supported dataset format from directory: {target}")


def build_info(
    *,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    fps: float,
    features: dict[str, Any],
    data_format: str = "lerobot_v21",
) -> dict[str, Any]:
    is_v3 = data_format == "lerobot_v3"
    is_lerobot = data_format.startswith("lerobot_")
    info: dict[str, Any] = {
        "dataset_format": data_format,
        "codebase_version": {
            "lerobot_v21": "v2.1",
            "lerobot_v3": "v3.0",
            "hdf5": "hdf5",
            "mcap": "mcap",
        }[data_format],
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": _DATA_PATHS[data_format],
        "features": features,
    }
    if is_v3:
        info.update(
            video_path=_V3_VIDEO_PATH,
            data_files_size_in_mb=100,
            video_files_size_in_mb=500,
        )
    elif is_lerobot:
        info.update(video_path=_V21_VIDEO_PATH, total_videos=total_videos, total_chunks=1)
    else:
        info.update(total_videos=0, total_chunks=1, embedded_images=True)
    return info


def summarize_quality_issues(
    issues: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int]:
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for issue in issues or []:
        severity = str(issue.get("severity", "red"))
        code = str(issue.get("code", "issue"))
        key = (severity, code)
        if key not in summaries:
            summaries[key] = {
                "severity": severity,
                "code": code,
                "detail": str(issue.get("detail", "")),
                "count": 0,
            }
        summaries[key]["count"] += 1
    return list(summaries.values()), len(issues or [])


def history_row(row: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    quality_issues, quality_issue_count = summarize_quality_issues(row.get("quality_issues"))
    return {
        "episode_index": int(row.get("episode_index", fallback_index)),
        "length": int(row.get("length", 0)),
        "status": "saved",
        "quality": row.get("quality", "green"),
        "qc_verdict": row.get("qc_verdict", ""),
        "qc_note": row.get("qc_note", ""),
        "quality_issues": quality_issues,
        "quality_issue_count": quality_issue_count,
        "error": "",
    }


def build_episode_metadata(row: Mapping[str, Any], fps: float) -> dict[str, Any]:
    metadata = dict(row)
    tasks = metadata.get("tasks") or []
    metadata.setdefault("task", str(tasks[0]) if tasks else "")
    metadata.setdefault("fps", fps)
    return metadata


def _matches_magic(path: Path, magic: bytes, *, tail: bool = False) -> bool:
    if path.stat().st_size < len(magic) * (2 if tail else 1):
        return False
    with path.open("rb") as stream:
        if stream.read(len(magic)) != magic:
            return False
        if tail:
            stream.seek(-len(magic), 2)
            return stream.read(len(magic)) == magic
    return True


def _matches_hdf5(path: Path) -> bool:
    size = path.stat().st_size
    with path.open("rb") as stream:
        for offset in (0, 512, 1024, 2048, 4096, 8192, 16384):
            if offset + len(_HDF5_MAGIC) <= size:
                stream.seek(offset)
                if stream.read(len(_HDF5_MAGIC)) == _HDF5_MAGIC:
                    return True
    return False


def select_default(candidates: list[str], preferred: tuple[str, ...]) -> str | None:
    for key in preferred:
        if key in candidates:
            return key
    return candidates[0] if candidates else None


def build_infer_keys(image_keys: list[str], column_keys: list[str]) -> InferKeysResult:
    images = sorted(dict.fromkeys(image_keys))
    vectors = [key for key in column_keys if key not in _BOOKKEEPING_COLUMNS]
    non_image = vectors or list(column_keys)
    state_candidates = [key for key in non_image if "state" in key.lower() or "qpos" in key.lower()]
    action_candidates = [key for key in non_image if "action" in key.lower()]
    return {
        "image": {
            "candidates": images,
            "default": images[0] if images else None,
        },
        "state": {
            "candidates": non_image,
            "default": select_default(state_candidates, _STATE_PREFERENCES)
            or select_default(non_image, _STATE_PREFERENCES),
        },
        "action": {
            "candidates": non_image,
            "default": select_default(action_candidates, _ACTION_PREFERENCES)
            or select_default(non_image, _ACTION_PREFERENCES),
        },
    }


def coerce_column(values: list[Any]) -> np.ndarray | list[object]:
    if not values:
        return np.asarray(values)
    array = np.asarray(values)
    if array.dtype != object:
        return array
    first = values[0]
    if isinstance(first, np.ndarray):
        try:
            return np.stack([np.asarray(value) for value in values], axis=0)
        except ValueError:
            return [np.asarray(value) for value in values]
    if isinstance(first, (list, tuple)):
        try:
            return np.asarray(values)
        except ValueError:
            return [list(value) if isinstance(value, tuple) else value for value in values]
    return values


def coerce_episode_data(payload: EpisodeData | dict[str, Any]) -> EpisodeData:
    if isinstance(payload, EpisodeData):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported episode payload type: {type(payload)!r}")
    videos = {
        key: _coerce_video_ref(value) for key, value in dict(payload.get("videos", {})).items()
    }
    images = {key: np.asarray(value) for key, value in dict(payload.get("images", {})).items()}
    columns = {
        key: value if isinstance(value, list) else np.asarray(value)
        for key, value in dict(payload.get("columns", {})).items()
    }
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    task = payload.get("task", metadata.get("task", ""))
    fps = payload.get("fps", metadata.get("fps"))
    return EpisodeData(
        columns=columns,
        task=str(task),
        fps=float(fps) if fps is not None else None,
        videos=videos,
        images=images,
    )


def pack_value(value: Any) -> bytes:
    packed = msgpack_numpy.packb(_to_messagepack(value))
    if not isinstance(packed, bytes):
        raise TypeError("msgpack encoder did not return bytes")
    return packed


def unpack_value(value: bytes) -> Any:
    return _from_messagepack(msgpack_numpy.unpackb(value))


def _to_messagepack(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.kind in {"O", "U"}:
        return {
            "__eva_numpy_array__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {key: _to_messagepack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_messagepack(item) for item in value]
    return value


def _from_messagepack(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__eva_numpy_array__") is True:
        array = np.asarray(value["values"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"]))
    if isinstance(value, dict):
        return {key: _from_messagepack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_messagepack(item) for item in value]
    return value


def _coerce_video_ref(value: Any) -> VideoRef:
    if isinstance(value, VideoRef):
        return value
    if isinstance(value, (str, Path)):
        return VideoRef(path=Path(value))
    if isinstance(value, dict):
        return VideoRef(
            path=Path(value["path"]),
            start_time=float(value.get("start_time", 0.0)),
            end_time=float(value["end_time"]) if value.get("end_time") is not None else None,
        )
    raise TypeError(f"Unsupported video reference type: {type(value)!r}")


class BaseDataset(ABC):
    format: str
    _registry: dict[str, type[BaseDataset]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.format:
            BaseDataset._registry[cls.format] = cls

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        video_keys: Iterable[str] = (),
    ) -> None:
        self.path = normalize_path(path)
        self.fps = float(fps) if fps is not None else None
        self.video_keys = tuple(sorted(set(video_keys)))

    @classmethod
    def open(
        cls,
        path: str | Path,
        format: str = "auto",
        *,
        fps: float | None = None,
        video_keys: tuple[str, ...] = (),
    ) -> BaseDataset:
        dataset_path = normalize_path(path)
        if format.strip().lower() == "auto" and fps is not None:
            raise ValueError("dataset_format='auto' is read-only; choose an explicit write format")
        name = (
            detect_data_format(dataset_path)
            if format.strip().lower() == "auto"
            else normalize_data_format(format)
        )
        return cls._registry[name](dataset_path, fps=fps, video_keys=video_keys)

    @abstractmethod
    def count_episodes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def load(self, episode_index: int) -> EpisodeData:
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        columns: dict[str, Any],
        row: dict[str, Any],
        videos: Mapping[str, Iterable[np.ndarray]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode_index in range(self.count_episodes()):
            episode = self.load(episode_index)
            first_column = next(iter(episode.columns.values()), ())
            rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [episode.task] if episode.task else [],
                    "length": len(first_column),
                }
            )
        return rows

    def infer_keys(self) -> InferKeysResult:
        sample = self.load(0)
        image_keys = list(sample.images) or list(sample.videos)
        return build_infer_keys(image_keys=image_keys, column_keys=list(sample.columns))

    def episode_videos(self, episode_index: int) -> dict[str, VideoRef]:
        return self.load(episode_index).videos

    def load_columns(self, episode_index: int) -> dict[str, ColumnValue]:
        return self.load(episode_index).columns

    def load_image_frame(
        self,
        episode_index: int,
        image_key: str,
        frame_index: int,
    ) -> np.ndarray | None:
        if frame_index < 0:
            return None
        frames = self.load(episode_index).images.get(image_key)
        if frames is None or frame_index >= len(frames):
            return None
        return np.asarray(frames[frame_index])

    def patch_episode(self, episode_index: int, values: Mapping[str, Any]) -> bool:
        del episode_index, values
        return False

    def patch_episode_metadata(self, episode_index: int, values: Mapping[str, Any]) -> bool:
        return self.patch_episode(episode_index, values)

    def mark_qc(self, episode_index: int, verdict: str, note: str = "") -> bool:
        values = {"qc_note": note}
        if verdict:
            values["qc_verdict"] = verdict
        return self.patch_episode(episode_index, values)

    def read_annotation(self, episode_index: int) -> str:
        del episode_index
        return ""

    def annotate_language(self, episode_index: int, annotation: str) -> bool:
        del episode_index, annotation
        return False

    def seal_current_shard(self) -> None:
        return

    def finalize(self) -> None:
        return

    def require_fps(self) -> float:
        if self.fps is None:
            raise ValueError(f"{self.format} writing requires fps")
        return self.fps

    def episode_paths(self, suffixes: tuple[str, ...]) -> list[Path]:
        if self.path.is_file():
            if self.path.suffix.lower() not in suffixes:
                raise FileNotFoundError(f"Unsupported dataset file: {self.path}")
            return [self.path]
        paths: list[Path] = []
        for suffix in suffixes:
            paths.extend(self.path.glob(f"episode_*{suffix}"))
            paths.extend(self.path.glob(f"data/chunk-*/episode_*{suffix}"))
            paths.extend(self.path.glob(f"data/chunk-*/file-*{suffix}"))
        return sorted(set(paths))

    def episode_path(self, episode_index: int, suffixes: tuple[str, ...]) -> Path:
        paths = self.episode_paths(suffixes)
        if self.path.is_file() and int(episode_index) == 0:
            return paths[0]
        expected = f"episode_{int(episode_index):06d}"
        for path in paths:
            if path.stem == expected:
                return path
        try:
            return paths[int(episode_index)]
        except IndexError as exc:
            raise IndexError(f"Episode index {episode_index} out of range for {self.path}") from exc
