from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.types import CollectionRawSample, Observation, RolloutInterventionSegment

COLLECTION_VECTOR_FIELDS = (
    "state_qpos",
    "state_eef",
    "action_qpos",
    "action_eef",
)


@dataclasses.dataclass
class QualityIssue:
    severity: str
    code: str
    detail: str


@dataclasses.dataclass
class SaveJob:
    episode_index: int
    task: str
    task_index: int
    global_index: int
    steps: list[Observation]
    episode_meta: dict[str, Any]
    task_to_index: dict[str, int]
    collection_columns: dict[str, Any] | None = None
    collection_episode_row: dict[str, Any] | None = None
    collection_payload: Any | None = None
    raw_episode_payload: Any | None = None
    videos: dict[str, list[np.ndarray]] | None = None
    raw_video_samples: dict[str, list[CollectionRawSample]] | None = None
    video_fps: float | None = None
    dataset_dir: Path | None = None
    intervention_segments: list[RolloutInterventionSegment] = dataclasses.field(
        default_factory=list
    )
    reason: str = ""
    meta_written: bool = False
    status: str = "queued"
    error: BaseException | None = None
    queued_wall_time: float = 0.0
    started_wall_time: float = 0.0
    finished_wall_time: float = 0.0


class StatAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        batch_min = arr.min(axis=0)
        batch_max = arr.max(axis=0)
        batch_sum = arr.sum(axis=0)
        batch_sumsq = (arr * arr).sum(axis=0)
        if self._min is None or self._max is None or self._sum is None or self._sumsq is None:
            self._min, self._max = batch_min, batch_max
            self._sum, self._sumsq = batch_sum, batch_sumsq
        else:
            self._min = np.minimum(self._min, batch_min)
            self._max = np.maximum(self._max, batch_max)
            self._sum += batch_sum
            self._sumsq += batch_sumsq
        self.count += arr.shape[0]

    def result(self) -> dict[str, list[float]]:
        assert self._min is not None and self._max is not None
        assert self._sum is not None and self._sumsq is not None
        mean = self._sum / self.count
        variance = np.maximum(self._sumsq / self.count - mean * mean, 0.0)
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for raw in stream if (line := raw.strip())]


def to_rgb_uint8(frame: np.ndarray, convert_bgr_to_rgb: bool) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if convert_bgr_to_rgb and array.ndim == 3 and array.shape[2] == 3:
        array = array[:, :, ::-1]
    return array
