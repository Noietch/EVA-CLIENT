"""Episode preview backend for the read-only viewer.

A trial's result row carries a clip_id and (joined) episode_index into the dataset
recorded under <results_dir>/episodes. This module reads that episode's per-frame
state/action series and camera mp4s, and runs the robot's forward kinematics so the
viewer can replay the URDF motion frame by frame — reusing the console's
"geometry once + per-frame 4x4 transform" scheme (UrdfScene).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import robots  # noqa: F401  (import side effect registers robots in ROBOT_REGISTRY)
from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger
from robots import build_robot

if TYPE_CHECKING:
    from robots.utils import UrdfScene

logger = logging.getLogger(__name__)


def read_result_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """Scored/owned trial rows for one per-model dataset, read from meta/episodes.jsonl.

    Args:
        dataset_root: the per-model lerobot dataset root (``<model>/episodes``).

    Returns:
        rows: list of result dicts, deduped to the latest episode per clip_id. Rows
            without a clip_id (plain collection) are skipped.
    """
    path = dataset_root / "meta" / "episodes.jsonl"
    if not path.exists():
        return []
    by_clip: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("clip_id")
            if not cid:
                continue
            rec = {
                "clip_id": cid,
                "episode_index": row.get("episode_index"),
                "prompt": row.get("prompt") or (row.get("tasks") or [None])[0],
                "trial": row.get("trial"),
                "score": row.get("score"),
                "max_score": row.get("max_score"),
                "milestones": row.get("milestones", {}),
                "note": row.get("note", ""),
                "scored_at": row.get("scored_at"),
                "duration_ms": row.get("duration_ms"),
            }
            if cid not in by_clip:
                order.append(cid)
            by_clip[cid] = rec  # later episode row for a clip wins
    return [by_clip[c] for c in order]


class EpisodePreview:
    """Lazily builds a robot + URDF scene + read-only episode reader for one run.

    All previews share the single dataset under ``<results_dir>/episodes`` (every
    ckpt's trials record into it, keyed by clip_id). The UrdfScene is optional —
    when the robot has no vis_config the URDF replay endpoints degrade to empty.
    """

    def __init__(self, config: ConfigDict, results_dir: Path) -> None:
        self._config = config
        self._dataset_root = results_dir / "episodes"
        self._robot = build_robot(config)
        self._reader: EpisodeLogger | None = None
        self._scene: UrdfScene | None = None
        self._scene_tried = False
        self._xf_cache: dict[tuple[Path, int, int], bytes] = {}

    def set_dataset_root(self, dataset_root: Path) -> None:
        """Repoint at a different per-model dataset (on ckpt switch). Drops the cached
        reader so the next read reflects the new model's episodes; the URDF scene is
        robot-scoped and kept."""
        if Path(dataset_root) == self._dataset_root:
            return
        self._dataset_root = Path(dataset_root)
        self._reader = None
        self._xf_cache.clear()

    def result_rows(self) -> list[dict[str, Any]]:
        """All scored/owned trials for the active model, read straight from the dataset's
        meta/episodes.jsonl (the single source of truth — no separate results.jsonl).

        Each row carries episode_index plus the eval keys the recorder embedded
        (clip_id, prompt, trial, score, max_score, milestones, note, ...). Deduped to the
        latest episode per clip_id so a re-test of the same clip shows once; rows without
        a clip_id (e.g. plain collection) are skipped.
        """
        return read_result_rows(self._dataset_root)

    def _episode_reader(self) -> EpisodeLogger | None:
        # A read-only EpisodeLogger over the recorded dataset. Construction touches
        # only meta/ (mkdir) and is cheap; reused across requests.
        if self._reader is None:
            if not self._dataset_root.exists():
                return None
            from core.app.handlers import resolve_storage

            storage = resolve_storage(self._config)
            self._reader = EpisodeLogger(
                log_dir=self._dataset_root,
                robot=self._robot,
                fps=storage.fps if storage else 30,
                dataset_keys=self._config.transport.dataset_keys,
            )
        return self._reader

    def _urdf_scene(self) -> UrdfScene | None:
        if not self._scene_tried:
            self._scene_tried = True
            try:
                from robots.utils import UrdfScene

                self._scene = UrdfScene(
                    self._robot,
                    gripper_open=self._config.robot.gripper_open,
                    gripper_close=self._config.robot.gripper_close,
                )
            except Exception as exc:
                logger.warning("URDF preview disabled: %s", exc)
                self._scene = None
        return self._scene

    def series(self, episode_index: int) -> dict[str, Any] | None:
        """Per-frame timestamp/state/action for the time-series chart + replay clock."""
        reader = self._episode_reader()
        if reader is None:
            return None
        return reader.load_episode_series(episode_index)

    def episode_index_by_clip(self) -> dict[str, int]:
        """Map clip_id -> episode_index from the recorded dataset's episodes.jsonl.

        A result row only stores clip_id; the recorder stamps both clip_id and
        episode_index onto the episode row, so this is how a result joins to its
        recorded episode (and thus to the playback endpoints).
        """
        return {
            row["clip_id"]: int(row["episode_index"])
            for row in read_result_rows(self._dataset_root)
            if row["episode_index"] is not None
        }

    def videos(self, episode_index: int) -> dict[str, Path]:
        """{video_key: mp4 path} for the episode's cameras (existing files only)."""
        reader = self._episode_reader()
        if reader is None:
            return {}
        return reader.list_episode_videos(episode_index)

    def frame_transforms(self, episode_index: int, frame: int) -> dict | None:
        """World 4x4 transforms for one replay frame: feed state[frame] to FK.

        Returns the same ``{part: {geom: 4x4}}`` shape the console's /api/scene uses,
        so the viewer's Three.js can apply matrices identically. None when out of range.
        """
        scene = self._urdf_scene()
        if scene is None:
            return None
        series = self.series(episode_index)
        if series is None:
            return None
        states = series["state"]
        if not states:
            return None
        frame = max(0, min(frame, len(states) - 1))
        return scene.transforms(states[frame])  # type: ignore[attr-defined]

    def transforms_blob(self, episode_index: int) -> bytes | None:
        """Whole-episode URDF transforms packed as an EVAXFRM1 binary blob."""
        scene = self._urdf_scene()
        if scene is None:
            return None
        series = self.series(episode_index)
        if series is None:
            return None
        states = series["state"]
        if not states:
            return None
        key = (self._dataset_root, int(episode_index), len(states))
        raw = self._xf_cache.get(key)
        if raw is None:
            raw = scene.all_transforms_blob(np.asarray(states, dtype=np.float32))  # type: ignore[attr-defined]
            if len(self._xf_cache) >= 4:
                self._xf_cache.pop(next(iter(self._xf_cache)))
            self._xf_cache[key] = raw
        return raw
