"""Shared LeRobot v2.1 meta helpers used by both episode and collection writers.

Only the genuinely-common core lives here: the ``info.json`` envelope (everything
except the per-caller features/eval block) and the episodes.jsonl -> history-row
projection. Per-caller differences (robot_type, total_tasks source, the eval
section, the features schema) stay inline at each call site.
"""

from __future__ import annotations

from typing import Any

_CODEBASE_VERSION = "v2.1"
_CHUNKS_SIZE = 1000
_DATA_PATH_TPL = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_VIDEO_PATH_TPL = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_STATE_FIELDS = ("robot_name", "task_name", "split", "scene_index", "seed", "scene")


def build_info(
    *,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    fps: float,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the common ``meta/info.json`` envelope shared by all writers.

    Callers pass already-resolved scalars (robot_type, totals, features) and add
    any per-caller section (e.g. the eval block) to the returned dict themselves.
    """
    return {
        "codebase_version": _CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": _CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": _DATA_PATH_TPL,
        "video_path": _VIDEO_PATH_TPL,
        "features": features,
    }


def history_row(row: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    """Project one episodes.jsonl row into the web-console history summary shape.

    Args:
        row: A parsed episodes.jsonl record.
        fallback_index: episode_index to use when the row omits one.

    Returns:
        The summary dict the console renders (status always "saved" on disk).
    """
    quality_issues = row.get("quality_issues", [])
    return {
        "episode_index": int(row.get("episode_index", fallback_index)),
        "length": int(row.get("length", 0)),
        "status": "saved",
        "quality": row.get("quality", "green"),
        "state_only": bool(row.get("state_only", False)),
        "qc_verdict": row.get("qc_verdict", ""),
        "qc_note": row.get("qc_note", ""),
        "quality_issues": [] if quality_issues is None else quality_issues,
        "error": "",
    }


def build_state(robot_type: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build simulator reconstruction metadata alongside a LeRobot v2.1 dataset.

    Args:
        robot_type: EVAClient robot identifier used when an episode has no simulator name.
        episodes: Rows from ``meta/episodes.jsonl``.

    Returns:
        Dataset-level robot identity and per-episode reconstruction fields.
    """
    episode_states = []
    for row in episodes:
        state = {"episode_index": int(row["episode_index"])}
        state.update({key: row[key] for key in _STATE_FIELDS if row.get(key) is not None})
        episode_states.append(state)
    return {
        "format_version": 2,
        "robot_type": robot_type,
        "episodes": episode_states,
    }
