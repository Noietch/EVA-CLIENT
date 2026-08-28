"""Raw-dataset aggregation for the collection and evaluation dashboard."""

from __future__ import annotations

import datetime as dt
import json
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_DATASET_CACHE_MAX = 32
_DATASET_CACHE_LOCK = threading.RLock()
_DATASET_CACHE: OrderedDict[
    tuple[str, str, tuple[tuple[str, int, int, int], ...]],
    tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]],
] = OrderedDict()


def _file_signature(path: Path) -> tuple[str, int, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0, 0)
    return (str(path), int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _parse_time(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    return parsed


def _iso(value: dt.datetime | None) -> str:
    return value.isoformat(timespec="seconds") if value is not None else ""


def _duration(row: dict[str, Any], fps: float) -> float:
    explicit = row.get("duration_seconds")
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    row_fps = row.get("alignment_fps") or row.get("collection_fps") or fps or 30.0
    try:
        return max(0.0, float(row.get("length", 0))) / max(float(row_fps), 0.001)
    except (TypeError, ValueError):
        return 0.0


def _dataset_mode(raw_dir: Path) -> str | None:
    if raw_dir.parent.name == "episodes":
        return "eval"
    if "collection" in raw_dir.parts:
        return "collection"
    return None


def _dataset_name(raw_dir: Path, mode: str) -> str:
    if mode == "eval" and raw_dir.parent.name == "episodes":
        return raw_dir.parent.parent.name
    return raw_dir.parent.name


def discover_raw_datasets(roots: Iterable[Path]) -> list[tuple[Path, str]]:
    """Find explicit ``raw`` LeRobot datasets, never export derivatives."""
    found: dict[Path, str] = {}
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.name == "raw" else []
        candidates.extend(path.parent.parent for path in root.rglob("raw/meta/episodes.jsonl"))
        for raw_dir in candidates:
            resolved = raw_dir.resolve()
            mode = _dataset_mode(resolved)
            if mode is not None and (resolved / "meta" / "episodes.jsonl").is_file():
                found[resolved] = mode
    return sorted(found.items(), key=lambda item: str(item[0]))


def _episode_record(
    row: dict[str, Any],
    raw_dir: Path,
    source_mode: str,
    info: dict[str, Any],
    requirements: dict[str, int],
) -> dict[str, Any] | None:
    mode = str(row.get("mode") or source_mode).lower()
    if mode not in {"collection", "eval"}:
        return None
    fps = float(info.get("fps") or 30.0)
    duration = _duration(row, fps)
    started = _parse_time(row.get("started_at") or row.get("recorded_at"))
    ended = _parse_time(row.get("ended_at"))
    if started is not None and ended is None:
        ended = started + dt.timedelta(seconds=duration)
    task_values = row.get("tasks") or []
    task = str(row.get("prompt") or (task_values[0] if task_values else ""))
    quality = str(row.get("quality") or "green").lower()
    qc_verdict = str(row.get("qc_verdict") or "").lower()
    status = str(row.get("status") or "valid").lower()
    valid = (
        quality != "red"
        and qc_verdict != "fail"
        and status
        not in {
            "invalid",
            "cancelled",
            "failed",
        }
    )
    robot_id = str(row.get("robot_id") or info.get("robot_type") or "unknown")
    result = row.get("eval_result")
    if result is None and mode == "eval" and row.get("score") is not None:
        try:
            result = (
                "success"
                if float(row["score"]) >= float(row.get("max_score", row["score"]))
                else "failure"
            )
        except (TypeError, ValueError):
            result = ""
    return {
        "episode_index": int(row.get("episode_index", 0)),
        "mode": mode,
        "dataset": _dataset_name(raw_dir, source_mode),
        "dataset_dir": str(raw_dir),
        "task": task,
        "required_episodes": requirements.get(task, 0),
        "robot_id": robot_id,
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "duration_seconds": round(duration, 3),
        "session_id": str(row.get("session_id") or ""),
        "valid": valid,
        "quality": quality,
        "result": str(result or ""),
    }


def _load_dataset_records(
    raw_dir: Path, mode: str
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    """Read one dataset's metadata once, invalidating on metadata file changes."""
    paths = (
        raw_dir / "meta" / "info.json",
        raw_dir / "meta" / "tasks.jsonl",
        raw_dir / "meta" / "episodes.jsonl",
    )
    signatures = tuple(_file_signature(path) for path in paths)
    key = (str(raw_dir.resolve()), mode, signatures)
    with _DATASET_CACHE_LOCK:
        cached = _DATASET_CACHE.get(key)
        if cached is not None:
            _DATASET_CACHE.move_to_end(key)
            return cached

    info = _read_json(paths[0])
    requirements = {
        str(row.get("task") or ""): int(row.get("required_episodes") or 0)
        for row in _read_jsonl(paths[1])
        if row.get("task")
    }
    records = []
    for row in _read_jsonl(paths[2]):
        record = _episode_record(row, raw_dir, mode, info, requirements)
        if record is not None:
            records.append(record)
    value = (info, requirements, records)
    with _DATASET_CACHE_LOCK:
        existing = _DATASET_CACHE.get(key)
        if existing is not None:
            _DATASET_CACHE.move_to_end(key)
            return existing
        _DATASET_CACHE[key] = value
        while len(_DATASET_CACHE) > _DATASET_CACHE_MAX:
            _DATASET_CACHE.popitem(last=False)
    return value


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(float(row["duration_seconds"]) for row in episodes)
    valid_rows = [row for row in episodes if row["valid"]]
    valid_duration = sum(float(row["duration_seconds"]) for row in valid_rows)
    robots = sorted({str(row["robot_id"]) for row in episodes if row["robot_id"]})

    sessions: dict[tuple[str, str, str], list[tuple[dt.datetime, dt.datetime]]] = defaultdict(list)
    for row in episodes:
        started = _parse_time(row["started_at"])
        ended = _parse_time(row["ended_at"])
        if started is None or ended is None or ended < started:
            continue
        fallback = f"{row['dataset']}:{started.date().isoformat()}"
        key = (str(row["mode"]), str(row["robot_id"]), str(row["session_id"] or fallback))
        sessions[key].append((started, ended))
    active_span = sum(
        (max(end for _, end in values) - min(start for start, _ in values)).total_seconds()
        for values in sessions.values()
    )

    by_day: dict[str, dict[str, Any]] = {}
    for row in episodes:
        started = _parse_time(row["started_at"])
        day = started.date().isoformat() if started is not None else "unknown"
        bucket = by_day.setdefault(
            day,
            {"date": day, "episodes": 0, "valid_episodes": 0, "duration_seconds": 0.0},
        )
        bucket["episodes"] += 1
        bucket["valid_episodes"] += int(bool(row["valid"]))
        bucket["duration_seconds"] += float(row["duration_seconds"])
    trend = []
    for day in sorted(by_day):
        bucket = by_day[day]
        bucket["duration_seconds"] = round(bucket["duration_seconds"], 3)
        trend.append(bucket)

    eval_rows = [row for row in episodes if row["mode"] == "eval" and row["result"]]
    eval_success = sum(row["result"] == "success" for row in eval_rows)
    task_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in episodes:
        key = (str(row["robot_id"]), str(row["dataset"]), str(row["task"]))
        task = task_groups.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "robot_id": row["robot_id"],
                "episodes": 0,
                "valid_episodes": 0,
                "required_episodes": int(row.get("required_episodes") or 0),
            },
        )
        task["episodes"] += 1
        task["valid_episodes"] += int(bool(row["valid"]))
        task["required_episodes"] = max(
            int(task["required_episodes"]), int(row.get("required_episodes") or 0)
        )
    tasks = sorted(task_groups.values(), key=lambda row: (-row["episodes"], row["task"]))
    recent = sorted(episodes, key=lambda row: row["started_at"], reverse=True)[:100]
    return {
        "episodes": len(episodes),
        "valid_episodes": len(valid_rows),
        "duration_seconds": round(duration, 3),
        "average_duration_seconds": round(duration / len(episodes), 3) if episodes else 0.0,
        "robots": robots,
        "robot_count": len(robots),
        "valid_rate": round(len(valid_rows) / len(episodes), 4) if episodes else 0.0,
        "efficiency": round(valid_duration / active_span, 4) if active_span > 0 else None,
        "active_span_seconds": round(active_span, 3),
        "eval_success_rate": round(eval_success / len(eval_rows), 4) if eval_rows else None,
        "tasks": tasks,
        "trend": trend,
        "recent": recent,
    }


def _date_filter(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def build_dashboard(
    roots: Iterable[Path], *, start_date: str | None = None, end_date: str | None = None
) -> dict[str, Any]:
    """Aggregate collection/eval metrics from explicit raw datasets under roots."""
    episodes: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for raw_dir, mode in discover_raw_datasets(roots):
        _, _, records = _load_dataset_records(raw_dir, mode)
        episodes.extend(records)
        sources.append(
            {
                "dataset": _dataset_name(raw_dir, mode),
                "dataset_dir": str(raw_dir),
                "mode": mode,
                "episodes": len(records),
            }
        )
    dated = [
        parsed.date() for row in episodes if (parsed := _parse_time(row["started_at"])) is not None
    ]
    available_range = {
        "start": min(dated).isoformat() if dated else "",
        "end": max(dated).isoformat() if dated else "",
    }
    start = _date_filter(start_date)
    end = _date_filter(end_date)
    if start is not None or end is not None:
        episodes = [
            row
            for row in episodes
            if (parsed := _parse_time(row["started_at"])) is not None
            and (start is None or parsed.date() >= start)
            and (end is None or parsed.date() <= end)
        ]
    return {
        "raw_only": True,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "available_range": available_range,
        "filters": {
            "start": start.isoformat() if start else "",
            "end": end.isoformat() if end else "",
        },
        "sources": sources,
        "views": {
            "all": _summary(episodes),
            "collection": _summary([row for row in episodes if row["mode"] == "collection"]),
            "eval": _summary([row for row in episodes if row["mode"] == "eval"]),
        },
    }
