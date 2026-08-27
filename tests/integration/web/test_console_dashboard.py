from __future__ import annotations

import json
from pathlib import Path

from core.app.console.dashboard import build_dashboard, discover_raw_datasets


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))


def _dataset(
    raw: Path,
    rows: list[dict[str, object]],
    *,
    robot: str = "arx_x5",
    requirements: dict[str, int] | None = None,
) -> None:
    _write_json(raw / "meta" / "info.json", {"fps": 30, "robot_type": robot})
    _write_jsonl(raw / "meta" / "episodes.jsonl", rows)
    _write_jsonl(
        raw / "meta" / "tasks.jsonl",
        [
            {"task_index": index, "task": task, "required_episodes": required}
            for index, (task, required) in enumerate((requirements or {}).items())
        ],
    )


def test_dashboard_counts_only_explicit_raw_and_keeps_unset_targets_blank(tmp_path: Path):
    mango = "pick up the mango and place it in the plate"
    other = "place the apple in the bowl"
    raw = tmp_path / "collection" / "arx_x5_vr" / "fruit" / "raw"
    rows = [
        {
            "episode_index": 0,
            "tasks": [mango],
            "length": 300,
            "started_at": "2026-08-03T09:00:00+08:00",
            "ended_at": "2026-08-03T09:00:10+08:00",
            "session_id": "morning",
        },
        {
            "episode_index": 1,
            "tasks": [other],
            "length": 600,
            "started_at": "2026-08-04T09:00:00+08:00",
            "ended_at": "2026-08-04T09:00:20+08:00",
            "session_id": "morning",
            "quality": "red",
        },
    ]
    _dataset(raw, rows, requirements={mango: 100})

    export = raw.parent / "export" / "accepted"
    _dataset(export, rows[:1], requirements={mango: 100})

    payload = build_dashboard([tmp_path])
    collection = payload["views"]["collection"]

    assert discover_raw_datasets([tmp_path]) == [(raw.resolve(), "collection")]
    assert payload["raw_only"] is True
    assert collection["episodes"] == 2
    assert collection["valid_episodes"] == 1
    assert collection["duration_seconds"] == 30.0
    assert collection["robot_count"] == 1
    assert {row["task"]: row["required_episodes"] for row in collection["tasks"]} == {
        mango: 100,
        other: 0,
    }


def test_dashboard_date_filter_and_eval_summary(tmp_path: Path):
    raw = tmp_path / "runs" / "policy_a" / "episodes" / "raw"
    _dataset(
        raw,
        [
            {
                "episode_index": 0,
                "tasks": ["mango eval"],
                "duration_seconds": 12,
                "started_at": "2026-08-03T10:00:00+08:00",
                "ended_at": "2026-08-03T10:00:12+08:00",
                "eval_result": "failure",
            },
            {
                "episode_index": 1,
                "tasks": ["mango eval"],
                "duration_seconds": 8,
                "started_at": "2026-08-04T10:00:00+08:00",
                "ended_at": "2026-08-04T10:00:08+08:00",
                "eval_result": "success",
            },
        ],
    )

    payload = build_dashboard([tmp_path], start_date="2026-08-04", end_date="2026-08-04")
    evaluation = payload["views"]["eval"]

    assert payload["available_range"] == {"start": "2026-08-03", "end": "2026-08-04"}
    assert payload["filters"] == {"start": "2026-08-04", "end": "2026-08-04"}
    assert evaluation["episodes"] == 1
    assert evaluation["duration_seconds"] == 8.0
    assert evaluation["eval_success_rate"] == 1.0
    assert evaluation["trend"] == [
        {
            "date": "2026-08-04",
            "episodes": 1,
            "valid_episodes": 1,
            "duration_seconds": 8.0,
        }
    ]
