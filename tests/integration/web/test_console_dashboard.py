from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.app.console.dashboard import (
    build_dashboard,
)
from core.utils.dataset_upload import record_dataset_upload_receipt

pytestmark = pytest.mark.integration


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


def test_dashboard_reports_uploaded_episodes_for_all_data_and_each_day(tmp_path: Path):
    raw = tmp_path / "collection" / "cup_set" / "raw"
    _dataset(
        raw,
        [
            {
                "episode_index": 0,
                "tasks": ["cup"],
                "length": 30,
                "started_at": "2026-08-03T10:00:00+08:00",
            },
            {
                "episode_index": 1,
                "tasks": ["cup"],
                "length": 60,
                "started_at": "2026-08-04T10:00:00+08:00",
            },
        ],
    )
    accepted = raw.parent / "export" / "lerobot_v21" / "accepted"
    _write_json(
        accepted / "meta" / "quality_split.json",
        {
            "subset": "accepted",
            "source_dir": str(raw),
            "source_episode_indices": [1],
            "dataset_format": "lerobot_v21",
        },
    )
    record_dataset_upload_receipt(
        accepted,
        destination="robot@host",
        remote_dir="/datasets/cup_set",
    )

    collection = build_dashboard([tmp_path])["views"]["collection"]

    assert collection["uploaded_episodes"] == 1
    assert collection["not_uploaded_episodes"] == 1
    by_date = {row["date"]: row for row in collection["daily"]}
    assert by_date["2026-08-03"]["uploaded_episodes"] == 0
    assert by_date["2026-08-03"]["not_uploaded_episodes"] == 1
    assert by_date["2026-08-04"]["uploaded_episodes"] == 1
    assert by_date["2026-08-04"]["frames"] == 60
