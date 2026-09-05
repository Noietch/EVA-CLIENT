"""HTTP contract tests for history split from the high-frequency status poll."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pytest

from tests.integration.web._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def _write_history(dataset_dir, count: int) -> None:
    path = dataset_dir / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["task"], "length": index + 1}) + "\n"
            for index in range(count)
        )
    )


class _CollectionHistoryLogger:
    def __init__(
        self,
        dataset_dir: Path,
        queue: list[dict[str, object]],
        *,
        expected_task: str | None = None,
        expected_collection_dataset: str | None = None,
    ) -> None:
        self._dataset_dir = dataset_dir
        self._queue = queue
        self._expected_task = expected_task
        self._expected_collection_dataset = expected_collection_dataset

    def status_snapshot(
        self, task: str, *, include_history: bool = True, collection_dataset: str | None = None
    ) -> dict[str, object]:
        assert include_history is False
        if self._expected_task is not None:
            assert task == self._expected_task
        if self._expected_collection_dataset is not None:
            assert collection_dataset == self._expected_collection_dataset
        return {"dataset_dir": str(self._dataset_dir), "queue": self._queue}


def test_status_excludes_rollout_history_and_history_endpoint_pages(tmp_path):
    config = console_config(
        rollout={
            "storage": {"enabled": True, "log_dir": str(tmp_path)},
        }
    )
    _write_history(tmp_path, 3)

    with serve_console(config) as console:
        status = console.get("/api/status").json
        assert "episodes" not in status["rollout"]
        assert status["rollout"]["completed_episodes"] == 3
        assert status["rollout"]["progress"] == 1.0

        first = console.get("/api/episodes?scope=rollout&limit=2").json
        assert first["ok"] is True
        assert first["scope"] == "rollout"
        assert [row["episode_index"] for row in first["episodes"]] == [0, 1]
        assert first["total"] == 3
        assert first["next_since"] == 2
        assert first["has_more"] is True
        assert first["cursor"]
        assert first["reset"] is False

        second_query = urlencode(
            {
                "scope": "rollout",
                "since": first["next_since"],
                "limit": 2,
                "cursor": first["cursor"],
            }
        )
        second = console.get(f"/api/episodes?{second_query}").json
        assert [row["episode_index"] for row in second["episodes"]] == [2]
        assert second["has_more"] is False
        assert second["reset"] is False


def test_collection_history_is_scoped_to_set_and_prompt_before_counting(tmp_path):
    first_dir = tmp_path / "first" / "raw"
    second_dir = tmp_path / "second" / "raw"
    _write_history(first_dir, 1)
    path = second_dir / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"episode_index": 4, "tasks": ["shared prompt"], "length": 5},
        {"episode_index": 8, "tasks": ["other prompt"], "length": 9},
        {"episode_index": 11, "tasks": ["shared prompt"], "length": 12},
        {"episode_index": 12, "tasks": ["shared prompt"], "length": 13},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    config = console_config(
        collection={
            "tasks": {
                "first": [("shared prompt", 1)],
                "second": [("shared prompt", 2), ("other prompt", 1)],
            }
        }
    )
    query = urlencode({"scope": "collect", "set": "second", "task": "shared prompt"})
    with serve_console(config) as console:
        console.runtime.episode_logger = _CollectionHistoryLogger(
            second_dir,
            [
                {"episode_index": 12, "task": "shared prompt", "status": "saving"},
                {"episode_index": 13, "task": "other prompt", "status": "queued"},
            ],
            expected_task="shared prompt",
            expected_collection_dataset="second",
        )
        response = console.get(f"/api/episodes?{query}")

    assert response.status == 200
    assert response.json["set"] == "second"
    assert response.json["task"] == "shared prompt"
    assert response.json["dataset_dir"] == str(second_dir)
    assert response.json["total"] == 2
    assert [row["episode_index"] for row in response.json["episodes"]] == [4, 11]
    assert [row["episode_index"] for row in response.json["queue"]] == [12]


def test_explicit_history_directory_does_not_reuse_another_dataset_queue(tmp_path):
    active_dir = tmp_path / "active" / "raw"
    explicit_dir = tmp_path / "explicit" / "raw"
    _write_history(active_dir, 2)
    _write_history(explicit_dir, 2)

    base_query = {"scope": "collect", "set": "set", "task": "task"}
    with serve_console(console_config()) as console:
        console.runtime.episode_logger = _CollectionHistoryLogger(
            active_dir,
            [{"episode_index": 0, "task": "task", "status": "saving"}],
        )
        active = console.get(f"/api/episodes?{urlencode(base_query)}").json
        explicit = console.get(
            f"/api/episodes?{urlencode({**base_query, 'dataset_dir': str(explicit_dir)})}"
        ).json

    assert [row["episode_index"] for row in active["episodes"]] == [1]
    assert [row["episode_index"] for row in active["queue"]] == [0]
    assert [row["episode_index"] for row in explicit["episodes"]] == [0, 1]
    assert explicit["queue"] == []
    assert explicit["total"] == 2
