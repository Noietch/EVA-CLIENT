"""HTTP contract tests for history split from the high-frequency status poll."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pytest

from core.config import _normalize_collection_task_set
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def test_collection_history_survives_description_change_with_task_identity(tmp_path):
    plan = tmp_path / "pour"
    plan.mkdir()
    header = "task_id,prompt_en,prompt_zh,total_epsiodes_count,scene_ids,scene_epsiodes_count\n"
    (plan / "tasks.csv").write_text(header + "TASK-1,old wording,旧描述,2,SC-1,2\n")
    config = console_config(collection={"task_set_dir": [str(plan)]})
    _normalize_collection_task_set(config)
    # This is a running process with the old prompt; the scene plan reloads new text.
    (plan / "tasks.csv").write_text(header + "TASK-1,new wording,新描述,2,SC-1,2\n")
    dataset_dir = tmp_path / "raw"
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    rows = [
        {"episode_index": 0, "tasks": ["historical wording"], "task_id": "TASK-1", "length": 5},
        {"episode_index": 1, "tasks": ["old wording"], "task_id": "TASK-OTHER", "length": 5},
    ]
    history_path = meta / "episodes.jsonl"
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = history_path.read_bytes()
    with serve_console(config) as console:
        console.runtime.episode_logger = _CollectionHistoryLogger(
            dataset_dir,
            [{"episode_index": 2, "task": "historical wording", "task_id": "TASK-1"}],
            expected_task="old wording",
            expected_collection_dataset="pour",
        )
        scene_plan = console.get("/api/scene_plan?set=pour").json
        assert scene_plan["tasks"][0]["prompt_zh"] == "新描述"
        assert scene_plan["tasks"][0]["runtime_prompt"] == "old wording"
        query = urlencode({"scope": "collect", "set": "pour", "task": "old wording", "limit": 1})
        response = console.get(f"/api/episodes?{query}")
    assert response.status == 200
    assert response.json["total"] == 1
    assert [row["episode_index"] for row in response.json["episodes"]] == [0]
    assert [row["episode_index"] for row in response.json["queue"]] == [2]
    assert history_path.read_bytes() == original


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
