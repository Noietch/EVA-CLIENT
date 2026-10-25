"""Scene identity filters apply to raw episode metadata without mixing workspaces."""

import json

import pytest

from core.app.console.dashboard import build_dashboard
from core.config import ConfigDict
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def _write_episodes(root, rows):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"fps": 30, "robot_type": "test-robot"}))
    (meta / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": index,
                    "length": 30,
                    "started_at": "2026-10-24T09:00:00+08:00",
                    "tasks": ["pick up the cup"],
                    **row,
                }
            )
            + "\n"
            for index, row in enumerate(rows)
        )
    )


@pytest.fixture
def raw_scene_datasets(tmp_path):
    _write_episodes(
        tmp_path / "datasets/collect-set",
        [
            {"task_id": "T1", "scene_id": "S1"},
            {"task_id": "T1", "scene_id": "S2"},
            {"task_id": "T2", "scene_id": "S1"},
            {"scene_id": "S1"},
            {"task_set": "different-set", "task_id": "T1", "scene_id": "S1"},
        ],
    )
    _write_episodes(
        tmp_path / "models/model-a/episodes/raw",
        [
            {"task_set": "eval-set", "task_id": "T1", "scene_id": "S1", "score": 1},
            {"task_set": "eval-set", "task_id": "T1", "scene_id": "S2", "score": 0},
            {"task_set": "other-set", "task_id": "T1", "scene_id": "S1"},
            {"scene_id": "S1"},
            {
                "task_set": "eval-set",
                "task_id": "T1",
                "scene_id": "S1",
                "started_at": "2026-10-23T09:00:00+08:00",
                "score": 0,
            },
        ],
    )
    return tmp_path


@pytest.mark.parametrize(
    ("scope", "task_set", "expected", "other_mode"),
    [("collect", "collect-set", 1, "eval"), ("eval", "eval-set", 2, "collection")],
)
def test_scene_filter_uses_explicit_identity_and_preserves_other_workspace(
    raw_scene_datasets, scope, task_set, expected, other_mode
):
    payload = build_dashboard(
        [raw_scene_datasets],
        scene_scope=scope,
        task_set=task_set,
        task_id="T1",
        scene_id="S1",
    )
    mode = "collection" if scope == "collect" else "eval"
    view = payload["views"][mode]
    assert view["episodes"] == expected
    assert view["frames"] == expected * 30
    assert {row["task_set"] for row in view["recent"]} == {task_set}
    assert {row["task_id"] for row in view["recent"]} == {"T1"}
    assert {row["scene_id"] for row in view["recent"]} == {"S1"}
    assert payload["views"][other_mode]["episodes"] == 5
    assert payload["views"]["all"]["episodes"] == expected + 5
    source = next(row for row in payload["sources"] if row["mode"] == mode)
    assert source["episodes"] == expected


def test_scene_and_date_filters_intersect_and_legacy_records_are_not_inferred(raw_scene_datasets):
    roots = [raw_scene_datasets]
    unfiltered = build_dashboard(roots)
    assert unfiltered["views"]["all"]["episodes"] == 10
    selected = build_dashboard(
        roots,
        scene_scope="eval",
        task_set="eval-set",
        task_id="T1",
        scene_id="S1",
        start_date="2026-10-24",
        end_date="2026-10-24",
    )
    assert selected["views"]["eval"]["episodes"] == 1
    assert selected["views"]["eval"]["eval_success_rate"] == 1
    assert selected["views"]["eval"]["daily"][0]["episodes"] == 1
    assert selected["filters"] == {
        "start": "2026-10-24",
        "end": "2026-10-24",
        "scene_scope": "eval",
        "task_set": "eval-set",
        "task_id": "T1",
        "scene_id": "S1",
    }
    missing = build_dashboard(roots, scene_scope="eval", task_set="model-a", scene_id="S1")
    assert missing["views"]["eval"]["episodes"] == 0
    legacy = build_dashboard(roots, scene_scope="collect", task_set="collect-set", scene_id="S1")
    assert legacy["views"]["collection"]["episodes"] == 3


def test_dashboard_http_forwards_scene_identity_filters(raw_scene_datasets):
    config = console_config()
    config.dashboard = ConfigDict(raw_roots_only=True, raw_roots=[str(raw_scene_datasets)])
    with serve_console(config) as harness:
        response = harness.get(
            "/api/dashboard?scene_scope=eval&task_set=eval-set&task_id=T1&scene_id=S1"
            "&from=2026-10-24&to=2026-10-24"
        )
        assert response.status == 200
        assert response.json["views"]["eval"]["episodes"] == 1
        assert response.json["views"]["collection"]["episodes"] == 5
        assert response.json["views"]["eval"]["recent"][0]["scene_id"] == "S1"
