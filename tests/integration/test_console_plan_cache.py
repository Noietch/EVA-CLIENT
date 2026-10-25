"""Exercise plan reuse and immediate file invalidation through the HTTP API."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.parse import urlencode

import pytest

from core.app.console import server
from core.config import _normalize_collection_task_set
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def test_scalar_task_set_keeps_configured_dataset_alias(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    (root / "tasks.csv").write_text("task_id,prompt_en,total_epsiodes_count\nTASK-1,put cup,1\n")
    config = console_config(collection={"task_set_dir": str(root), "task_set_name": "alias"})
    _normalize_collection_task_set(config)
    with serve_console(config) as console:
        response = console.get("/api/collection_slots?dataset=alias&all=1")
    assert response.status == 200
    assert response.json["dataset"] == "alias"
    assert response.json["counts"]["total"] == 1


def test_concurrent_plan_loads_reuse_one_version(tmp_path):
    (tmp_path / "tasks.csv").write_text(
        "task_id,prompt_en,total_epsiodes_count\nTASK-1,put cup,1\n"
    )
    config = console_config(collection={"task_set_dir": [str(tmp_path)]})
    with (
        patch.object(server, "_build_scene_plan", wraps=server._build_scene_plan) as build,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        plans = list(
            executor.map(lambda _: server._load_scene_plan(config, tmp_path.name), range(2))
        )
    assert build.call_count == 1
    assert plans[0] is plans[1]


def test_slot_pages_reuse_plan_and_reload_changed_task_counts(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    path = root / "tasks.csv"
    (root / "scene.csv").write_text("scene_id,placements\nSC-1,[]\n")
    header = "task_id,prompt_en,scene_ids,scene_epsiodes_count,total_epsiodes_count\n"
    path.write_text(header + "TASK-1,put cup,SC-1,120,120\n")
    config = console_config(collection={"task_set_dir": [str(root)]})
    _normalize_collection_task_set(config)
    with serve_console(config) as console:
        first = console.get("/api/collection_slots?dataset=batch&page=1&all=1")
        assert first.status == 200
        assert first.json["counts"]["total"] == 120
        assert len(first.json["slots"]) == 50
        version = first.json["scene_plan_version"]
        query = urlencode({"dataset": "batch", "page": 2, "all": 1, "scene_plan_version": version})
        second = console.get(f"/api/collection_slots?{query}")
        assert second.status == 200
        assert second.json["page"] == 2
        assert second.json["scene_plan_version"] == version
        assert second.json["slots"][0]["ordinal"] == 50
        assert not {"scene_plan", "scenes", "tasks"} & second.json.keys()

        path.write_text(header + "TASK-1,put cup,SC-1,121,121\n")
        refreshed = console.get(f"/api/collection_slots?{query}")
        assert refreshed.json["scene_plan_version"] != version
        assert refreshed.json["counts"]["total"] == 121
        assert {"scene_plan", "scenes", "tasks"} <= refreshed.json.keys()


def test_unconfigured_collection_plan_does_not_use_work_dirs_or_environment(tmp_path, monkeypatch):
    work = tmp_path / "work_dirs"
    work.mkdir()
    (work / "tasks.csv").write_text("task_id,prompt_en\nTASK-1,put cup\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVA_SCENE_PLAN_DIR", str(work))
    config = console_config()
    with serve_console(config) as console:
        response = console.get("/api/scene_plan")
    assert response.status == 400
    assert "not configured" in response.json["error"]


def test_explicit_collection_roots_override_legacy_environment(tmp_path, monkeypatch):
    root = tmp_path / "configured"
    root.mkdir()
    (root / "tasks.csv").write_text("task_id,prompt_en\nTASK-1,put cup\n")
    monkeypatch.setenv("EVA_SCENE_PLAN_DIR", str(tmp_path / "other"))
    config = console_config(collection={"task_set_dir": str(root)})
    with serve_console(config) as console:
        response = console.get("/api/scene_plan?set=configured")
    assert response.status == 200
    assert response.json["task_set"] == "configured"
