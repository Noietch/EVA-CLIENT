"""Exercise the task-set transfer engine behind the dataset manager card."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.datasets.collection import PlanCatalog
from tools.datasets.dataset_transfer import DatasetTransfer

pytestmark = pytest.mark.unit

BATCH = "bench_batch"


def _catalog(tmp_path: Path) -> PlanCatalog:
    plans = tmp_path / "task_sets"
    root = plans / BATCH
    root.mkdir(parents=True)
    (root / "info.yaml").write_text(
        "dataset_name: bench\nrobot_type: arx_x5\ncollection_dir: bench/raw\ntarget_episodes: 2\n",
        encoding="utf-8",
    )
    (root / "tasks.csv").write_text(
        "﻿task_id,prompt_en,total_epsiodes_count,scene_ids,scene_epsiodes_count\n"
        "TASK-1,pick up cup,2,SC-1,2\n",
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "objects.csv").write_text("﻿object_id,object_name\nOBJ-1,cup\n", encoding="utf-8")
    collection = tmp_path / "collection"
    (collection / "bench" / "raw" / "meta").mkdir(parents=True)
    return PlanCatalog(plans, assets, collection)


def _wait_job(transfer: DatasetTransfer, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = next(j for j in transfer.snapshot()["jobs"] if j["id"] == job_id)
        if snapshot["state"] != "running":
            return snapshot
        threading.Event().wait(0.01)
    pytest.fail("Job did not finish")


def test_inspect_resolves_the_remote_prefix_when_collection_dir_is_local(monkeypatch, tmp_path):
    import huggingface_hub

    transfer = DatasetTransfer(tmp_path, _catalog(tmp_path))
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text('{"episode_index": 0}\n{"episode_index": 1}\n', encoding="utf-8")
    listings = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixed-revision")

        def list_repo_files(self, repo, **kwargs):
            return [f"datasets/{BATCH}/meta/episodes.jsonl"]

        def list_repo_tree(self, repo, path_in_repo, **kwargs):
            listings.append(path_in_repo)
            if path_in_repo == f"datasets/{BATCH}":
                return [
                    SimpleNamespace(
                        path=f"{path_in_repo}/meta/episodes.jsonl", blob_id="b", size=10, lfs=None
                    )
                ]
            return []

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(episodes))
    result = transfer._inspect(transfer._target(BATCH))
    assert result["episodes"] == 2
    assert f"datasets/{BATCH}" in listings
    assert result["verification"]["data"]["state"] == "different"


def test_qc_upload_synthesizes_records_without_touching_source_episodes(monkeypatch, tmp_path):
    transfer = DatasetTransfer(tmp_path, _catalog(tmp_path))
    dataset = tmp_path / "collection" / "bench" / "raw"
    episodes = json.dumps({"episode_index": 3, "qc_verdict": "fail"}) + "\n"
    (dataset / "meta/episodes.jsonl").write_text(episodes, encoding="utf-8")
    published = {}

    def publish(root, path, name, expected_path=None):
        published.update(root=root, path=path, name=name, expected_path=expected_path)
        assert (path / "meta/qc.jsonl").is_file()
        return {"revision": "fixed"}

    monkeypatch.setattr("tools.datasets.dataset_transfer.publish_qc", publish)
    assert _wait_job(transfer, transfer.start("upload_qc", [BATCH]))["results"][0]["ok"]
    assert published["name"] == BATCH
    assert published["path"] != dataset
    assert (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8") == episodes
    assert not (dataset / "meta/qc.jsonl").exists()
