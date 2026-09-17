"""Exercise the /api/transfers routes of the dataset service."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tools.datasets.app import create_app
from tools.datasets.assets import ObjectCatalog

pytestmark = pytest.mark.integration

BATCH = "bench_batch"


def _task_set(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "info.yaml").write_text(
        "dataset_name: bench\nrobot_type: arx_x5\ncollection_dir: bench/raw\ntarget_episodes: 2\n",
        encoding="utf-8",
    )
    (root / "tasks.csv").write_text(
        "﻿task_id,prompt_en,total_epsiodes_count\nTASK-1,pick up cup,2\n",
        encoding="utf-8",
    )


def _client(tmp_path: Path, read_only: bool = False):
    plans = tmp_path / "task_sets"
    _task_set(plans / BATCH)
    assets = tmp_path / "assets"
    ObjectCatalog(assets, [{"object_id": "OBJ-1", "object_name": "cup"}])
    collection = tmp_path / "collection"
    (collection / "bench" / "raw" / "meta").mkdir(parents=True, exist_ok=True)
    app = create_app(plans, assets, collection, read_only=read_only)
    app.config["TESTING"] = True
    client = app.test_client()
    client.environ_base["HTTP_X_EVA_DATASET_EDITOR"] = "1"
    client.environ_base["HTTP_X_EVA_EDIT_MODE"] = "1"
    return client


def _start(client, action: str, datasets: list[str]):
    return client.post("/api/transfers", json={"action": action, "datasets": datasets})


def test_transfer_validation_rejects_bad_requests(tmp_path):
    client = _client(tmp_path)
    assert _start(client, "publish", [BATCH]).status_code == 400
    assert _start(client, "refresh", []).status_code == 400
    assert client.post("/api/transfers", json={"action": "refresh"}).status_code == 400
    assert _start(client, "refresh", ["missing_set"]).status_code == 404
    unknown_stop = client.post("/api/transfers", json={"action": "stop", "job_id": "nope"})
    assert unknown_stop.status_code == 400


def test_download_data_pulls_the_selected_set_into_its_collection_dir(tmp_path, monkeypatch):
    from tools.datasets import dataset_transfer as module

    calls = []

    def fetch_dataset(
        project_root, name, destination, storage=None, *, expected_path=None, target_dir=None
    ):
        calls.append((name, destination, expected_path, target_dir))

    monkeypatch.setattr(module, "fetch_dataset", fetch_dataset)
    client = _client(tmp_path)
    assert _start(client, "download_data", [BATCH]).status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get("/api/transfers/job").get_json()["job"]
        if job["state"] != "running":
            break
        time.sleep(0.01)
    assert job["results"] == [{"dataset": BATCH, "ok": True, "detail": ""}]
    assert calls == [
        (BATCH, tmp_path / "collection" / "bench", None, tmp_path / "collection" / "bench" / "raw")
    ]


def test_transfer_writes_require_edit_mode_and_are_read_only_safe(tmp_path):
    client = _client(tmp_path)
    del client.environ_base["HTTP_X_EVA_EDIT_MODE"]
    assert _start(client, "refresh", [BATCH]).status_code == 403
    read_only = _client(tmp_path, read_only=True)
    assert _start(read_only, "refresh", [BATCH]).status_code == 403
