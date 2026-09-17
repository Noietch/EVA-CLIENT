"""Exercise the task-set transfer engine behind the dataset manager card."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.datasets.collection import PlanCatalog
from tools.datasets.dataset_transfer import CatalogTransfer, DatasetTransfer

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

    transfer = CatalogTransfer(tmp_path, _catalog(tmp_path))
    target = transfer.targets([BATCH])[0]
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
    result = transfer._inspect(target, None)
    assert result["episodes"] == 2
    assert f"datasets/{BATCH}" in listings
    assert result["verification"]["data"]["state"] == "different"


def test_qc_upload_sends_the_local_ledger_and_requires_one(monkeypatch, tmp_path):
    transfer = DatasetTransfer(tmp_path)
    dataset = tmp_path / "collection" / "bench" / "raw"
    (dataset / "meta").mkdir(parents=True)
    episodes = json.dumps({"episode_index": 3, "qc_verdict": "fail"}) + "\n"
    (dataset / "meta/episodes.jsonl").write_text(episodes, encoding="utf-8")
    target = {
        "name": BATCH,
        "dataset_dir": dataset,
        "plan": tmp_path / BATCH,
        "remote_path": None,
    }
    published = []

    def publish(root, path, name, storage=None, expected_path=None):
        published.append(path)
        return {"revision": "fixed"}

    monkeypatch.setattr("tools.datasets.dataset_transfer.publish_qc", publish)
    failed = _wait_job(transfer, transfer.start("upload_qc", [target]))["results"][0]
    assert failed["ok"] is False
    assert "No QC ledger" in failed["error"]
    assert published == []

    (dataset / "meta/qc.jsonl").write_text(episodes, encoding="utf-8")
    assert _wait_job(transfer, transfer.start("upload_qc", [target]))["results"][0]["ok"]
    assert published == [dataset]
    assert (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8") == episodes


def test_data_download_targets_the_local_dataset_and_holds_the_upload_off(monkeypatch, tmp_path):
    from tools.datasets import dataset_transfer as module

    dataset = tmp_path / "collection" / "bench" / "raw"
    (dataset / "meta").mkdir(parents=True)
    target = {
        "name": BATCH,
        "dataset_dir": dataset,
        "plan": tmp_path / BATCH,
        "remote_path": "datasets/real_robot/dual_yam/bench_batch",
    }
    pulled = {}
    started = threading.Event()
    release = threading.Event()

    def fetch_dataset(project_root, name, destination, storage=None, *, expected_path=None):
        pulled.update(name=name, destination=destination, remote_path=expected_path)
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(module, "fetch_dataset", fetch_dataset)
    monkeypatch.setattr(module, "publish_dataset", lambda *args, **kwargs: None)
    transfer = DatasetTransfer(tmp_path)
    download_id = transfer.start("download_data", [target])
    assert started.wait(5)
    assert pulled == {
        "name": BATCH,
        "destination": dataset.parent,
        "remote_path": target["remote_path"],
    }
    # The download rewrites the dataset and its ledger, so an upload of the
    # same set waits instead of racing it.
    upload_id = transfer.start("upload_data", [target])
    deadline = time.monotonic() + 3
    while not transfer.jobs[upload_id]["waiting"] and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert transfer.jobs[upload_id]["waiting_resources"] == ["local_data", "remote_data"]
    release.set()
    assert _wait_job(transfer, download_id)["results"][0]["ok"]
    assert _wait_job(transfer, upload_id)["results"][0]["ok"]


def test_cloud_refresh_inspects_datasets_in_parallel(monkeypatch, tmp_path):
    from tools.datasets import dataset_transfer as module

    monkeypatch.setattr(module, "PARALLEL_WORKERS", 3)
    # A sequential refresh never gathers three inspections at once.
    overlapping = threading.Barrier(3, timeout=5)
    transfer = DatasetTransfer(tmp_path)

    def inspect(target, storage, progress):
        overlapping.wait()
        return {"checked_at": 0.0, "verification": {"data": {"state": "same"}}}

    monkeypatch.setattr(transfer, "_inspect", inspect)
    targets = [
        {
            "name": name,
            "plan": tmp_path / name,
            "dataset_dir": tmp_path / name,
            "remote_path": f"datasets/real_robot/dual_yam/{name}",
        }
        for name in ("set_a", "set_b", "set_c")
    ]
    job = _wait_job(transfer, transfer.start("refresh", targets))
    assert [result["ok"] for result in job["results"]] == [True, True, True]
    assert sorted(result["dataset"] for result in job["results"]) == ["set_a", "set_b", "set_c"]


def test_jobs_run_in_parallel_and_cancel_only_the_conflicting_waiter(monkeypatch, tmp_path):
    from tools.datasets import dataset_transfer as module

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def publish(root, path, storage):
        calls.append(path.name)
        if path.name == "first":
            started.set()
            assert release.wait(5)
            raise ValueError("offline")
        finished.set()

    monkeypatch.setattr(module, "publish_task_set", publish)
    transfer = DatasetTransfer(tmp_path)
    targets = [{"name": name, "plan": tmp_path / name} for name in ("first", "second")]
    first_id = transfer.start("upload_task", targets)
    assert started.wait(5)
    parallel_id = transfer.start("upload_task", [{"name": "third", "plan": tmp_path / "third"}])
    assert _wait_job(transfer, parallel_id)["state"] == "done"
    waiting_id = transfer.start("upload_task", targets[:1])
    transfer.stop(waiting_id)
    assert _wait_job(transfer, waiting_id)["state"] == "stopped"
    assert calls == ["first", "third"]
    release.set()
    assert finished.wait(5)
    first = _wait_job(transfer, first_id)
    assert calls == ["first", "third", "second"]
    assert first["results"][0]["error"] == "offline"
    assert first["results"][1]["ok"]


def test_same_dataset_disjoint_uploads_and_remote_reads_run_together(monkeypatch, tmp_path):
    from tools.datasets import dataset_transfer as module

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/qc.jsonl").write_text("{}\n")
    started = {name: threading.Event() for name in ("data", "qc", "task")}
    release = threading.Event()

    def upload(kind):
        def run(*args, **kwargs):
            if kind == "data":
                assert kwargs["include_qc"] is False
            started[kind].set()
            assert release.wait(5)

        return run

    monkeypatch.setattr(module, "publish_dataset", upload("data"))
    monkeypatch.setattr(module, "publish_qc", upload("qc"))
    monkeypatch.setattr(module, "publish_task_set", upload("task"))
    transfer = DatasetTransfer(tmp_path)
    monkeypatch.setattr(
        transfer,
        "_inspect",
        lambda *args: {"revision": "snapshot", "verification": {"data": {"state": "same"}}},
    )
    target = {
        "name": "fold",
        "dataset_dir": tmp_path,
        "plan": tmp_path,
        "remote_path": "datasets/fold",
    }
    ids = []
    try:
        for kind in started:
            ids.append(transfer.start(f"upload_{kind}", [target]))
        assert all(event.wait(3) for event in started.values())
        for action in ("refresh", "verify"):
            read_id = transfer.start(action, [target])
            assert _wait_job(transfer, read_id)["results"][0]["ok"]
            cloud = transfer.snapshot()["remote"]["fold"]
            assert cloud["stale"] is True
            assert "verification" not in cloud
        blocked_id = transfer.start("download_task", [target])
        deadline = time.monotonic() + 3
        while not transfer.jobs[blocked_id]["waiting"] and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert transfer.jobs[blocked_id]["waiting_resources"] == ["local_task"]
        transfer.stop(blocked_id)
        assert _wait_job(transfer, blocked_id)["state"] == "stopped"
    finally:
        release.set()
        for job_id in ids:
            assert _wait_job(transfer, job_id)["results"][0]["ok"]
