import threading
import time

import pytest

from core.app.console.dataset_manager import DatasetManager

pytestmark = pytest.mark.unit


def wait_job(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = next(j for j in manager.snapshot()["jobs"] if j["id"] == job_id)
        if snapshot["state"] != "running":
            return snapshot
        threading.Event().wait(0.01)
    pytest.fail("Job did not finish")


def test_jobs_run_in_parallel_and_cancel_only_the_conflicting_waiter(monkeypatch, tmp_path):
    from core.app.console import dataset_manager as module

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
    manager = DatasetManager()
    targets = [{"name": name, "plan": tmp_path / name} for name in ("first", "second")]
    first_id = manager.start("upload_task", targets, tmp_path, {})
    assert started.wait(5)
    parallel_id = manager.start(
        "upload_task", [{"name": "third", "plan": tmp_path / "third"}], tmp_path, {}
    )
    assert wait_job(manager, parallel_id)["state"] == "done"
    waiting_id = manager.start("upload_task", targets[:1], tmp_path, {})
    manager.stop(waiting_id)
    assert wait_job(manager, waiting_id)["state"] == "stopped"
    assert calls == ["first", "third"]
    release.set()
    assert finished.wait(5)
    first = wait_job(manager, first_id)
    assert calls == ["first", "third", "second"]
    assert first["results"][0]["error"] == "offline"
    assert first["results"][1]["ok"]


def test_same_dataset_disjoint_uploads_and_remote_reads_run_together(monkeypatch, tmp_path):
    from core.app.console import dataset_manager as module

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
    manager = DatasetManager()
    monkeypatch.setattr(
        manager,
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
            ids.append(manager.start(f"upload_{kind}", [target], tmp_path, {}))
        assert all(event.wait(3) for event in started.values())
        for action in ("refresh", "verify"):
            read_id = manager.start(action, [target], tmp_path, {})
            assert wait_job(manager, read_id)["results"][0]["ok"]
            cloud = manager.snapshot()["remote"]["fold"]
            assert cloud["stale"] is True
            assert "verification" not in cloud
        blocked_id = manager.start("download_task", [target], tmp_path, {})
        deadline = time.monotonic() + 3
        while not manager.jobs[blocked_id]["waiting"] and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert manager.jobs[blocked_id]["waiting_resources"] == ["local_task"]
        manager.stop(blocked_id)
        assert wait_job(manager, blocked_id)["state"] == "stopped"
    finally:
        release.set()
        for job_id in ids:
            assert wait_job(manager, job_id)["results"][0]["ok"]
