import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from core.app.console.dataset_manager import (
    DatasetManager,
    compare_files,
    compare_qc,
    qc_entries,
    qc_summary,
)

pytestmark = pytest.mark.unit


def wait_job(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = next(j for j in manager.snapshot()["jobs"] if j["id"] == job_id)
        if snapshot["state"] != "running":
            return snapshot
        threading.Event().wait(0.01)
    pytest.fail("Job did not finish")


@pytest.mark.parametrize("action", ["refresh", "verify"])
def test_refresh_and_verify_both_keep_all_comparison_results(monkeypatch, tmp_path, action):
    manager = DatasetManager()
    verification = {kind: {"state": "same"} for kind in ("data", "task", "qc")}

    def inspect(target, project_root, storage, verify, progress):
        assert verify is True
        assert manager.leases
        access = next(iter(manager.leases.values()))[1]
        assert access["local_task"] == "read"
        return {"episodes": 20, "task_count": 4, "verification": verification}

    monkeypatch.setattr(manager, "_inspect", inspect)
    job_id = manager.start(action, [{"name": "fold"}], tmp_path, {})
    assert wait_job(manager, job_id)["results"][0]["ok"]
    cloud = manager.snapshot()["remote"]["fold"]
    assert cloud["episodes"] == 20
    assert cloud["verification"] == verification


def test_remote_task_count_uses_task_plan_rows(monkeypatch, tmp_path):
    import huggingface_hub

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text('{"episode_index": 0}\n{"episode_index": 1}\n')
    tasks = tmp_path / "tasks.csv"
    tasks.write_text('task_id,prompt_en\nTASK-001,"fold, then release"\nTASK-002,"fold\nleft"\n')
    downloads = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixed-revision")

        def list_repo_tree(self, repo, path_in_repo, **kwargs):
            name = "tasks.csv" if path_in_repo.startswith("task_sets/") else "meta/episodes.jsonl"
            return [SimpleNamespace(path=f"{path_in_repo}/{name}", blob_id="test", size=10)]

    def download(repo, filename, **kwargs):
        downloads.append(kwargs["revision"])
        return str(tasks if filename.endswith("tasks.csv") else episodes)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = DatasetManager()._inspect(
        {"name": "fold", "remote_path": "datasets/fold"},
        tmp_path,
        {"huggingface": {"repo_id": "test/repo"}},
        False,
    )
    assert result["episodes"] == 2
    assert result["task_count"] == 2
    assert downloads == ["fixed-revision", "fixed-revision"]


def test_hash_comparison_detects_missing_extra_and_same_size_changes(tmp_path):
    same = tmp_path / "same"
    same.write_bytes(b"abc")
    changed = tmp_path / "changed"
    changed.write_bytes(b"abd")
    git = hashlib.sha1(b"blob 3\0abc").hexdigest()
    sha = hashlib.sha256(b"abc").hexdigest()
    remote = {
        "same": SimpleNamespace(size=3, blob_id=git, lfs=None),
        "changed": SimpleNamespace(size=3, blob_id="", lfs={"sha256": sha}),
        "missing": SimpleNamespace(size=3, blob_id=git, lfs=None),
    }
    result = compare_files({"same": same, "changed": changed, "extra": same}, remote)
    assert result == {
        "state": "different",
        "missing_local": ["missing"],
        "local_only": ["extra"],
        "changed": ["changed"],
        "unknown": [],
    }


def test_qc_comparison_overlays_reviews_and_compares_episode_identity():
    episodes = [
        {"episode_index": 3, "qc_verdict": "fail", "quality": "green"},
        {"episode_index": 4, "quality": "red"},
    ]
    qc = [{"episode_index": 3, "qc_verdict": "pass"}]
    assert qc_summary(episodes, qc) == {"accept": 1, "fail": 1, "unreviewed": 0}
    local = qc_entries(episodes, qc)
    assert compare_qc(local, qc_entries(list(reversed(episodes)), qc))["state"] == "same"
    assert compare_qc(local, qc_entries(episodes, []))["changed"] == ["3"]
    assert compare_qc({}, {})["state"] == "absent"


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


def test_qc_upload_supports_embedded_verdicts_without_mutating_source(monkeypatch, tmp_path):
    from core.app.console import dataset_manager as module

    root = tmp_path / "raw"
    (root / "meta").mkdir(parents=True)
    source = root / "meta/episodes.jsonl"
    source.write_text(json.dumps({"episode_index": 1, "qc_verdict": "pass"}) + "\n")
    original = source.read_bytes()
    captured = []
    monkeypatch.setattr(
        module,
        "publish_qc",
        lambda project, path, name, storage, **kw: captured.append(
            json.loads((path / "meta/qc.jsonl").read_text())
        ),
    )
    manager = DatasetManager()
    job_id = manager.start(
        "upload_qc",
        [{"name": "test", "qc_root": root, "remote_path": "datasets/test"}],
        tmp_path,
        {},
    )
    assert wait_job(manager, job_id)["state"] == "done"
    assert captured[0]["qc_verdict"] == "pass"
    assert source.read_bytes() == original
    assert not (root / "meta/qc.jsonl").exists()


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
        "raw": tmp_path,
        "qc_root": tmp_path,
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


def test_data_upload_can_exclude_qc_without_changing_default(monkeypatch, tmp_path):
    import huggingface_hub

    from tools.datasets.hf_task_sets import publish_dataset

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text("{}\n")
    calls = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def list_repo_files(self, *args, **kwargs):
            return []

        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(oid="revision")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    storage = {"huggingface": {"repo_id": "test/repo"}}
    publish_dataset(tmp_path, tmp_path, "fold", storage, include_qc=False)
    assert calls[0]["ignore_patterns"] == ["meta/qc.jsonl"]
    publish_dataset(tmp_path, tmp_path, "fold", storage)
    assert "ignore_patterns" not in calls[1]


@pytest.mark.parametrize("local_report", [False, True])
def test_verification_ignores_export_report_but_detects_recorded_content_changes(
    monkeypatch, tmp_path, local_report
):
    import huggingface_hub

    raw = tmp_path / "raw"
    (raw / "meta").mkdir(parents=True)
    (raw / "videos").mkdir()
    (raw / "meta/episodes.jsonl").write_text('{"episode_index": 0}\n')
    (raw / "videos/episode.mp4").write_bytes(b"video")
    if local_report:
        (raw / "meta/quality_split.json").write_text('{"old": "local"}')
    content = {
        "meta/episodes.jsonl": (raw / "meta/episodes.jsonl").read_bytes(),
        "videos/episode.mp4": b"video",
        "meta/quality_split.json": b'{"old":"remote"}',
    }

    class Api:
        def __init__(self, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="revision")

        def list_repo_tree(self, repo, path_in_repo, **kwargs):
            if path_in_repo.startswith("task_sets/"):
                return []
            return [
                SimpleNamespace(
                    path=f"{path_in_repo}/{name}",
                    size=len(value),
                    lfs=None,
                    blob_id=hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest(),
                )
                for name, value in content.items()
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(raw / "meta/episodes.jsonl")
    )
    target = {
        "name": "fold",
        "remote_path": "datasets/fold",
        "raw": raw,
        "qc_root": raw,
        "plan": tmp_path / "plan",
    }
    manager = DatasetManager()
    storage = {"huggingface": {"repo_id": "test/repo"}}
    result = manager._inspect(target, tmp_path, storage, True)["verification"]["data"]
    assert result["state"] == "same"
    assert result["missing_local"] == result["local_only"] == result["changed"] == []
    (raw / "videos/episode.mp4").write_bytes(b"other")
    result = manager._inspect(target, tmp_path, storage, True)["verification"]["data"]
    assert result["state"] == "different"
    assert result["changed"] == ["videos/episode.mp4"]
