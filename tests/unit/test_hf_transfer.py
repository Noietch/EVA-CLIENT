"""Hugging Face transfer paths without network access."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.app.console import server
from core.config import ConfigDict
from tools.datasets import hf_qc, hf_task_sets

pytestmark = pytest.mark.unit


def test_fetch_qc_selects_configured_path_when_remote_cache_duplicates_set(tmp_path, monkeypatch):
    name = "larybench2_20260908_dual_yam_pickplace"
    canonical = f"datasets/real_robot/dual_yam/{name}"
    cached = f"datasets/real_robot/dual_yam/.hf_dataset_cache/{canonical}"
    source = tmp_path / "remote_qc.jsonl"
    source.write_text('{"episode_index": 1, "verdict": "pass"}\n')
    downloads = []

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return [
                f"{cached}/meta/episodes.jsonl",
                f"{canonical}/meta/episodes.jsonl",
                f"{canonical}/meta/qc.jsonl",
            ]

    def fake_download(**kwargs):
        downloads.append(kwargs["filename"])
        return str(source)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download),
    )
    target = tmp_path / "local" / name
    storage = {"huggingface": {"repo_id": "team/data"}}
    result = hf_task_sets.fetch_qc(tmp_path, name, target, storage, expected_path=canonical)
    assert downloads == [f"{canonical}/meta/qc.jsonl"]
    saved = json.loads((target / "meta/qc.jsonl").read_text())
    assert saved["verdict"] == "pass"
    assert saved["qc_updated_at"] == hf_qc.LEGACY_QC_UPDATED_AT
    assert result["source"] == "qc.jsonl"
    assert (
        hf_task_sets.dataset_repo_path(FakeApi(None), "team/data", name, "revision-sha")
        == canonical
    )


def test_fetch_qc_reports_a_dataset_without_a_published_ledger(tmp_path, monkeypatch):
    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return ["datasets/campaign/set_a/meta/episodes.jsonl"]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=lambda **kwargs: ""),
    )
    with pytest.raises(FileNotFoundError, match="set_a"):
        hf_task_sets.fetch_qc(
            tmp_path,
            "set_a",
            tmp_path / "local" / "set_a",
            {"huggingface": {"repo_id": "team/data"}},
        )


def test_fetch_assets_copies_only_selected_set_photos(tmp_path, monkeypatch):
    task_set = tmp_path / "task_sets" / "set_a"
    task_set.mkdir(parents=True)
    (task_set / "scene.csv").write_text('scene_id,placements\nSC-1,"[{""object_id"":""AST-1""}]"\n')
    remote = tmp_path / "remote"
    (remote / "assets/object_photos/cup").mkdir(parents=True)
    (remote / "assets/object_photos/bowl").mkdir(parents=True)
    (remote / "assets/objects.csv").write_text("object_id,photo_dir\nAST-1,cup\nAST-2,bowl\n")
    (remote / "assets/object_photos/cup/top.png").write_bytes(b"cup photo")
    (remote / "assets/object_photos/bowl/top.png").write_bytes(b"bowl photo")
    downloaded = []

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return [
                "assets/objects.csv",
                "assets/object_photos/cup/top.png",
                "assets/object_photos/bowl/top.png",
            ]

    def fake_download(**kwargs):
        downloaded.append(kwargs["filename"])
        return str(remote / kwargs["filename"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download),
    )
    target = tmp_path / "local" / "assets"
    storage = {"huggingface": {"repo_id": "team/data", "token": "test-token"}}
    result = hf_task_sets.fetch_assets(tmp_path, task_set, target, storage)
    assert (target / "object_photos/cup/top.png").read_bytes() == b"cup photo"
    assert not (target / "object_photos/bowl").exists()
    assert not (target / "objects.csv").exists()
    assert downloaded == ["assets/objects.csv", "assets/object_photos/cup/top.png"]
    assert result["files"] == "1"
    assert result["set"] == "set_a"


def test_dataset_upload_route_names_selected_set_instead_of_raw(tmp_path, monkeypatch):
    plan = tmp_path / "task_sets" / "set_a"
    plan.mkdir(parents=True)
    (plan / "info.yaml").write_text("collection_dir: datasets/real_robot/dual_yam/set_a\n")
    dataset = tmp_path / "set_a" / "raw"
    dataset.mkdir(parents=True)
    config = ConfigDict(collection=dict(tasks={"set_a": [("task a", 1)]}, storage={}))
    captured = {}
    handler = SimpleNamespace(
        ctx=SimpleNamespace(config=config, runtime=SimpleNamespace(active_config=None)),
        _active_collection_dataset=lambda body: dataset,
        _send_json=lambda status, payload: captured.update(response=(status, payload)),
    )
    monkeypatch.setattr(server, "_scene_plan_root", lambda cfg, name=None: plan)

    def fake_publish(project, path, name, storage, **kwargs):
        captured.update(path=path, name=name, remote=kwargs["new_remote_path"])
        return {}

    monkeypatch.setattr(server, "publish_dataset", fake_publish)
    server.ConsoleRequestHandler._post_hf_dataset_upload(handler, {"dataset": "set_a"})
    assert captured["path"] == dataset
    assert captured["name"] == "set_a"
    assert captured["remote"] == "datasets/real_robot/dual_yam/set_a"


def test_qc_download_route_targets_selected_collection_dir_without_raw(tmp_path, monkeypatch):
    plan = tmp_path / "data_collection/task_sets/set_a"
    plan.mkdir(parents=True)
    (plan / "info.yaml").write_text("collection_dir: datasets/real_robot/dual_yam/set_a\n")
    config = ConfigDict(collection=dict(tasks={"set_a": [("task a", 1)]}, storage={}))
    captured = {}
    handler = SimpleNamespace(
        ctx=SimpleNamespace(config=config, runtime=SimpleNamespace(active_config=None)),
        _send_json=lambda status, payload: captured.update(response=(status, payload)),
    )
    monkeypatch.setattr(server, "_scene_plan_root", lambda cfg, name=None: plan)

    def fake_fetch(project, name, destination, storage, **kwargs):
        captured.update(name=name, destination=destination, remote=kwargs["expected_path"])
        return {"path": str(destination / "meta/qc.jsonl")}

    monkeypatch.setattr(server, "fetch_qc", fake_fetch)
    server.ConsoleRequestHandler._post_hf_qc_sync(
        handler, {"dataset": "set_a", "direction": "download"}
    )
    assert captured["name"] == "set_a"
    assert captured["destination"] == (
        tmp_path / "data_collection/datasets/real_robot/dual_yam/set_a"
    )
    assert captured["remote"] == "datasets/real_robot/dual_yam/set_a"
    assert captured["response"][0] == 200


def test_publish_dataset_merges_qc_after_data_upload(tmp_path, monkeypatch):
    dataset = tmp_path / "set_a"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/episodes.jsonl").write_text("{}\n")
    (dataset / "meta/qc.jsonl").write_text('{"episode_index": 0, "qc_verdict": "pass"}\n')
    uploaded = {}
    qc_call = {}

    class FakeApi:
        def __init__(self, token):
            pass

        def list_repo_files(self, repo_id, repo_type, revision):
            return []

        def upload_folder(self, **kwargs):
            uploaded.update(kwargs)
            return SimpleNamespace(oid="data-revision")

    def fake_publish_qc(project_root, dataset_dir, dataset_name, storage, **kwargs):
        qc_call.update(dataset_dir=dataset_dir, dataset_name=dataset_name, **kwargs)
        return {"repo_id": "team/data", "revision": "qc-revision"}

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setattr(hf_task_sets, "publish_qc", fake_publish_qc)
    result = hf_task_sets.publish_dataset(
        tmp_path,
        dataset,
        "set_a",
        {"huggingface": {"repo_id": "team/data"}},
        new_remote_path="datasets/real_robot/dual_yam/set_a",
    )
    # Only recorded content may be uploaded: everything else is excluded by
    # construction, so caches and local state can never reach the remote.
    assert uploaded["allow_patterns"] == [
        "data/**",
        "videos/**",
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
        "meta/stats.json",
    ]
    assert qc_call == {
        "dataset_dir": dataset,
        "dataset_name": "set_a",
        "expected_path": "datasets/real_robot/dual_yam/set_a",
    }
    assert result["revision"] == "qc-revision"
    # Local-only files are reported instead of silently uploading them.
    assert result["not_uploaded"] == ["meta/qc.jsonl"]

    # Task sets are allow-listed the same way: plan files only.
    plan = tmp_path / "task_sets" / "set_a"
    plan.mkdir(parents=True)
    for name in ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv"):
        (plan / name).write_text("x\n")
    (plan / "notes.txt").write_text("draft\n")
    hf_task_sets.publish_task_set(tmp_path, plan)
    assert uploaded["allow_patterns"] == ["info.yaml", "layout.yaml", "scene.csv", "tasks.csv"]


def test_mirror_dataset_drops_replaced_files_and_replaces_the_ledger(tmp_path, monkeypatch):
    name = "set_a"
    remote = f"datasets/real_robot/dual_yam/{name}"
    dataset = tmp_path / "datasets/real_robot/dual_yam" / name
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta/episodes.jsonl").write_text('{"episode_index": 0}\n', encoding="utf-8")
    (dataset / "meta/qc.jsonl").write_text(
        '{"episode_index": 0, "qc_verdict": "pass"}\n', encoding="utf-8"
    )
    (dataset / "data/chunk-000/episode_000000.parquet").write_bytes(b"take")
    uploaded, deleted, added = [], [], []

    class Delete:
        def __init__(self, path_in_repo):
            self.path_in_repo = path_in_repo

    class Add:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def upload_folder(self, **kwargs):
            uploaded.append(kwargs["allow_patterns"])
            return SimpleNamespace(oid="upload-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return [
                f"{remote}/meta/episodes.jsonl",
                f"{remote}/meta/qc.jsonl",
                f"{remote}/data/chunk-000/episode_000000.parquet",
                f"{remote}/data/chunk-000/episode_000009.parquet",
            ]

        def create_commit(self, **kwargs):
            for operation in kwargs["operations"]:
                if isinstance(operation, Delete):
                    deleted.append(operation.path_in_repo)
                else:
                    added.append(operation.path_in_repo)
            return SimpleNamespace(oid="commit-sha")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, CommitOperationDelete=Delete, CommitOperationAdd=Add),
    )
    result = hf_task_sets.mirror_dataset(
        tmp_path, dataset, name, {"huggingface": {"repo_id": "team/data"}}, replace_qc=True
    )

    assert uploaded == [hf_task_sets.DATASET_ALLOW_PATTERNS]
    # The replaced take goes, the shared ledger is never deleted, and it is
    # overwritten rather than merged because the numbering changed.
    assert deleted == [f"{remote}/data/chunk-000/episode_000009.parquet"]
    assert added == [f"{remote}/meta/qc.jsonl"]
    assert result["deleted"] == "1"


def test_dataset_download_replaces_content_and_merges_the_qc_ledger(tmp_path, monkeypatch):
    name = "set_a"
    remote = f"datasets/real_robot/dual_yam/{name}"
    requested = []
    day = "2026-09-16T00:00:00+08:00"

    def snapshot_download(**kwargs):
        requested.append(kwargs["allow_patterns"])
        source = Path(kwargs["local_dir"]) / remote
        (source / "meta").mkdir(parents=True)
        (source / "meta/episodes.jsonl").write_text(
            '{"episode_index": 0}\n{"episode_index": 1}\n', encoding="utf-8"
        )
        (source / "meta/qc.jsonl").write_text(
            f'{{"episode_index": 0, "qc_verdict": "pass", "qc_updated_at": "{day}"}}\n'
            f'{{"episode_index": 1, "qc_verdict": "fail", "qc_updated_at": "{day}"}}\n',
            encoding="utf-8",
        )

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return [f"{remote}/meta/episodes.jsonl"]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=snapshot_download),
    )
    dataset = tmp_path / "data_collection/datasets/real_robot/dual_yam/set_a"
    newer, older = "2026-09-17T08:00:00+08:00", "2026-09-10T08:00:00+08:00"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/episodes.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset / "meta/qc.jsonl").write_text(
        f'{{"episode_index": 0, "qc_verdict": "fail", "qc_updated_at": "{newer}"}}\n'
        f'{{"episode_index": 2, "qc_verdict": "pass", "qc_updated_at": "{older}"}}\n',
        encoding="utf-8",
    )
    hf_task_sets.fetch_dataset(
        tmp_path,
        name,
        dataset.parent,
        {"huggingface": {"repo_id": "team/data"}},
        expected_path=remote,
    )
    assert requested == [[f"{remote}/**"]]
    # Dataset content comes down as published ...
    assert (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8") == (
        '{"episode_index": 0}\n{"episode_index": 1}\n'
    )
    # ... while QC keeps the newest row per episode from either side.
    rows = [json.loads(line) for line in (dataset / "meta/qc.jsonl").read_text().splitlines()]
    assert [(row["episode_index"], row["qc_verdict"]) for row in rows] == [
        (0, "fail"),
        (1, "fail"),
        (2, "pass"),
    ]
