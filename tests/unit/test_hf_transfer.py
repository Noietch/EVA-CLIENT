"""Hugging Face transfer paths without network access."""

import json
import sys
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


def test_fetch_qc_falls_back_to_episode_quality_without_erasing_local_verdict(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "remote_episodes.jsonl"
    source.write_text('{"episode_index": 1, "quality": "yellow", "quality_issues": ["skew"]}\n')
    target = tmp_path / "local" / "set_a" / "raw" / "meta" / "episodes.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(
        '{"episode_index": 1, "quality": "green", "qc_verdict": "fail", "qc_note": "retake"}\n'
        '{"episode_index": 2, "quality": "green"}\n'
    )

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
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=lambda **kwargs: str(source)),
    )
    result = hf_task_sets.fetch_qc(
        tmp_path, "set_a", target.parent.parent, {"huggingface": {"repo_id": "team/data"}}
    )
    import json

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert rows[0]["quality"] == "yellow"
    assert rows[0]["quality_issues"] == ["skew"]
    assert rows[0]["qc_verdict"] == "fail"
    assert rows[0]["qc_note"] == "retake"
    assert rows[1]["quality"] == "green"
    assert result["source"] == "episodes.jsonl"
    assert result["episodes"] == "1"


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

    # Task sets and assets are allow-listed the same way: plans and photos only.
    plan = tmp_path / "task_sets" / "set_a"
    plan.mkdir(parents=True)
    for name in ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv"):
        (plan / name).write_text("x\n")
    (plan / "notes.txt").write_text("draft\n")
    hf_task_sets.publish_task_set(tmp_path, plan)
    assert uploaded["allow_patterns"] == ["info.yaml", "layout.yaml", "scene.csv", "tasks.csv"]

    assets = tmp_path / "assets"
    (assets / "object_photos").mkdir(parents=True)
    (assets / "objects.csv").write_text("object_id\n")
    (assets / ".preview_cache").mkdir()
    hf_task_sets.publish_assets(tmp_path, assets)
    assert uploaded["allow_patterns"] == ["objects.csv", "object_photos/**"]
