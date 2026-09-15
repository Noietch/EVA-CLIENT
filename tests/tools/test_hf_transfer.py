"""Hugging Face transfer paths without network access."""

import sys
from types import SimpleNamespace

import pytest

from core.app.console import server
from core.config import ConfigDict
from tools.datasets import hf_task_sets

pytestmark = pytest.mark.unit


def test_fetch_qc_uses_actual_dataset_repo_path(tmp_path, monkeypatch):
    source = tmp_path / "remote_qc.jsonl"
    source.write_text('{"episode_index": 1, "verdict": "pass"}\n')
    calls = {}

    class FakeApi:
        def __init__(self, token):
            assert token == "test-token"

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return ["datasets/campaign/set_a/meta/episodes.jsonl",
                    "datasets/campaign/set_a/meta/qc.jsonl"]

    def fake_download(**kwargs):
        calls.update(kwargs)
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=fake_download))
    target = tmp_path / "local" / "set_a"
    storage = {"huggingface": {"repo_id": "team/data", "token": "test-token"}}
    result = hf_task_sets.fetch_qc(tmp_path, "set_a", target, storage)
    assert calls["filename"] == "datasets/campaign/set_a/meta/qc.jsonl"
    assert calls["revision"] == "revision-sha"
    assert (target / "meta/qc.jsonl").read_bytes() == source.read_bytes()
    assert result["path"] == str(target / "meta/qc.jsonl")


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
            return [f"{cached}/meta/episodes.jsonl",
                    f"{canonical}/meta/episodes.jsonl",
                    f"{canonical}/meta/qc.jsonl"]

    def fake_download(**kwargs):
        downloads.append(kwargs["filename"])
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=fake_download))
    target = tmp_path / "local" / name
    storage = {"huggingface": {"repo_id": "team/data"}}
    result = hf_task_sets.fetch_qc(tmp_path, name, target, storage,
                                    expected_path=canonical)
    assert downloads == [f"{canonical}/meta/qc.jsonl"]
    assert (target / "meta/qc.jsonl").read_bytes() == source.read_bytes()
    assert result["source"] == "qc.jsonl"
    assert hf_task_sets.dataset_repo_path(FakeApi(None), "team/data", name,
                                          "revision-sha") == canonical


def test_fetch_qc_falls_back_to_episode_quality_without_erasing_local_verdict(
    tmp_path, monkeypatch,
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

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=lambda **kwargs: str(source)))
    result = hf_task_sets.fetch_qc(tmp_path, "set_a", target.parent.parent,
                                    {"huggingface": {"repo_id": "team/data"}})
    import json
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert rows[0]["quality"] == "yellow"
    assert rows[0]["quality_issues"] == ["skew"]
    assert rows[0]["qc_verdict"] == "fail"
    assert rows[0]["qc_note"] == "retake"
    assert rows[1]["quality"] == "green"
    assert result["source"] == "episodes.jsonl"
    assert result["episodes"] == "1"


def test_fetch_qc_reports_unpublished_qc_without_local_canonical_metadata(
    tmp_path, monkeypatch,
):
    downloads = []

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return ["datasets/real_robot/dual_yam/debug_set/meta/episodes.jsonl"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=lambda **kwargs: downloads.append(kwargs)))
    target = tmp_path / "data_collection/datasets/real_robot/dual_yam/debug_set"
    result = hf_task_sets.fetch_qc(
        tmp_path, "debug_set", target,
        {"huggingface": {"repo_id": "team/data"}},
        expected_path="datasets/real_robot/dual_yam/debug_set",
    )
    assert result["source"] == "none"
    assert "QC is not published" in result["message"]
    assert downloads == []
    assert not target.exists()


def test_fetch_assets_copies_only_selected_set_photos(tmp_path, monkeypatch):
    task_set = tmp_path / "task_sets" / "set_a"
    task_set.mkdir(parents=True)
    (task_set / "scene.csv").write_text('scene_id,placements\nSC-1,"[{""object_id"":""AST-1""}]"\n')
    remote = tmp_path / "remote"
    (remote / "assets/object_photos/cup").mkdir(parents=True)
    (remote / "assets/object_photos/bowl").mkdir(parents=True)
    (remote / "assets/objects.csv").write_text('object_id,photo_dir\nAST-1,cup\nAST-2,bowl\n')
    (remote / "assets/object_photos/cup/top.png").write_bytes(b"cup photo")
    (remote / "assets/object_photos/bowl/top.png").write_bytes(b"bowl photo")
    downloaded = []

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return ["assets/objects.csv", "assets/object_photos/cup/top.png",
                    "assets/object_photos/bowl/top.png"]

    def fake_download(**kwargs):
        downloaded.append(kwargs["filename"])
        return str(remote / kwargs["filename"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=fake_download))
    target = tmp_path / "local" / "assets"
    storage = {"huggingface": {"repo_id": "team/data", "token": "test-token"}}
    result = hf_task_sets.fetch_assets(tmp_path, task_set, target, storage)
    assert (target / "object_photos/cup/top.png").read_bytes() == b"cup photo"
    assert not (target / "object_photos/bowl").exists()
    assert not (target / "objects.csv").exists()
    assert downloaded == ["assets/objects.csv", "assets/object_photos/cup/top.png"]
    assert result["files"] == "1"
    assert result["set"] == "set_a"


def test_publish_dataset_uses_selected_set_remote_path_when_new(tmp_path, monkeypatch):
    dataset = tmp_path / "set_a" / "raw"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/episodes.jsonl").write_text('{"episode_index": 0}\n')
    uploaded = {}

    class FakeApi:
        def __init__(self, token):
            pass

        def list_repo_files(self, repo_id, repo_type, revision):
            return []

        def upload_folder(self, **kwargs):
            uploaded.update(kwargs)
            return SimpleNamespace(oid="new-revision")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    result = hf_task_sets.publish_dataset(
        tmp_path, dataset, "set_a", {"huggingface": {"repo_id": "team/data"}},
        new_remote_path="datasets/real_robot/dual_yam/set_a",
    )
    assert uploaded["path_in_repo"] == "datasets/real_robot/dual_yam/set_a"
    assert result["dataset"] == "set_a"


def test_publish_qc_uploads_only_selected_set_qc_to_configured_path(tmp_path, monkeypatch):
    dataset = tmp_path / "datasets/real_robot/dual_yam/set_a"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/qc.jsonl").write_text('{"episode_index": 0, "qc_verdict": "pass"}\n')
    uploaded = {}

    class FakeApi:
        def __init__(self, token):
            pass

        def list_repo_files(self, repo_id, repo_type, revision):
            return []

        def upload_file(self, **kwargs):
            uploaded.update(kwargs)
            return SimpleNamespace(oid="new-revision")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    result = hf_task_sets.publish_qc(
        tmp_path, dataset, "set_a", {"huggingface": {"repo_id": "team/data"}},
        expected_path="datasets/real_robot/dual_yam/set_a",
    )
    assert uploaded["path_or_fileobj"] == str(dataset / "meta/qc.jsonl")
    assert uploaded["path_in_repo"] == "datasets/real_robot/dual_yam/set_a/meta/qc.jsonl"
    assert result["dataset"] == "set_a"


def test_fetch_qc_finds_legacy_top_level_qc_for_selected_set(tmp_path, monkeypatch):
    name = "debug_set"
    canonical = f"datasets/real_robot/dual_yam/{name}"
    legacy = f"datasets/{name}/meta/qc.jsonl"
    source = tmp_path / "remote-qc.jsonl"
    source.write_text('{"episode_index": 4, "qc_verdict": "fail"}\n')
    downloads = []

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, repo_id, repo_type, revision):
            return SimpleNamespace(sha="revision-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            return [f"{canonical}/meta/episodes.jsonl", legacy]

    def fake_download(**kwargs):
        downloads.append(kwargs["filename"])
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=fake_download))
    target = tmp_path / "data_collection/datasets/real_robot/dual_yam/debug_set"
    result = hf_task_sets.fetch_qc(
        tmp_path, name, target, {"huggingface": {"repo_id": "team/data"}},
        expected_path=canonical,
    )
    assert downloads == [legacy]
    assert (target / "meta/qc.jsonl").read_bytes() == source.read_bytes()
    assert result["source"] == "qc.jsonl"


def test_asset_and_task_routes_reject_other_sets(tmp_path, monkeypatch):
    plan = tmp_path / "task_sets" / "set_a"
    plan.mkdir(parents=True)
    config = ConfigDict(collection=dict(tasks={"set_a": [("task a", 1)]}, storage={}))
    responses = []
    handler = SimpleNamespace(
        ctx=SimpleNamespace(config=config, runtime=SimpleNamespace(active_config=None)),
        _send_json=lambda status, payload: responses.append((status, payload)),
    )
    monkeypatch.setattr(server, "_scene_plan_root", lambda cfg, dataset=None: plan)
    fetched = []
    monkeypatch.setattr(server, "fetch_assets", lambda project, selected, destination, storage:
                        fetched.append((selected, destination)) or {"set": selected.name})
    monkeypatch.setattr(server, "fetch_task_set", lambda project, name, destination, storage:
                        fetched.append((name, destination)) or {"task_set": name})
    server.ConsoleRequestHandler._post_hf_assets_download(handler, {"dataset": "set_b"})
    server.ConsoleRequestHandler._post_hf_task_set_sync(handler, {"task_set": "set_b"})
    assert responses == [(409, {"ok": False, "error": "unknown collection set"})] * 2
    assert fetched == []
    server.ConsoleRequestHandler._post_hf_assets_download(handler, {"dataset": "set_a"})
    server.ConsoleRequestHandler._post_hf_task_set_sync(handler, {"task_set": "set_a"})
    assert fetched[0][0] == plan
    assert fetched[1][0] == "set_a"


def test_dataset_upload_route_names_selected_set_instead_of_raw(tmp_path, monkeypatch):
    plan = tmp_path / "task_sets" / "set_a"
    plan.mkdir(parents=True)
    (plan / "info.yaml").write_text('collection_dir: datasets/real_robot/dual_yam/set_a\n')
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
    (plan / "info.yaml").write_text(
        "collection_dir: datasets/real_robot/dual_yam/set_a\n")
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
        handler, {"dataset": "set_a", "direction": "download"})
    assert captured["name"] == "set_a"
    assert captured["destination"] == (
        tmp_path / "data_collection/datasets/real_robot/dual_yam/set_a")
    assert captured["remote"] == "datasets/real_robot/dual_yam/set_a"
    assert captured["response"][0] == 200


def test_qc_upload_route_uses_selected_collection_dir_and_remote_path(tmp_path, monkeypatch):
    plan = tmp_path / "data_collection/task_sets/set_a"
    plan.mkdir(parents=True)
    (plan / "info.yaml").write_text(
        "collection_dir: datasets/real_robot/dual_yam/set_a\n")
    config = ConfigDict(collection=dict(tasks={"set_a": [("task a", 1)]}, storage={}))
    captured = {}
    handler = SimpleNamespace(
        ctx=SimpleNamespace(config=config, runtime=SimpleNamespace(active_config=None)),
        _send_json=lambda status, payload: captured.update(response=(status, payload)),
    )
    monkeypatch.setattr(server, "_scene_plan_root", lambda cfg, name=None: plan)

    def fake_publish(project, destination, name, storage, **kwargs):
        captured.update(destination=destination, name=name, remote=kwargs["expected_path"])
        return {"path": f"{kwargs['expected_path']}/meta/qc.jsonl"}

    monkeypatch.setattr(server, "publish_qc", fake_publish)
    server.ConsoleRequestHandler._post_hf_qc_sync(
        handler, {"dataset": "set_a", "direction": "upload"})
    assert captured["destination"] == (
        tmp_path / "data_collection/datasets/real_robot/dual_yam/set_a")
    assert captured["remote"] == "datasets/real_robot/dual_yam/set_a"
    assert captured["response"][0] == 200


def test_default_hf_config_matches_sync_script(tmp_path):
    config = hf_task_sets._config(tmp_path, {"huggingface": {}})
    assert config["repo_id"] == "Noietch/data_collection"
