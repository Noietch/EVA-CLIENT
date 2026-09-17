"""Check QC sync paths without contacting Hugging Face."""

import importlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from core.utils.lerobot import LeRobotDatasetIO
from tools.datasets.hf_qc import LEGACY_QC_UPDATED_AT, merge_qc
from tools.datasets.hf_task_sets import fetch_qc, publish_qc

pytestmark = pytest.mark.unit


def test_qc_sync_uses_existing_nested_dataset_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "sample"
    remote_dir = f"datasets/real_robot/dual_yam/{name}"
    dataset_dir = tmp_path / name
    qc_path = dataset_dir / "meta/qc.jsonl"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text(
        '{"episode_index": 0, "qc_verdict": "pass", '
        '"qc_updated_at": "2026-09-16T03:00:00Z"}\n'
        '{"episode_index": 1, "qc_verdict": "fail"}\n',
        encoding="utf-8",
    )
    original_mode = qc_path.stat().st_mode & 0o777
    downloaded = tmp_path / "remote-qc.jsonl"
    downloaded.write_text(
        '{"episode_index": 0, "qc_verdict": "fail", '
        '"qc_updated_at": "2026-09-15T01:00:00Z"}\n'
        '{"episode_index": 2, "qc_verdict": "pass"}\n',
        encoding="utf-8",
    )

    api = Mock()
    api.list_repo_files.return_value = [
        f"{remote_dir}/meta/episodes.jsonl",
        f"{remote_dir}/meta/qc.jsonl",
    ]
    api.repo_info.return_value = SimpleNamespace(sha="before")
    api.create_commit.return_value = SimpleNamespace(oid="revision")
    download = Mock(return_value=str(downloaded))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    storage = {"huggingface": {"repo_id": "owner/data", "token": "test", "revision": "main"}}

    result = publish_qc(tmp_path, dataset_dir, name, storage, expected_path=remote_dir)
    assert result["revision"] == "revision"
    kwargs = api.create_commit.call_args.kwargs
    assert kwargs["parent_commit"] == "before"
    assert kwargs["operations"][0].path_in_repo == f"{remote_dir}/meta/qc.jsonl"
    saved = [json.loads(line) for line in qc_path.read_text().splitlines()]
    assert [row["episode_index"] for row in saved] == [0, 1, 2]
    assert saved[0]["qc_verdict"] == "pass"
    assert saved[1]["qc_updated_at"] == LEGACY_QC_UPDATED_AT
    assert qc_path.stat().st_mode & 0o777 == original_mode

    result = fetch_qc(tmp_path, name, dataset_dir, storage, expected_path=remote_dir)
    assert download.call_args.kwargs["filename"] == f"{remote_dir}/meta/qc.jsonl"
    assert result["source"] == "qc.jsonl"
    verdicts = [
        json.loads(line)["qc_verdict"] for line in Path(result["path"]).read_text().splitlines()
    ]
    assert verdicts == [
        "pass",
        "fail",
        "pass",
    ]


def test_merge_qc_rejects_ambiguous_ties_and_duplicate_keys() -> None:
    local = b'{"episode_index": 1, "qc_verdict": "pass"}\n'
    remote = b'{"episode_index": 1, "qc_verdict": "fail"}\n'
    with pytest.raises(ValueError, match="same time: episode 1"):
        merge_qc(local, remote)
    with pytest.raises(ValueError, match="Duplicate QC episode_index: 1"):
        merge_qc(local + local, b"")


def test_merge_qc_uses_newest_complete_row_and_keeps_unmatched_rows() -> None:
    local = (
        b'{"episode_index": 0, "qc_verdict": "pass", '
        b'"qc_updated_at": "2026-09-15T03:00:00+08:00"}\n'
        b'{"episode_index": 1, "qc_verdict": "fail", '
        b'"qc_updated_at": "2026-09-16T03:00:00Z"}\n'
        b'{"episode_index": 2, "qc_verdict": "pass"}\n'
    )
    remote = (
        b'{"episode_index": 0, "qc_verdict": "fail", '
        b'"qc_updated_at": "2026-09-16T01:00:00+08:00"}\n'
        b'{"episode_index": 1, "qc_verdict": "pass", '
        b'"qc_updated_at": "2026-09-15T01:00:00Z"}\n'
        b'{"episode_index": 3, "qc_verdict": "fail"}\n'
    )
    rows = [json.loads(line) for line in merge_qc(local, remote).splitlines()]
    assert [row["episode_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["qc_verdict"] for row in rows] == ["fail", "fail", "pass", "fail"]
    assert rows[2]["qc_updated_at"] == LEGACY_QC_UPDATED_AT
    assert rows[3]["qc_updated_at"] == LEGACY_QC_UPDATED_AT


def test_new_qc_edits_have_monotonic_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text('{"episode_index": 7}\n', encoding="utf-8")
    io = LeRobotDatasetIO(root)
    assert io.mark_qc(7, "fail", "first")
    saved = json.loads((meta / "qc.jsonl").read_text())
    first = datetime.fromisoformat(saved["qc_updated_at"])
    assert first.utcoffset() is not None
    saved["qc_updated_at"] = "2099-01-01T00:00:00+00:00"
    (meta / "qc.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")
    assert io.mark_qc(7, "pass", "second")
    saved = json.loads((meta / "qc.jsonl").read_text())
    assert datetime.fromisoformat(saved["qc_updated_at"]) > datetime.fromisoformat(
        "2099-01-01T00:00:00+00:00"
    )
    assert saved["qc_verdict"] == "pass"


def test_publish_qc_reports_missing_local_file(tmp_path: Path) -> None:
    storage = {"huggingface": {"repo_id": "owner/data", "token": "test"}}
    with pytest.raises(FileNotFoundError, match=r"QC file is missing: .*meta/qc.jsonl"):
        publish_qc(tmp_path, tmp_path / "missing", "missing", storage)


def test_publish_qc_failure_leaves_local_file_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    qc_path = dataset / "meta/qc.jsonl"
    qc_path.parent.mkdir(parents=True)
    original = b'{"episode_index": 0, "qc_verdict": "fail"}\n'
    qc_path.write_bytes(original)
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="before")
    api.list_repo_files.return_value = []
    api.create_commit.side_effect = RuntimeError("commit failed")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)

    with pytest.raises(RuntimeError, match="commit failed"):
        publish_qc(
            tmp_path,
            dataset,
            "dataset",
            {"huggingface": {"repo_id": "owner/data"}},
            expected_path="datasets/real_robot/dual_yam/dataset",
        )
    assert qc_path.read_bytes() == original


def test_qc_routes_pass_nested_collection_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_app = importlib.import_module("tools.datasets.app")
    plans = tmp_path / "task_sets"
    batch = plans / "set_a"
    batch.mkdir(parents=True)
    (batch / "info.yaml").write_text(
        "collection_dir: datasets/real_robot/dual_yam/set_a\n", encoding="utf-8"
    )
    app = dataset_app.create_app(plans, tmp_path / "assets", tmp_path / "collection")
    app.config["TESTING"] = True
    client = app.test_client()
    client.environ_base["HTTP_X_EVA_DATASET_EDITOR"] = "1"
    client.environ_base["HTTP_X_EVA_EDIT_MODE"] = "1"
    calls = []

    def fake_sync(*args, **kwargs):
        calls.append(kwargs["expected_path"])
        return {"repo_id": "owner/data"}

    monkeypatch.setattr(dataset_app, "publish_qc", fake_sync)
    monkeypatch.setattr(dataset_app, "fetch_qc", fake_sync)
    assert client.post("/api/batches/set_a/hf/qc/upload").status_code == 200
    assert client.post("/api/batches/set_a/hf/qc/download").status_code == 200
    assert calls == ["datasets/real_robot/dual_yam/set_a"] * 2
