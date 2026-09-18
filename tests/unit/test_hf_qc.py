"""Check QC sync paths without contacting Hugging Face."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

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
