from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.utils import dataset_upload
from core.utils.dataset_upload import DatasetUploadProgress


def _result(local_dir: Path, remote_dir: str, files: int, bytes_count: int):
    return SimpleNamespace(
        local_dir=str(local_dir),
        remote_dir=remote_dir,
        destination=remote_dir,
        files=files,
        bytes=bytes_count,
    )


def test_resolve_dataset_uploads_returns_configured_sftp() -> None:
    specs = dataset_upload.resolve_dataset_uploads(
        {
            "sftp": {
                "host": "server",
                "port": 2222,
                "user": "robot",
                "remote_dir": "/datasets/root",
            },
        },
        "accepted",
    )

    assert [spec.backend for spec in specs] == ["sftp"]
    assert specs[0].target == "/datasets/root/accepted"
    assert specs[0].display_root == "server:2222 · /datasets/root"


@pytest.mark.parametrize("remote_root", ["/datasets/", "/datasets///"])
def test_resolve_dataset_uploads_normalizes_trailing_root_slashes(remote_root: str) -> None:
    specs = dataset_upload.resolve_dataset_uploads(
        {"sftp": {"host": "server", "remote_dir": remote_root}},
        "accepted",
    )

    assert specs[0].root == "/datasets"
    assert specs[0].target == "/datasets/accepted"


@pytest.mark.parametrize("dataset_name", [".", "..", "nested/name"])
def test_resolve_dataset_uploads_rejects_invalid_dataset_name(dataset_name: str) -> None:
    with pytest.raises(ValueError, match="one path component"):
        dataset_upload.resolve_dataset_uploads(
            {"sftp": {"host": "server", "remote_dir": "/datasets/root"}},
            dataset_name,
        )


@pytest.mark.parametrize(
    "remote_root", ["/datasets/./root", "/datasets//root", "/datasets/root\nx"]
)
def test_resolve_dataset_uploads_rejects_non_canonical_root(remote_root: str) -> None:
    with pytest.raises(ValueError, match="absolute canonical path"):
        dataset_upload.resolve_dataset_uploads(
            {"sftp": {"host": "server", "remote_dir": remote_root}},
            "accepted",
        )


def test_upload_dataset_directory_reports_sftp_progress_and_actual_remote_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "accepted"
    local_dir.mkdir()
    (local_dir / "one").write_bytes(b"123")
    (local_dir / "two").write_bytes(b"45")
    specs = dataset_upload.resolve_dataset_uploads(
        {"sftp": {"host": "host", "remote_dir": "/root"}},
        local_dir.name,
    )
    calls = []
    progress = []

    def fake_sftp_upload(path, *, remote_dir, progress_callback, **_kwargs):
        calls.append((path, remote_dir))
        progress_callback(DatasetUploadProgress(2, 2, 5, 5, "two"))
        return _result(path, "/root/accepted.copy_20260828T060553123456Z", 2, 5)

    monkeypatch.setattr(dataset_upload, "upload_directory_sftp", fake_sftp_upload)

    result = dataset_upload.upload_dataset_directory(
        local_dir,
        specs,
        progress_callback=progress.append,
    )

    assert calls == [(local_dir.resolve(), "/root/accepted")]
    assert result.remote_dir == "/root/accepted.copy_20260828T060553123456Z"
    assert result.files == 2
    assert result.bytes == 5
    assert progress[0] == DatasetUploadProgress(0, 2, 0, 5, "")
    assert progress[-1] == DatasetUploadProgress(2, 2, 5, 5, "sftp: two")


def test_upload_dataset_directory_names_failed_sftp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "accepted"
    local_dir.mkdir()
    (local_dir / "file").write_text("data")
    specs = dataset_upload.resolve_dataset_uploads(
        {"sftp": {"host": "host", "remote_dir": "/root"}},
        local_dir.name,
    )
    monkeypatch.setattr(
        dataset_upload,
        "upload_directory_sftp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("denied")),
    )

    with pytest.raises(RuntimeError, match="sftp dataset upload failed"):
        dataset_upload.upload_dataset_directory(local_dir, specs)
