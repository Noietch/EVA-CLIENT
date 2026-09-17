from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import h5py
import pytest

from core.utils import s3_upload
from core.utils.s3_upload import S3UploadConfig

pytestmark = pytest.mark.integration


class _Response:
    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.body


def _config(prefix: str = "exports/demo") -> S3UploadConfig:
    return S3UploadConfig(
        endpoint="s3.example.com",
        bucket="robot-data",
        prefix=prefix,
        sign_service="https://sign.example.com",
    )


def _write_hdf5(path: Path, value: bytes = b"payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("value", data=list(value))


def test_upload_hdf5_s3_uses_signed_multipart_and_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = tmp_path / "accepted"
    _write_hdf5(accepted / "data" / "episode_000000.hdf5", b"multipart payload")
    config = _config()
    plan = s3_upload.scan_hdf5_s3(accepted, config=config)
    monkeypatch.setattr(s3_upload, "_PART_SIZE", 16)
    monkeypatch.setenv("EVA_S3_SIGN_TOKEN", "runtime-token")
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: int):
        requests.append(request)
        if request.full_url == "https://sign.example.com/api/collector/sign":
            return _Response(
                json.dumps(
                    {
                        "code": 0,
                        "data": {
                            "AWSAccessKeyId": "id",
                            "Signature": "signature",
                            "Date": "date",
                        },
                    }
                ).encode()
            )
        if request.full_url.endswith("?uploads"):
            return _Response(
                b"<InitiateMultipartUploadResult><UploadId>upload-id</UploadId>"
                b"</InitiateMultipartUploadResult>"
            )
        if request.get_method() == "PUT":
            return _Response(headers={"ETag": '"part-etag"'})
        return _Response()

    monkeypatch.setattr(s3_upload.urllib.request, "urlopen", urlopen)
    progress = []

    result = s3_upload.upload_hdf5_s3(
        accepted,
        config=config,
        plan=plan,
        progress_callback=progress.append,
    )

    sign_bodies = [
        json.loads(request.data)
        for request in requests
        if request.full_url == "https://sign.example.com/api/collector/sign"
    ]
    assert sign_bodies[0]["type"] == "initMultiUpload"
    assert [body["partNumber"] for body in sign_bodies[1:-1]] == list(
        range(1, len(sign_bodies) - 1)
    )
    assert sign_bodies[-1]["type"] == "completeUpload"
    assert all(
        request.get_header("X-collector-token") == "runtime-token"
        for request in requests
        if request.full_url == "https://sign.example.com/api/collector/sign"
    )
    complete = requests[-1]
    assert complete.full_url.endswith("?uploadId=upload-id")
    assert complete.get_method() == "POST"
    assert b"<CompleteMultipartUpload>" in complete.data
    assert result.files == 1
    assert result.bytes == plan.bytes_total
    assert progress[-1].files_completed == 1
    assert progress[-1].bytes_completed == plan.bytes_total


def test_upload_hdf5_s3_rejects_source_change_after_scan(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    path = accepted / "episode.hdf5"
    _write_hdf5(path)
    config = _config()
    plan = s3_upload.scan_hdf5_s3(accepted, config=config)
    _write_hdf5(path, b"changed")

    with pytest.raises(ValueError, match="changed after scan"):
        s3_upload.upload_hdf5_s3(accepted, config=config, plan=plan)
