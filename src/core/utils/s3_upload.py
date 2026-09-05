"""Upload HDF5 exports to S3 through the collector signing service."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import h5py

from core.utils.upload_plan import UploadProgress, UploadResult, file_digest

_PART_SIZE = 50 * 1024 * 1024
_TOKEN_ENV = "EVA_S3_SIGN_TOKEN"


@dataclasses.dataclass(frozen=True)
class S3UploadConfig:
    endpoint: str
    bucket: str
    prefix: str
    sign_service: str
    secure: bool = False

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip()
        endpoint_url = urllib.parse.urlsplit(f"//{endpoint}")
        try:
            endpoint_port = endpoint_url.port
        except ValueError as error:
            raise ValueError("S3 endpoint has an invalid port") from error
        if (
            not endpoint_url.hostname
            or endpoint_url.username is not None
            or endpoint_url.password is not None
            or endpoint_url.path
            or endpoint_url.query
            or endpoint_url.fragment
            or endpoint.startswith("-")
            or any(char in endpoint for char in "\r\n")
        ):
            raise ValueError("S3 endpoint must be a host with an optional port")
        if endpoint_port is not None and not 1 <= endpoint_port <= 65535:
            raise ValueError("S3 endpoint port must be in [1, 65535]")

        bucket = self.bucket.strip()
        if (
            not bucket
            or PurePosixPath(bucket).name != bucket
            or bucket in {".", ".."}
            or any(char in bucket for char in "\r\n")
        ):
            raise ValueError("S3 bucket must be one path component")

        prefix = self.prefix.rstrip("/")
        prefix_path = PurePosixPath(prefix)
        if (
            not prefix
            or prefix_path.is_absolute()
            or any(part in {"", ".", ".."} for part in prefix_path.parts)
            or prefix_path.as_posix() != prefix
            or any(char in prefix for char in "\r\n")
        ):
            raise ValueError("S3 prefix must be a canonical relative path")

        sign_service = self.sign_service.rstrip("/")
        sign_url = urllib.parse.urlsplit(sign_service)
        if (
            sign_url.scheme not in {"http", "https"}
            or not sign_url.hostname
            or sign_url.username is not None
            or sign_url.password is not None
            or sign_url.query
            or sign_url.fragment
            or any(char in sign_service for char in "\r\n")
        ):
            raise ValueError("S3 sign service must be an HTTP(S) base URL")
        if type(self.secure) is not bool:
            raise ValueError("S3 secure must be a boolean")

        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "sign_service", sign_service)

    @property
    def destination(self) -> str:
        return f"s3://{self.bucket}"


@dataclasses.dataclass(frozen=True)
class S3FilePlan:
    relative_path: str
    size: int
    digest: str


@dataclasses.dataclass(frozen=True)
class S3UploadPlan:
    local_dir: str
    remote_dir: str
    destination: str
    files: tuple[S3FilePlan, ...]

    @property
    def files_total(self) -> int:
        return len(self.files)

    @property
    def bytes_total(self) -> int:
        return sum(item.size for item in self.files)

    @property
    def files_to_upload(self) -> tuple[S3FilePlan, ...]:
        return self.files

    @property
    def bytes_to_upload(self) -> int:
        return self.bytes_total

    @property
    def files_skipped(self) -> int:
        return 0

    @property
    def bytes_skipped(self) -> int:
        return 0

    @property
    def new_files(self) -> int:
        return self.files_total

    @property
    def changed_files(self) -> int:
        return 0

    @property
    def files_to_delete(self) -> int:
        return 0


def _hdf5_files(local_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(local_root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError("S3 upload directories must contain only files and directories")
        if not stat.S_ISREG(mode) or path.suffix.lower() not in {".h5", ".hdf5"}:
            continue
        relative_path = path.relative_to(local_root).as_posix()
        if any(char in relative_path for char in "\r\n\t"):
            raise ValueError("S3 upload file paths must not contain control characters")
        if not h5py.is_hdf5(path):
            raise ValueError(f"invalid HDF5 file: {relative_path}")
        files.append(path)
    if not files:
        raise FileNotFoundError("HDF5 export contains no .h5 or .hdf5 files")
    return files


def scan_hdf5_s3(local_dir: Path, *, config: S3UploadConfig) -> S3UploadPlan:
    """Freeze the HDF5 files that a confirmed S3 upload will transfer."""
    unresolved = Path(local_dir)
    if unresolved.is_symlink():
        raise ValueError("S3 upload directory root must not be a symlink")
    local_root = unresolved.resolve()
    if not local_root.is_dir():
        raise FileNotFoundError("local HDF5 upload directory not found")
    return S3UploadPlan(
        local_dir=str(local_root),
        remote_dir=config.prefix,
        destination=config.destination,
        files=tuple(
            S3FilePlan(
                relative_path=path.relative_to(local_root).as_posix(),
                size=path.stat().st_size,
                digest=file_digest(path),
            )
            for path in _hdf5_files(local_root)
        ),
    )


class _SignedS3Uploader:
    def __init__(self, config: S3UploadConfig) -> None:
        self.config = config
        self.sign_url = f"{config.sign_service}/api/collector/sign"
        scheme = "https" if config.secure else "http"
        self.s3_base = f"{scheme}://{config.endpoint}/{config.bucket}"
        self.token = os.environ.get(_TOKEN_ENV, "")

    def _sign(
        self,
        sign_type: str,
        key: str,
        *,
        upload_id: str = "",
        part_number: int = 0,
        content_type: str = "",
    ) -> dict[str, str]:
        body: dict[str, str | int] = {
            "type": sign_type,
            "key": key,
            "contentType": content_type,
        }
        if upload_id:
            body["uploadId"] = upload_id
        if part_number:
            body["partNumber"] = part_number
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Collector-Token"] = self.token
        request = urllib.request.Request(
            self.sign_url,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict) or result.get("code") != 0:
            raise RuntimeError("S3 signing service rejected the upload request")
        data = result.get("data")
        if not isinstance(data, dict) or not all(
            isinstance(data.get(name), str) and data[name]
            for name in ("AWSAccessKeyId", "Signature", "Date")
        ):
            raise RuntimeError("S3 signing service returned invalid credentials")
        return data

    def _request(
        self,
        method: str,
        path: str,
        *,
        sign_data: dict[str, str],
        body: bytes = b"",
        content_type: str = "",
    ) -> tuple[bytes, dict[str, str]]:
        request = urllib.request.Request(
            f"{self.s3_base}/{path}",
            data=body,
            headers={
                "Authorization": (f"AWS {sign_data['AWSAccessKeyId']}:{sign_data['Signature']}"),
                "x-amz-date": sign_data["Date"],
                "Content-Type": content_type,
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read(), dict(response.headers)

    def upload_file(
        self,
        local_path: Path,
        key: str,
        expected: S3FilePlan,
        progress_callback: Callable[[int], None] | None,
    ) -> None:
        encoded_key = urllib.parse.quote(key, safe="/=")
        response, _ = self._request(
            "POST",
            f"{encoded_key}?uploads",
            sign_data=self._sign("initMultiUpload", key),
        )
        root = ElementTree.fromstring(response)
        namespace = root.tag.split("}", 1)[0] + "}" if root.tag.startswith("{") else ""
        upload_id_item = root.find(f"{namespace}UploadId")
        upload_id = upload_id_item.text if upload_id_item is not None else ""
        if not upload_id or any(char in upload_id for char in "&?#\r\n"):
            raise RuntimeError("S3 multipart upload returned an invalid upload ID")

        parts: list[tuple[int, str]] = []
        digest = hashlib.md5(usedforsecurity=False)
        uploaded = 0
        with local_path.open("rb") as handle:
            for part_number, chunk in enumerate(iter(lambda: handle.read(_PART_SIZE), b""), 1):
                digest.update(chunk)
                content_type = "application/octet-stream"
                _, headers = self._request(
                    "PUT",
                    f"{encoded_key}?partNumber={part_number}&uploadId={upload_id}",
                    sign_data=self._sign(
                        "uploadPart",
                        key,
                        upload_id=upload_id,
                        part_number=part_number,
                        content_type=content_type,
                    ),
                    body=chunk,
                    content_type=content_type,
                )
                etag = next(
                    (value for name, value in headers.items() if name.lower() == "etag"), ""
                ).strip('"')
                if not etag:
                    raise RuntimeError("S3 multipart upload part returned no ETag")
                parts.append((part_number, etag))
                uploaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(uploaded)

        if uploaded != expected.size or digest.hexdigest() != expected.digest:
            raise ValueError("local HDF5 file changed during upload; scan again")

        complete = ElementTree.Element("CompleteMultipartUpload")
        for part_number, etag in parts:
            part = ElementTree.SubElement(complete, "Part")
            ElementTree.SubElement(part, "PartNumber").text = str(part_number)
            ElementTree.SubElement(part, "ETag").text = f'"{etag}"'
        content_type = "application/xml"
        self._request(
            "POST",
            f"{encoded_key}?uploadId={upload_id}",
            sign_data=self._sign(
                "completeUpload",
                key,
                upload_id=upload_id,
                content_type=content_type,
            ),
            body=ElementTree.tostring(complete),
            content_type=content_type,
        )


def upload_hdf5_s3(
    local_dir: Path,
    *,
    config: S3UploadConfig,
    plan: S3UploadPlan,
    progress_callback: Callable[[UploadProgress], None] | None = None,
) -> UploadResult:
    """Upload exactly the HDF5 files captured by a confirmed scan."""
    current = scan_hdf5_s3(local_dir, config=config)
    if (
        current.local_dir != plan.local_dir
        or current.remote_dir != plan.remote_dir
        or current.destination != plan.destination
    ):
        raise ValueError("S3 upload plan does not match the requested upload")
    if current.files != plan.files:
        raise ValueError("local HDF5 upload directory changed after scan; scan again")

    uploader = _SignedS3Uploader(config)
    completed_bytes = 0
    if progress_callback is not None:
        progress_callback(UploadProgress(0, plan.files_total, 0, plan.bytes_total, ""))
    for index, item in enumerate(plan.files, start=1):
        local_path = Path(plan.local_dir) / item.relative_path
        key = f"{config.prefix}/{item.relative_path}"

        def report_file(
            file_bytes: int,
            completed: int = completed_bytes,
            file_index: int = index - 1,
            name: str = item.relative_path,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    UploadProgress(
                        file_index,
                        plan.files_total,
                        completed + file_bytes,
                        plan.bytes_total,
                        name,
                    )
                )

        uploader.upload_file(
            local_path,
            key,
            item,
            report_file if progress_callback is not None else None,
        )
        completed_bytes += item.size
        if progress_callback is not None:
            progress_callback(
                UploadProgress(
                    index,
                    plan.files_total,
                    completed_bytes,
                    plan.bytes_total,
                    item.relative_path,
                )
            )
    return UploadResult(
        local_dir=plan.local_dir,
        remote_dir=config.prefix,
        destination=config.destination,
        files=plan.files_total,
        bytes=plan.bytes_total,
    )


__all__ = [
    "S3UploadConfig",
    "S3UploadPlan",
    "UploadProgress",
    "UploadResult",
    "scan_hdf5_s3",
    "upload_hdf5_s3",
]
