"""Select and run a configured dataset upload backend."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from core.utils.loopback_upload import (
    scan_directory_loopback,
    upload_directory_loopback,
)
from core.utils.s3_upload import (
    S3UploadConfig,
    S3UploadPlan,
    scan_hdf5_s3,
    upload_hdf5_s3,
)
from core.utils.sftp_upload import (
    scan_directory_sftp,
    upload_directory_sftp,
)
from core.utils.upload_plan import DirectoryUploadPlan, UploadProgress, UploadResult

_UPLOAD_RECEIPT_NAME = "upload_receipt.json"


@dataclasses.dataclass(frozen=True)
class DatasetUploadResult:
    local_dir: str
    remote_dir: str
    destination: str
    files: int
    bytes: int
    skipped: bool = False
    files_total: int = 0
    bytes_total: int = 0
    files_skipped: int = 0
    bytes_skipped: int = 0
    files_deleted: int = 0


@dataclasses.dataclass(frozen=True)
class DatasetUploadSpec:
    backend: str
    root: str
    target: str
    display_root: str
    options: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class DatasetUploadPlan:
    """Frozen per-backend scan result used by a confirmed upload."""

    local_dir: str
    backends: tuple[tuple[DatasetUploadSpec, DirectoryUploadPlan | S3UploadPlan], ...]

    @property
    def files_total(self) -> int:
        return sum(plan.files_total for _, plan in self.backends)

    @property
    def bytes_total(self) -> int:
        return sum(plan.bytes_total for _, plan in self.backends)

    @property
    def files_to_upload(self) -> int:
        return sum(len(plan.files_to_upload) for _, plan in self.backends)

    @property
    def bytes_to_upload(self) -> int:
        return sum(plan.bytes_to_upload for _, plan in self.backends)

    @property
    def files_skipped(self) -> int:
        return sum(plan.files_skipped for _, plan in self.backends)

    @property
    def bytes_skipped(self) -> int:
        return sum(plan.bytes_skipped for _, plan in self.backends)

    @property
    def new_files(self) -> int:
        return sum(plan.new_files for _, plan in self.backends)

    @property
    def changed_files(self) -> int:
        return sum(plan.changed_files for _, plan in self.backends)

    @property
    def files_to_delete(self) -> int:
        return sum(plan.files_to_delete for _, plan in self.backends)


def _episode_indices(dataset_dir: Path) -> list[int]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    return sorted(
        int(json.loads(line)["episode_index"])
        for line in path.read_text().splitlines()
        if line.strip()
    )


def _export_signature(local_dir: Path) -> dict[str, int]:
    """Cheap fingerprint of an export so a receipt cannot outlive its content.

    Exporting publishes a new directory, and every file the export gains or
    loses touches the directory holding it, so directory identity plus each
    directory's mtime and entry count identify the content without statting
    every recorded file. ``mtime_ns`` is the sum over the export's directories.
    """
    if not (local_dir / "meta" / "episodes.jsonl").is_file():
        return {}
    inode = int(local_dir.stat().st_ino)
    directories = 0
    entries = 0
    mtime_ns = 0
    for parent, directory_names, file_names in os.walk(local_dir):
        directories += 1
        entries += len(directory_names) + len(file_names)
        mtime_ns += Path(parent).stat().st_mtime_ns
    return {
        "inode": inode,
        "directories": directories,
        "entries": entries,
        "mtime_ns": mtime_ns,
    }


def upload_receipt_path(local_dir: Path) -> Path:
    """Receipts sit beside the export, one per exported copy."""
    local_root = Path(local_dir).resolve()
    return local_root.parent / f"{local_root.name}_{_UPLOAD_RECEIPT_NAME}"


def record_dataset_upload_receipt(
    local_dir: Path,
    *,
    source_dir: Path,
    destination: str,
    remote_dir: str,
) -> Path:
    """Persist which source episodes an uploaded export was proven to hold."""
    local_root = Path(local_dir).resolve()
    source_root = Path(source_dir).resolve()
    if not (local_root / "meta" / "episodes.jsonl").is_file():
        raise ValueError(f"exported dataset is missing meta/episodes.jsonl: {local_root}")
    receipt_path = upload_receipt_path(local_root)
    temporary = receipt_path.with_suffix(f"{receipt_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "uploaded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "source_dir": str(source_root),
                "source_episode_indices": _episode_indices(source_root),
                "dataset_format": local_root.name.removeprefix(f"{source_root.name}_"),
                "export_signature": _export_signature(local_root),
                "destination": str(destination),
                "remote_dir": str(remote_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(receipt_path)
    return receipt_path


def uploaded_source_episode_indices(dataset_dir: Path) -> set[int]:
    """Return source episode indices backed by current upload receipts."""
    dataset_root = Path(dataset_dir).resolve()
    uploaded: set[int] = set()
    for receipt_path in sorted(
        dataset_root.parent.glob(f"{dataset_root.name}_*_{_UPLOAD_RECEIPT_NAME}")
    ):
        export_name = receipt_path.name.removesuffix(f"_{_UPLOAD_RECEIPT_NAME}")
        uploaded.update(
            uploaded_source_episode_indices_for_export(
                receipt_path.parent / export_name, dataset_root
            )
        )
    return uploaded


def uploaded_source_episode_indices_for_export(
    local_dir: Path,
    dataset_dir: Path | None = None,
) -> set[int]:
    """Return source indices proven uploaded for one current export copy."""
    local_root = Path(local_dir).resolve()
    receipt_path = upload_receipt_path(local_root)
    try:
        receipt = json.loads(receipt_path.read_text())
        if not isinstance(receipt, dict):
            return set()
        receipt_indices = receipt.get("source_episode_indices")
        receipt_source = Path(str(receipt.get("source_dir") or "")).resolve()
        expected_source = Path(dataset_dir).resolve() if dataset_dir is not None else receipt_source
        if (
            int(receipt.get("schema_version", 0)) != 3
            or receipt_source != expected_source
            or receipt.get("export_signature") != _export_signature(local_root)
            or not isinstance(receipt_indices, list)
            or any(type(value) is not int or value < 0 for value in receipt_indices)
        ):
            return set()
    except (OSError, TypeError, ValueError):
        return set()
    return set(receipt_indices)


def _sftp_identity_file(spec: DatasetUploadSpec) -> Path | None:
    identity_value = str(spec.options.get("identity_file", "")).strip()
    return Path(identity_value) if identity_value else None


def _s3_config(spec: DatasetUploadSpec) -> S3UploadConfig:
    return S3UploadConfig(
        endpoint=str(spec.options["endpoint"]),
        bucket=str(spec.options["bucket"]),
        prefix=spec.target,
        sign_service=str(spec.options["sign_service"]),
        secure=spec.options["secure"],
    )


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized_name = str(dataset_name).strip()
    if not normalized_name:
        return ""
    path = PurePosixPath(normalized_name)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError("dataset upload name must be one path component")
    return path.name


def _normalize_remote_root(remote_root: str) -> str:
    normalized_root = str(remote_root)
    canonical_root = normalized_root.rstrip("/") or "/"
    path = PurePosixPath(canonical_root)
    if (
        any(char in normalized_root for char in "\r\n")
        or not path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != canonical_root
    ):
        raise ValueError("dataset upload root must be an absolute canonical path")
    return path.as_posix()


def _normalize_loopback_root(remote_root: str) -> str:
    normalized_root = str(remote_root).strip()
    path = Path(normalized_root)
    if (
        not normalized_root
        or path == Path(".")
        or any(char in normalized_root for char in "\r\n")
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != normalized_root.rstrip("/")
    ):
        raise ValueError("dataset loopback root must be a canonical directory path")
    resolved = path.expanduser().resolve()
    if resolved == Path("/"):
        raise ValueError("dataset loopback root must not be the filesystem root")
    return str(resolved)


def resolve_dataset_uploads(
    storage: Mapping[str, Any], dataset_name: str = ""
) -> tuple[DatasetUploadSpec, ...]:
    """Resolve configured dataset upload backends."""
    normalized_name = _normalize_dataset_name(dataset_name)

    specs: list[DatasetUploadSpec] = []
    sftp = storage.get("sftp") or {}
    if sftp:
        root = _normalize_remote_root(str(sftp.get("remote_dir", "")))
        target = str(PurePosixPath(root) / normalized_name) if normalized_name else root
        host = str(sftp.get("host", "")).strip()
        port = int(sftp.get("port", 22))
        specs.append(
            DatasetUploadSpec(
                backend="sftp",
                root=root,
                target=target,
                display_root=f"{host}:{port} · {root}",
                options={
                    "host": host,
                    "port": port,
                    "user": str(sftp.get("user", "")).strip(),
                    "identity_file": str(sftp.get("identity_file", "")).strip(),
                },
            )
        )
    s3 = storage.get("s3") or {}
    if s3:
        config = S3UploadConfig(
            endpoint=str(s3.get("endpoint", "")),
            bucket=str(s3.get("bucket", "")),
            prefix=str(s3.get("prefix", "")),
            sign_service=str(s3.get("sign_service", "")),
            secure=s3.get("secure", False),
        )
        target = (
            str(PurePosixPath(config.prefix) / normalized_name)
            if normalized_name
            else config.prefix
        )
        specs.append(
            DatasetUploadSpec(
                backend="s3",
                root=config.prefix,
                target=target,
                display_root=f"s3://{config.bucket}/{config.prefix}",
                options={
                    "endpoint": config.endpoint,
                    "bucket": config.bucket,
                    "sign_service": config.sign_service,
                    "secure": config.secure,
                },
            )
        )
    loopback = storage.get("loopback") or {}
    if loopback:
        root = _normalize_loopback_root(str(loopback.get("remote_dir", "")))
        target = str(Path(root) / normalized_name) if normalized_name else root
        specs.append(
            DatasetUploadSpec(
                backend="loopback",
                root=root,
                target=target,
                display_root=f"loopback · {root}",
                options={},
            )
        )
    return tuple(specs)


def scan_dataset_directory(
    local_dir: Path,
    specs: tuple[DatasetUploadSpec, ...],
) -> DatasetUploadPlan:
    """Scan every local file against each configured upload target."""
    if not specs:
        raise ValueError("no dataset upload backend is configured")
    local_root = Path(local_dir).resolve()

    def scan_one(
        spec: DatasetUploadSpec,
    ) -> DirectoryUploadPlan | S3UploadPlan:
        if spec.backend == "sftp":
            return scan_directory_sftp(
                local_root,
                host=str(spec.options["host"]),
                port=int(spec.options["port"]),
                remote_dir=spec.target,
                user=str(spec.options.get("user", "")),
                identity_file=_sftp_identity_file(spec),
            )
        if spec.backend == "loopback":
            return scan_directory_loopback(local_root, remote_dir=spec.target)
        if spec.backend == "s3":
            return scan_hdf5_s3(local_root, config=_s3_config(spec))
        raise ValueError(f"unsupported dataset upload backend: {spec.backend}")

    plans: list[DirectoryUploadPlan | S3UploadPlan | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = {
            executor.submit(scan_one, spec): (index, spec) for index, spec in enumerate(specs)
        }
        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                plans[index] = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"{spec.backend} dataset scan failed: {type(error).__name__}: {error}"
                ) from error
    return DatasetUploadPlan(
        local_dir=str(local_root),
        backends=tuple(
            (spec, plan) for spec, plan in zip(specs, plans, strict=True) if plan is not None
        ),
    )


def upload_dataset_directory(
    local_dir: Path,
    specs: tuple[DatasetUploadSpec, ...],
    *,
    plan: DatasetUploadPlan,
    progress_callback: Callable[[UploadProgress], None] | None = None,
) -> DatasetUploadResult:
    """Execute a confirmed scan plan against every configured backend."""
    if not specs:
        raise ValueError("no dataset upload backend is configured")
    local_root = Path(local_dir).resolve()
    if plan.local_dir != str(local_root) or tuple(spec for spec, _ in plan.backends) != specs:
        raise ValueError("dataset upload plan does not match the requested upload")
    progress_by_backend: dict[str, UploadProgress] = {
        spec.backend: UploadProgress(
            0,
            len(backend_plan.files_to_upload),
            0,
            backend_plan.bytes_to_upload,
            "",
        )
        for spec, backend_plan in plan.backends
    }
    progress_lock = threading.Lock()

    def report(backend: str, progress: UploadProgress) -> None:
        with progress_lock:
            progress_by_backend[backend] = progress
            if progress_callback is None:
                return
            progress_callback(
                UploadProgress(
                    files_completed=sum(
                        item.files_completed for item in progress_by_backend.values()
                    ),
                    files_total=plan.files_to_upload,
                    bytes_completed=sum(
                        item.bytes_completed for item in progress_by_backend.values()
                    ),
                    bytes_total=plan.bytes_to_upload,
                    current_file=(
                        f"{backend}: {progress.current_file}" if progress.current_file else backend
                    ),
                )
            )

    if progress_callback is not None:
        progress_callback(
            UploadProgress(
                0,
                plan.files_to_upload,
                0,
                plan.bytes_to_upload,
                "",
            )
        )

    def upload_one(
        spec: DatasetUploadSpec,
        backend_plan: DirectoryUploadPlan | S3UploadPlan,
    ) -> UploadResult:
        if spec.backend == "sftp":
            if not isinstance(backend_plan, DirectoryUploadPlan):
                raise ValueError("SFTP upload plan type mismatch")
            return upload_directory_sftp(
                local_root,
                host=str(spec.options["host"]),
                port=int(spec.options["port"]),
                remote_dir=spec.target,
                user=str(spec.options.get("user", "")),
                identity_file=_sftp_identity_file(spec),
                plan=backend_plan,
                progress_callback=lambda value: report(spec.backend, value),
            )
        if spec.backend == "loopback":
            if not isinstance(backend_plan, DirectoryUploadPlan):
                raise ValueError("loopback upload plan type mismatch")
            return upload_directory_loopback(
                local_root,
                remote_dir=spec.target,
                plan=backend_plan,
                progress_callback=lambda value: report(spec.backend, value),
            )
        if spec.backend == "s3":
            if not isinstance(backend_plan, S3UploadPlan):
                raise ValueError("S3 upload plan type mismatch")
            return upload_hdf5_s3(
                local_root,
                config=_s3_config(spec),
                plan=backend_plan,
                progress_callback=lambda value: report(spec.backend, value),
            )
        raise ValueError(f"unsupported dataset upload backend: {spec.backend}")

    results: list[UploadResult | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = {
            executor.submit(upload_one, spec, backend_plan): (index, spec)
            for index, (spec, backend_plan) in enumerate(plan.backends)
        }
        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"{spec.backend} dataset upload failed: {type(error).__name__}: {error}"
                ) from error
    completed_results = [result for result in results if result is not None]

    return DatasetUploadResult(
        local_dir=str(local_root),
        remote_dir=", ".join(result.remote_dir for result in completed_results),
        destination=", ".join(result.destination for result in completed_results),
        files=sum(result.files for result in completed_results),
        bytes=sum(result.bytes for result in completed_results),
        skipped=bool(completed_results) and all(result.skipped for result in completed_results),
        files_total=plan.files_total,
        bytes_total=plan.bytes_total,
        files_skipped=plan.files_skipped,
        bytes_skipped=plan.bytes_skipped,
        files_deleted=sum(result.files_deleted for result in completed_results),
    )


__all__ = [
    "DatasetUploadPlan",
    "DatasetUploadResult",
    "DatasetUploadSpec",
    "record_dataset_upload_receipt",
    "resolve_dataset_uploads",
    "scan_dataset_directory",
    "upload_dataset_directory",
    "uploaded_source_episode_indices",
]
