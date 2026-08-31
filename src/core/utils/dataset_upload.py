"""Select and run a configured dataset upload backend."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from core.utils.sftp_upload import upload_directory_sftp


@dataclasses.dataclass(frozen=True)
class DatasetUploadProgress:
    files_completed: int
    files_total: int
    bytes_completed: int
    bytes_total: int
    current_file: str


@dataclasses.dataclass(frozen=True)
class DatasetUploadResult:
    local_dir: str
    remote_dir: str
    destination: str
    files: int
    bytes: int
    skipped: bool = False


@dataclasses.dataclass(frozen=True)
class DatasetUploadSpec:
    backend: str
    root: str
    target: str
    display_root: str
    options: dict[str, Any]


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


def resolve_dataset_uploads(
    storage: Mapping[str, Any], dataset_name: str = ""
) -> tuple[DatasetUploadSpec, ...]:
    """Resolve the configured public SFTP upload backend."""
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
    return tuple(specs)


def upload_dataset_directory(
    local_dir: Path,
    specs: tuple[DatasetUploadSpec, ...],
    *,
    progress_callback: Callable[[DatasetUploadProgress], None] | None = None,
) -> DatasetUploadResult:
    """Upload a directory to every configured backend."""
    if not specs:
        raise ValueError("no dataset upload backend is configured")
    local_root = Path(local_dir).resolve()
    files = [path for path in local_root.rglob("*") if path.is_file()]
    bytes_per_backend = sum(path.stat().st_size for path in files)
    progress_by_backend: dict[str, DatasetUploadProgress] = {
        spec.backend: DatasetUploadProgress(0, len(files), 0, bytes_per_backend, "")
        for spec in specs
    }
    progress_lock = threading.Lock()

    def report(backend: str, progress: Any) -> None:
        normalized = DatasetUploadProgress(
            files_completed=int(progress.files_completed),
            files_total=int(progress.files_total),
            bytes_completed=int(progress.bytes_completed),
            bytes_total=int(progress.bytes_total),
            current_file=str(progress.current_file),
        )
        with progress_lock:
            progress_by_backend[backend] = normalized
            if progress_callback is None:
                return
            progress_callback(
                DatasetUploadProgress(
                    files_completed=sum(
                        item.files_completed for item in progress_by_backend.values()
                    ),
                    files_total=len(files) * len(specs),
                    bytes_completed=sum(
                        item.bytes_completed for item in progress_by_backend.values()
                    ),
                    bytes_total=bytes_per_backend * len(specs),
                    current_file=(
                        f"{backend}: {normalized.current_file}"
                        if normalized.current_file
                        else backend
                    ),
                )
            )

    if progress_callback is not None:
        progress_callback(
            DatasetUploadProgress(
                0,
                len(files) * len(specs),
                0,
                bytes_per_backend * len(specs),
                "",
            )
        )

    def upload_one(spec: DatasetUploadSpec) -> Any:
        if spec.backend == "sftp":
            identity_value = str(spec.options.get("identity_file", "")).strip()
            return upload_directory_sftp(
                local_root,
                host=str(spec.options["host"]),
                port=int(spec.options["port"]),
                remote_dir=spec.target,
                user=str(spec.options.get("user", "")),
                identity_file=Path(identity_value) if identity_value else None,
                progress_callback=lambda value: report(spec.backend, value),
            )
        raise ValueError(f"unsupported dataset upload backend: {spec.backend}")

    results: list[Any | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = {
            executor.submit(upload_one, spec): (index, spec) for index, spec in enumerate(specs)
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
        remote_dir=", ".join(str(result.remote_dir) for result in completed_results),
        destination=", ".join(str(result.destination) for result in completed_results),
        files=sum(int(result.files) for result in completed_results),
        bytes=sum(int(result.bytes) for result in completed_results),
        skipped=bool(completed_results)
        and all(bool(getattr(result, "skipped", False)) for result in completed_results),
    )


__all__ = [
    "DatasetUploadProgress",
    "DatasetUploadResult",
    "DatasetUploadSpec",
    "resolve_dataset_uploads",
    "upload_dataset_directory",
]
