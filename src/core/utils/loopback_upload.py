"""Scan and mirror one accepted dataset into a configured local loopback root."""

from __future__ import annotations

import shutil
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

from core.utils.upload_plan import (
    DirectoryUploadPlan,
    RemoteFilePlan,
    UploadFilePlan,
    UploadProgress,
    UploadResult,
    file_digest,
)


def _files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError("loopback upload directories must contain only files and directories")
        if stat.S_ISREG(mode):
            files.append(path)
    return files


def _validate_roots(local_dir: Path, remote_dir: str) -> tuple[Path, Path]:
    if Path(local_dir).is_symlink() or Path(remote_dir).expanduser().is_symlink():
        raise ValueError("loopback upload directory roots must not be symlinks")
    local_root = Path(local_dir).resolve()
    remote_root = Path(remote_dir).expanduser().resolve()
    if not local_root.is_dir():
        raise FileNotFoundError("local upload directory not found")
    if remote_root == Path("/") or local_root == remote_root:
        raise ValueError("loopback upload target must be a distinct non-root directory")
    if local_root in remote_root.parents or remote_root in local_root.parents:
        raise ValueError("loopback upload source and target must not overlap")
    if remote_root.exists() and not remote_root.is_dir():
        raise ValueError("loopback upload target must be a directory")
    return local_root, remote_root


def scan_directory_loopback(local_dir: Path, *, remote_dir: str) -> DirectoryUploadPlan:
    """Compare every accepted file with one local directory used as the remote."""
    local_root, remote_root = _validate_roots(local_dir, remote_dir)
    local_files = _files(local_root)
    remote_exists = remote_root.exists()
    remote_digests = (
        {
            path.relative_to(remote_root).as_posix(): file_digest(path)
            for path in _files(remote_root)
        }
        if remote_exists
        else {}
    )
    planned_files = []
    for path in local_files:
        relative_path = path.relative_to(local_root).as_posix()
        digest = file_digest(path)
        remote_digest = remote_digests.get(relative_path)
        planned_files.append(
            UploadFilePlan(
                relative_path=relative_path,
                size=path.stat().st_size,
                digest=digest,
                action=(
                    "new"
                    if remote_digest is None
                    else "same"
                    if remote_digest == digest
                    else "changed"
                ),
            )
        )
    local_paths = {item.relative_path for item in planned_files}
    return DirectoryUploadPlan(
        local_dir=str(local_root),
        remote_dir=str(remote_root),
        destination="loopback",
        remote_exists=remote_exists,
        files=tuple(planned_files),
        remote_only_files=tuple(
            RemoteFilePlan(relative_path, remote_digests[relative_path])
            for relative_path in sorted(remote_digests.keys() - local_paths)
        ),
    )


def _validate_plan(plan: DirectoryUploadPlan, local_root: Path, remote_root: Path) -> None:
    if plan.local_dir != str(local_root) or plan.remote_dir != str(remote_root):
        raise ValueError("loopback upload plan does not match the requested upload")
    current = scan_directory_loopback(local_root, remote_dir=str(remote_root))
    planned_sources = tuple((item.relative_path, item.size, item.digest) for item in plan.files)
    current_sources = tuple((item.relative_path, item.size, item.digest) for item in current.files)
    if current_sources != planned_sources:
        raise ValueError("local upload directory changed after scan; scan again")
    if current != plan:
        raise ValueError("remote upload directory changed after scan; scan again")


def upload_directory_loopback(
    local_dir: Path,
    *,
    remote_dir: str,
    plan: DirectoryUploadPlan,
    progress_callback: Callable[[UploadProgress], None] | None = None,
) -> UploadResult:
    """Apply one confirmed loopback scan plan without changing the source."""
    local_root, remote_root = _validate_roots(local_dir, remote_dir)
    _validate_plan(plan, local_root, remote_root)
    upload_items = plan.files_to_upload
    if not upload_items and not plan.remote_only_files:
        if progress_callback is not None:
            progress_callback(UploadProgress(0, 0, 0, 0, ""))
        return UploadResult(str(local_root), str(remote_root), "loopback", 0, 0, skipped=True)

    remote_root.parent.mkdir(parents=True, exist_ok=True)
    staging = remote_root.parent / f".{remote_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    bytes_total = sum(item.size for item in upload_items)
    bytes_completed = 0
    if progress_callback is not None:
        progress_callback(UploadProgress(0, len(upload_items), 0, bytes_total, ""))
    try:
        for index, item in enumerate(upload_items, start=1):
            source = local_root / item.relative_path
            staged = staging / item.relative_path
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            bytes_completed += item.size
            if progress_callback is not None:
                progress_callback(
                    UploadProgress(
                        index,
                        len(upload_items),
                        bytes_completed,
                        bytes_total,
                        item.relative_path,
                    )
                )

        if not plan.remote_exists:
            staging.replace(remote_root)
        else:
            for item in upload_items:
                staged = staging / item.relative_path
                target = remote_root / item.relative_path
                if target.exists() and not target.is_file():
                    raise ValueError("loopback target has a file/directory conflict")
                target.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(target)
            for item in plan.remote_only_files:
                target = remote_root / item.relative_path
                if not target.is_file() or file_digest(target) != item.digest:
                    raise ValueError("remote upload directory changed after scan; scan again")
                target.unlink()
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return UploadResult(
        local_dir=str(local_root),
        remote_dir=str(remote_root),
        destination="loopback",
        files=len(upload_items),
        bytes=bytes_total,
        files_deleted=len(plan.remote_only_files),
    )


__all__ = [
    "DirectoryUploadPlan",
    "UploadProgress",
    "UploadResult",
    "scan_directory_loopback",
    "upload_directory_loopback",
]
