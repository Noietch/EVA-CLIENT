"""Shared immutable progress, results, and directory upload plans."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class UploadProgress:
    files_completed: int
    files_total: int
    bytes_completed: int
    bytes_total: int
    current_file: str


@dataclass(frozen=True)
class UploadResult:
    local_dir: str
    remote_dir: str
    destination: str
    files: int
    bytes: int
    skipped: bool = False
    files_deleted: int = 0


@dataclass(frozen=True)
class UploadFilePlan:
    relative_path: str
    size: int
    digest: str
    action: str

    @property
    def upload(self) -> bool:
        return self.action != "same"


@dataclass(frozen=True)
class RemoteFilePlan:
    relative_path: str
    digest: str


@dataclass(frozen=True)
class DirectoryUploadPlan:
    local_dir: str
    remote_dir: str
    destination: str
    remote_exists: bool
    files: tuple[UploadFilePlan, ...]
    remote_only_files: tuple[RemoteFilePlan, ...] = ()

    @property
    def files_total(self) -> int:
        return len(self.files)

    @property
    def bytes_total(self) -> int:
        return sum(item.size for item in self.files)

    @property
    def files_to_upload(self) -> tuple[UploadFilePlan, ...]:
        return tuple(item for item in self.files if item.upload)

    @property
    def bytes_to_upload(self) -> int:
        return sum(item.size for item in self.files_to_upload)

    @property
    def files_skipped(self) -> int:
        return self.files_total - len(self.files_to_upload)

    @property
    def bytes_skipped(self) -> int:
        return self.bytes_total - self.bytes_to_upload

    @property
    def new_files(self) -> int:
        return sum(item.action == "new" for item in self.files)

    @property
    def changed_files(self) -> int:
        return sum(item.action == "changed" for item in self.files)

    @property
    def files_to_delete(self) -> int:
        return len(self.remote_only_files)
