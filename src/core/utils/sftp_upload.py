"""Upload a finalized dataset directory through the system OpenSSH client."""

from __future__ import annotations

import dataclasses
import errno
import os
import pty
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath


@dataclasses.dataclass(frozen=True)
class SftpUploadResult:
    local_dir: str
    remote_dir: str
    destination: str
    files: int
    bytes: int


@dataclasses.dataclass(frozen=True)
class SftpUploadProgress:
    files_completed: int
    files_total: int
    bytes_completed: int
    bytes_total: int
    current_file: str


def _destination(host: str, user: str) -> str:
    normalized_host = host.strip()
    if not normalized_host or any(char in normalized_host for char in "\r\n"):
        raise ValueError("SFTP host must not be empty")
    normalized_user = user.strip()
    if any(char in normalized_user for char in "\r\n@"):
        raise ValueError("SFTP user is invalid")
    return f"{normalized_user}@{normalized_host}" if normalized_user else normalized_host


def _sftp_quote(value: str) -> str:
    if any(char in value for char in "\r\n"):
        raise ValueError("SFTP paths must not contain newlines")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _remote_mkdir_paths(
    remote_path: PurePosixPath, relative_dirs: set[Path]
) -> list[PurePosixPath]:
    """Return every remote directory needed by an upload, in parent-first order."""
    directories: set[PurePosixPath] = set()

    current = PurePosixPath("/")
    for component in remote_path.parts[1:]:
        current /= component
        directories.add(current)

    for relative_dir in relative_dirs:
        current = remote_path
        for component in relative_dir.parts:
            current /= component
            directories.add(current)

    return sorted(directories, key=lambda path: (len(path.parts), path.as_posix()))


def _run_sftp_command(
    process: subprocess.Popen[bytes],
    master_fd: int,
    command: str,
    output_callback: Callable[[bytes], None] | None = None,
) -> str:
    marker = b"Remote working directory:"
    os.write(master_fd, command.encode() + b"\npwd\n")
    output = bytearray()
    while marker not in output:
        try:
            chunk = os.read(master_fd, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                chunk = b""
            else:
                raise
        if not chunk:
            return_code = process.poll()
            detail = output.decode(errors="replace").strip()
            raise RuntimeError(
                f"SFTP command failed with exit code {return_code}"
                + (f": {detail}" if detail else "")
            )
        output.extend(chunk)
        if output_callback is not None:
            output_callback(chunk)
    return output.decode(errors="replace")


def _upload_files_sftp(
    sftp: str,
    sftp_args: list[str],
    local_dir: Path,
    remote_path: PurePosixPath,
    files: list[Path],
    progress_callback: Callable[[SftpUploadProgress], None] | None,
) -> None:
    master_fd, slave_fd = pty.openpty()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [sftp, "-b", "-", *sftp_args],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1

        relative_dirs = sorted(
            {path.parent.relative_to(local_dir) for path in files if path.parent != local_dir},
            key=lambda path: (len(path.parts), path.as_posix()),
        )
        # SFTP mkdir is not recursive. The leading '-' keeps already-existing
        # directories harmless while every missing parent is created first.
        mkdir_commands = [
            f"-mkdir {_sftp_quote(str(path))}"
            for path in _remote_mkdir_paths(remote_path, set(relative_dirs))
        ]
        _run_sftp_command(process, master_fd, "\n".join(mkdir_commands))

        bytes_total = sum(path.stat().st_size for path in files)
        bytes_completed = 0
        if progress_callback is not None:
            progress_callback(SftpUploadProgress(0, len(files), 0, bytes_total, ""))
        for index, local_path in enumerate(files, start=1):
            relative_path = local_path.relative_to(local_dir).as_posix()
            remote_file = str(remote_path / relative_path)
            file_size = local_path.stat().st_size
            last_percent = [-1]
            progress_tail = [b""]

            def update_file_progress(
                chunk: bytes,
                *,
                file_index: int = index,
                bytes_before: int = bytes_completed,
                current_file_size: int = file_size,
                current_file: str = relative_path,
                percent_state: list[int] = last_percent,
                tail_state: list[bytes] = progress_tail,
            ) -> None:
                progress_output = tail_state[0] + chunk
                tail_state[0] = progress_output[-32:]
                matches = re.findall(rb"(?<!\d)(\d{1,3})%", progress_output)
                if not matches or progress_callback is None:
                    return
                for match in matches:
                    percent = min(100, int(match))
                    if percent <= percent_state[0]:
                        continue
                    percent_state[0] = percent
                    progress_callback(
                        SftpUploadProgress(
                            file_index - 1,
                            len(files),
                            bytes_before + (current_file_size * percent // 100),
                            bytes_total,
                            current_file,
                        )
                    )

            _run_sftp_command(
                process,
                master_fd,
                f"put {_sftp_quote(str(local_path))} {_sftp_quote(remote_file)}",
                update_file_progress,
            )
            bytes_completed += file_size
            if progress_callback is not None:
                progress_callback(
                    SftpUploadProgress(
                        index,
                        len(files),
                        bytes_completed,
                        bytes_total,
                        relative_path,
                    )
                )
        os.write(master_fd, b"quit\n")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"SFTP upload failed with exit code {return_code}")
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def upload_directory_sftp(
    local_dir: Path,
    *,
    host: str,
    port: int,
    remote_dir: str,
    user: str = "",
    identity_file: Path | None = None,
    progress_callback: Callable[[SftpUploadProgress], None] | None = None,
) -> SftpUploadResult:
    """Recursively upload one directory, replacing files with matching remote paths."""
    local_dir = Path(local_dir).resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"upload directory not found: {local_dir}")
    if not 1 <= int(port) <= 65535:
        raise ValueError("SFTP port must be in [1, 65535]")
    remote_path = PurePosixPath(remote_dir)
    if not remote_path.is_absolute() or remote_path.name in {"", ".", ".."}:
        raise ValueError("SFTP remote directory must be an absolute dataset path")
    destination = _destination(host, user)

    identity_args: list[str] = []
    if identity_file is not None:
        identity = Path(identity_file).expanduser().resolve()
        if not identity.is_file():
            raise FileNotFoundError(f"SFTP identity file not found: {identity}")
        identity_args = ["-i", str(identity)]

    sftp = shutil.which("sftp")
    if sftp is None:
        raise RuntimeError("OpenSSH sftp executable is required")
    target = str(remote_path)
    files = sorted(path for path in local_dir.rglob("*") if path.is_file())
    try:
        _upload_files_sftp(
            sftp,
            [
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-P",
                str(port),
                *identity_args,
                destination,
            ],
            local_dir,
            remote_path,
            files,
            progress_callback,
        )
    except Exception as error:
        raise RuntimeError(
            "SFTP upload failed; the remote target may be partial: " + str(error)
        ) from error
    return SftpUploadResult(
        local_dir=str(local_dir),
        remote_dir=target,
        destination=destination,
        files=len(files),
        bytes=sum(path.stat().st_size for path in files),
    )


__all__ = ["SftpUploadProgress", "SftpUploadResult", "upload_directory_sftp"]
