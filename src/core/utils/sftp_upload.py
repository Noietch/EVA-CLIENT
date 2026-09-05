"""Upload a finalized dataset directory through the system OpenSSH client."""

from __future__ import annotations

import errno
import os
import pty
import re
import shlex
import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from core.utils.upload_plan import (
    DirectoryUploadPlan,
    RemoteFilePlan,
    UploadFilePlan,
    UploadProgress,
    UploadResult,
    file_digest,
)


def _destination(host: str, user: str) -> str:
    normalized_host = host.strip()
    if (
        not normalized_host
        or normalized_host.startswith("-")
        or any(char in normalized_host for char in "\r\n")
    ):
        raise ValueError("SFTP host must not be empty")
    normalized_user = user.strip()
    if normalized_user.startswith("-") or any(char in normalized_user for char in "\r\n@"):
        raise ValueError("SFTP user is invalid")
    return f"{normalized_user}@{normalized_host}" if normalized_user else normalized_host


def _sftp_quote(value: str) -> str:
    if any(char in value for char in "\r\n"):
        raise ValueError("SFTP paths must not contain newlines")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _remote_dataset_path(remote_dir: str) -> PurePosixPath:
    remote_path = PurePosixPath(remote_dir)
    if (
        any(char in remote_dir for char in "\r\n")
        or not remote_path.is_absolute()
        or remote_path == PurePosixPath("/")
        or remote_path.name in {"", ".", ".."}
        or ".." in remote_path.parts
        or "." in remote_path.parts
        or remote_path.as_posix() != remote_dir
    ):
        raise ValueError("SFTP remote directory must be an absolute dataset path")
    return remote_path


def _ssh_payload(script: str, *script_args: str) -> str:
    quoted_args = " ".join(shlex.quote(value) for value in script_args)
    if quoted_args:
        return f"set -- {quoted_args}\n{script}"
    return f"set --\n{script}"


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
            raise RuntimeError(f"SFTP command failed with exit code {return_code}")
        output.extend(chunk)
        if output_callback is not None:
            output_callback(chunk)
    return output.decode(errors="replace")


def _ssh_options(destination: str, port: int, identity_file: Path | None) -> tuple[str, list[str]]:
    ssh = shutil.which("ssh")
    if ssh is None:
        raise RuntimeError("OpenSSH ssh executable is required")
    identity_args: list[str] = []
    if identity_file is not None:
        identity = Path(identity_file).expanduser().resolve()
        if not identity.is_file():
            raise FileNotFoundError("SFTP identity file not found")
        identity_args = ["-i", str(identity)]
    return ssh, [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-p",
        str(port),
        *identity_args,
        destination,
    ]


def _run_ssh_script(
    destination: str,
    port: int,
    identity_file: Path | None,
    script: str,
    *script_args: str,
) -> str:
    ssh, ssh_args = _ssh_options(destination, port, identity_file)
    try:
        result = subprocess.run(
            [ssh, *ssh_args, "sh", "-s", "--"],
            check=True,
            capture_output=True,
            input=_ssh_payload(script, *script_args),
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError("remote dataset publish failed") from error


def _remote_cleanup_script() -> str:
    return (
        "set -eu\n"
        'staging="$1"\n'
        'if [ -L "$staging" ]; then\n'
        "    exit 43\n"
        "fi\n"
        'if [ -e "$staging" ]; then\n'
        '    rm -rf -- "$staging"\n'
        "fi\n"
    )


def _remote_target_exists_script() -> str:
    return (
        "set -eu\n"
        'target="$1"\n'
        'if [ -e "$target" ] || [ -L "$target" ]; then\n'
        '    printf "exists\\n"\n'
        "else\n"
        '    printf "missing\\n"\n'
        "fi\n"
    )


def _remote_file_manifest_script() -> str:
    """Print md5 and absolute path for every regular file below a dataset."""
    return (
        "set -eu\n"
        'root="$1"\n'
        'if [ -L "$root" ] || [ ! -d "$root" ]; then\n'
        "    exit 44\n"
        "fi\n"
        'find "$root" -type f -exec md5sum -- {} +\n'
    )


def _remote_publish_script() -> str:
    return (
        "set -eu\n"
        'preferred="$1"\n'
        'staging="$2"\n'
        "cleanup() {\n"
        "    status=$?\n"
        '    if [ "$status" -ne 0 ] && [ -d "$staging" ]; then\n'
        '        rm -rf -- "$staging" || true\n'
        "    fi\n"
        '    exit "$status"\n'
        "}\n"
        "trap cleanup EXIT INT TERM\n"
        'if [ -L "$staging" ] || [ ! -d "$staging" ]; then\n'
        "    exit 45\n"
        "fi\n"
        'if [ ! -e "$preferred" ] && [ ! -L "$preferred" ]; then\n'
        '    if mv -T -n -- "$staging" "$preferred" 2>/dev/null; then\n'
        '        if [ ! -e "$staging" ] && [ ! -L "$staging" ]; then\n'
        '            printf "published\\n%s\\n" "$preferred"\n'
        "            trap - EXIT INT TERM\n"
        "            exit 0\n"
        "        fi\n"
        "    fi\n"
        "fi\n"
        'if [ -L "$preferred" ] || [ ! -d "$preferred" ]; then\n'
        "    exit 46\n"
        "fi\n"
        'if find "$preferred" -type l -print -quit | grep -q .; then\n'
        "    exit 47\n"
        "fi\n"
        "validate_delete_target() {\n"
        '    expected_digest="$1"\n'
        '    relative_path="$2"\n'
        '    case "$relative_path" in\n'
        '        ""|.|/*|./*|../*|*/../*|*/..) exit 48 ;;\n'
        "    esac\n"
        '    delete_target="$preferred/$relative_path"\n'
        '    if [ -L "$delete_target" ] || [ ! -f "$delete_target" ]; then\n'
        "        exit 48\n"
        "    fi\n"
        '    actual_digest=$(md5sum -- "$delete_target")\n'
        "    actual_digest=${actual_digest%% *}\n"
        '    if [ "$actual_digest" != "$expected_digest" ]; then\n'
        "        exit 49\n"
        "    fi\n"
        "}\n"
        "(\n"
        '    while [ "$#" -gt 2 ]; do\n'
        '        validate_delete_target "$3" "$4"\n'
        "        shift 2\n"
        "    done\n"
        '    if [ "$#" -ne 2 ]; then\n'
        "        exit 48\n"
        "    fi\n"
        ")\n"
        'cp -a -- "$staging"/. "$preferred"/\n'
        'while [ "$#" -gt 2 ]; do\n'
        '    validate_delete_target "$3" "$4"\n'
        '    rm -f -- "$delete_target"\n'
        "    shift 2\n"
        "done\n"
        'if [ "$#" -ne 2 ]; then\n'
        "    exit 48\n"
        "fi\n"
        'rm -rf -- "$staging"\n'
        'printf "merged\\n%s\\n" "$preferred"\n'
        "trap - EXIT INT TERM\n"
        "exit 0\n"
    )


def _remote_target_exists(
    destination: str,
    port: int,
    identity_file: Path | None,
    target: PurePosixPath,
) -> bool:
    state = _run_ssh_script(
        destination,
        port,
        identity_file,
        _remote_target_exists_script(),
        target.as_posix(),
    ).strip()
    if state not in {"exists", "missing"}:
        raise RuntimeError("remote target check returned an invalid result")
    return state == "exists"


def _publish_remote_dataset(
    destination: str,
    port: int,
    identity_file: Path | None,
    preferred: PurePosixPath,
    staging: PurePosixPath,
    remote_only_files: tuple[RemoteFilePlan, ...] = (),
) -> tuple[str, bool]:
    delete_args = [
        value for item in remote_only_files for value in (item.digest, item.relative_path)
    ]
    publish_output = _run_ssh_script(
        destination,
        port,
        identity_file,
        _remote_publish_script(),
        preferred.as_posix(),
        staging.as_posix(),
        *delete_args,
    ).splitlines()
    if len(publish_output) != 2 or publish_output[0] not in {"published", "merged"}:
        raise RuntimeError("remote dataset publish returned an invalid result")
    published_value = _remote_dataset_path(publish_output[1]).as_posix()
    if published_value != preferred.as_posix():
        raise RuntimeError("remote dataset publish returned an invalid destination")
    return published_value, False


def _cleanup_remote_staging(
    destination: str,
    port: int,
    identity_file: Path | None,
    staging: PurePosixPath,
) -> None:
    try:
        _run_ssh_script(
            destination,
            port,
            identity_file,
            _remote_cleanup_script(),
            staging.as_posix(),
        )
    except RuntimeError:
        return


def _validate_local_directory(local_dir: Path) -> Path:
    unresolved = Path(local_dir)
    if unresolved.is_symlink():
        raise ValueError("upload directory root must not be a symlink")
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"upload directory not found: {resolved}")
    return resolved


def _collect_local_files(local_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("upload directory must not contain symlinks")
        if path.is_file():
            relative = path.relative_to(local_dir)
            if any(char in relative.as_posix() for char in "\r\n\t"):
                raise ValueError("upload file paths must not contain control characters")
            files.append(path)
    return files


def _remote_file_digests(
    destination: str,
    port: int,
    identity_file: Path | None,
    remote_path: PurePosixPath,
) -> dict[str, str]:
    output = _run_ssh_script(
        destination,
        port,
        identity_file,
        _remote_file_manifest_script(),
        remote_path.as_posix(),
    )
    prefix = remote_path.as_posix() + "/"
    digests: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        if len(line) < 35 or line[32:34] not in {"  ", " *"}:
            raise RuntimeError("remote file manifest returned an invalid record")
        digest = line[:32].lower()
        absolute_path = line[34:]
        if not re.fullmatch(r"[0-9a-f]{32}", digest) or not absolute_path.startswith(prefix):
            raise RuntimeError("remote file manifest returned an invalid path")
        relative_path = absolute_path[len(prefix) :]
        if (
            not relative_path
            or any(char in relative_path for char in "\r\n\t")
            or any(part in {"", ".", ".."} for part in PurePosixPath(relative_path).parts)
        ):
            raise RuntimeError("remote file manifest returned an invalid path")
        digests[relative_path] = digest
    return digests


def scan_directory_sftp(
    local_dir: Path,
    *,
    host: str,
    port: int,
    remote_dir: str,
    user: str = "",
    identity_file: Path | None = None,
) -> DirectoryUploadPlan:
    """Scan all local files and compare them with the remote dataset."""
    local_root = _validate_local_directory(local_dir)
    if not 1 <= int(port) <= 65535:
        raise ValueError("SFTP port must be in [1, 65535]")
    remote_path = _remote_dataset_path(remote_dir)
    destination = _destination(host, user)
    if identity_file is not None:
        identity = Path(identity_file).expanduser().resolve()
        if not identity.is_file():
            raise FileNotFoundError("SFTP identity file not found")
        identity_file = identity

    local_files = _collect_local_files(local_root)
    remote_exists = _remote_target_exists(destination, port, identity_file, remote_path)
    remote_digests = (
        _remote_file_digests(destination, port, identity_file, remote_path) if remote_exists else {}
    )
    planned_files: list[UploadFilePlan] = []
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
    return DirectoryUploadPlan(
        local_dir=str(local_root),
        remote_dir=remote_path.as_posix(),
        destination=destination,
        remote_exists=remote_exists,
        files=tuple(planned_files),
        remote_only_files=tuple(
            RemoteFilePlan(relative_path, remote_digests[relative_path])
            for relative_path in sorted(
                remote_digests.keys() - {item.relative_path for item in planned_files}
            )
        ),
    )


def _validate_upload_plan(
    plan: DirectoryUploadPlan,
    local_root: Path,
    remote_path: PurePosixPath,
    destination: str,
    port: int,
    identity_file: Path | None,
) -> list[Path]:
    if (
        Path(plan.local_dir) != local_root
        or plan.remote_dir != remote_path.as_posix()
        or plan.destination != destination
    ):
        raise ValueError("SFTP upload plan does not match the requested upload")

    local_files = _collect_local_files(local_root)
    local_by_relative = {path.relative_to(local_root).as_posix(): path for path in local_files}
    if set(local_by_relative) != {item.relative_path for item in plan.files}:
        raise ValueError("local upload directory changed after scan; scan again")
    for item in plan.files:
        path = local_by_relative[item.relative_path]
        if path.stat().st_size != item.size or file_digest(path) != item.digest:
            raise ValueError("local upload directory changed after scan; scan again")

    remote_exists = _remote_target_exists(destination, port, identity_file, remote_path)
    remote_digests = (
        _remote_file_digests(destination, port, identity_file, remote_path) if remote_exists else {}
    )
    if remote_exists != plan.remote_exists:
        raise ValueError("remote upload directory changed after scan; scan again")
    for item in plan.files:
        remote_digest = remote_digests.get(item.relative_path)
        action = (
            "new"
            if remote_digest is None
            else "same"
            if remote_digest == item.digest
            else "changed"
        )
        if action != item.action:
            raise ValueError("remote upload directory changed after scan; scan again")
    planned_remote_only = {item.relative_path: item.digest for item in plan.remote_only_files}
    current_remote_only = {
        relative_path: digest
        for relative_path, digest in remote_digests.items()
        if relative_path not in local_by_relative
    }
    if current_remote_only != planned_remote_only:
        raise ValueError("remote upload directory changed after scan; scan again")
    return [local_by_relative[item.relative_path] for item in plan.files_to_upload]


def _upload_files_sftp(
    sftp: str,
    sftp_args: list[str],
    local_dir: Path,
    remote_path: PurePosixPath,
    files: list[Path],
    progress_callback: Callable[[UploadProgress], None] | None,
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
            progress_callback(UploadProgress(0, len(files), 0, bytes_total, ""))
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
                        UploadProgress(
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
                    UploadProgress(
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
    plan: DirectoryUploadPlan | None = None,
    progress_callback: Callable[[UploadProgress], None] | None = None,
) -> UploadResult:
    """Upload only files selected by a current directory scan."""
    local_dir = _validate_local_directory(local_dir)
    if not 1 <= int(port) <= 65535:
        raise ValueError("SFTP port must be in [1, 65535]")
    remote_path = _remote_dataset_path(remote_dir)
    destination = _destination(host, user)
    if identity_file is not None:
        identity = Path(identity_file).expanduser().resolve()
        if not identity.is_file():
            raise FileNotFoundError("SFTP identity file not found")
        identity_file = identity

    if plan is None:
        plan = scan_directory_sftp(
            local_dir,
            host=host,
            port=port,
            remote_dir=remote_dir,
            user=user,
            identity_file=identity_file,
        )
    files = _validate_upload_plan(
        plan,
        local_dir,
        remote_path,
        destination,
        port,
        identity_file,
    )
    if not files and plan.files_to_delete == 0:
        if progress_callback is not None:
            progress_callback(UploadProgress(0, 0, 0, 0, ""))
        return UploadResult(
            local_dir=str(local_dir),
            remote_dir=remote_path.as_posix(),
            destination=destination,
            files=0,
            bytes=0,
            skipped=True,
            files_deleted=0,
        )

    sftp = shutil.which("sftp")
    if sftp is None:
        raise RuntimeError("OpenSSH sftp executable is required")
    parent = remote_path.parent
    token = uuid.uuid4().hex
    staging_path = parent / f".{remote_path.name}.staging-{token}"

    # Upload every file into a same-parent staging directory.
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
                *(["-i", str(identity_file)] if identity_file is not None else []),
                destination,
            ],
            local_dir,
            staging_path,
            files,
            progress_callback,
        )
        actual_remote_dir, skipped = _publish_remote_dataset(
            destination,
            port,
            identity_file,
            remote_path,
            staging_path,
            plan.remote_only_files,
        )
    except Exception as error:
        _cleanup_remote_staging(destination, port, identity_file, staging_path)
        raise RuntimeError("SFTP upload failed; the new remote target was not published") from error
    return UploadResult(
        local_dir=str(local_dir),
        remote_dir=actual_remote_dir,
        destination=destination,
        files=0 if skipped else len(files),
        bytes=0 if skipped else sum(path.stat().st_size for path in files),
        skipped=skipped,
        files_deleted=len(plan.remote_only_files),
    )


__all__ = [
    "UploadProgress",
    "UploadResult",
    "DirectoryUploadPlan",
    "RemoteFilePlan",
    "scan_directory_sftp",
    "upload_directory_sftp",
]
