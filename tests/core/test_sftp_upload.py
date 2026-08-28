from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from typing import TypedDict

import pytest

from core.utils import sftp_upload


class _SshCall(TypedDict):
    args: list[str]
    input: str


class _FakeSshRunner:
    def __init__(
        self,
        *,
        published_target: str = "/datasets/accepted",
        publish_error: int = 0,
    ) -> None:
        self.publish_error = publish_error
        self.published_target = published_target
        self.calls: list[_SshCall] = []

    def __call__(self, args, *, check, capture_output, input, text):
        assert check is True
        assert capture_output is True
        assert text is True
        call = {"args": list(args), "input": input}
        self.calls.append(call)
        if "copy_base" in str(input) and self.publish_error:
            raise subprocess.CalledProcessError(self.publish_error, args, stderr="publish")
        return subprocess.CompletedProcess(args, 0, stdout=self.published_target + "\n", stderr="")


def test_remote_mkdir_paths_include_all_parent_directories() -> None:
    paths = sftp_upload._remote_mkdir_paths(
        PurePosixPath("/datasets/pick_place"),
        {Path("data/chunk-000"), Path("meta")},
    )

    assert paths == [
        PurePosixPath("/datasets"),
        PurePosixPath("/datasets/pick_place"),
        PurePosixPath("/datasets/pick_place/data"),
        PurePosixPath("/datasets/pick_place/meta"),
        PurePosixPath("/datasets/pick_place/data/chunk-000"),
    ]


def test_upload_directory_uses_copy_target_when_publish_detects_existing_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    local.mkdir()
    (local / "file.txt").write_text("data")
    uploaded: list[PurePosixPath] = []
    remote_dir = "/datasets/accepted ;semi space"
    fake_ssh = _FakeSshRunner(
        published_target="/datasets/accepted ;semi space.copy_20260828T060553123456Z"
    )

    monkeypatch.setattr(
        sftp_upload.shutil,
        "which",
        lambda name: f"/tmp/fake-{name}" if name in {"ssh", "sftp"} else None,
    )
    monkeypatch.setattr(sftp_upload.subprocess, "run", fake_ssh)
    monkeypatch.setattr(sftp_upload, "_copy_timestamp", lambda: "20260828T060553123456Z")
    monkeypatch.setattr(
        sftp_upload,
        "_upload_files_sftp",
        lambda _sftp, _args, _local_dir, remote_path, _files, _progress: uploaded.append(
            remote_path
        ),
    )

    result = sftp_upload.upload_directory_sftp(
        local,
        host="10.0.0.8",
        port=22,
        user="robot",
        remote_dir=remote_dir,
    )

    assert result.remote_dir == "/datasets/accepted ;semi space.copy_20260828T060553123456Z"
    assert result.destination == "robot@10.0.0.8"
    assert len(uploaded) == 1
    assert uploaded[0].parent == PurePosixPath("/datasets")
    assert uploaded[0].name.startswith(".accepted ;semi space.staging-")
    publish_call = fake_ssh.calls[0]
    assert publish_call["args"][-3:] == ["10.0.0.8", "sh", "-s", "--"][-3:]
    assert remote_dir not in publish_call["args"]
    assert str(uploaded[0]) not in publish_call["args"]
    assert "/datasets/accepted ;semi space.copy_20260828T060553123456Z" not in publish_call["args"]
    assert (
        "set -- '/datasets/accepted ;semi space' "
        f"{sftp_upload.shlex.quote(str(uploaded[0]))} "
        "'/datasets/accepted ;semi space.copy_20260828T060553123456Z'"
    ) in publish_call["input"]


@pytest.mark.parametrize(
    "remote_dir", ["/datasets/../escape", "/datasets/./name", "/datasets//name", "/"]
)
def test_upload_directory_rejects_non_canonical_remote_target(remote_dir: str) -> None:
    local_dir = Path.cwd()
    with pytest.raises(ValueError, match="absolute dataset path"):
        sftp_upload.upload_directory_sftp(
            local_dir,
            host="server",
            port=22,
            remote_dir=remote_dir,
        )


def test_upload_directory_rejects_remote_target_with_newline() -> None:
    local_dir = Path.cwd()
    with pytest.raises(ValueError, match="absolute dataset path"):
        sftp_upload.upload_directory_sftp(
            local_dir,
            host="server",
            port=22,
            remote_dir="/datasets/name\nnext",
        )


def test_upload_directory_rejects_root_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    symlink_dir = tmp_path / "accepted"
    symlink_dir.symlink_to(real_dir)

    monkeypatch.setattr(
        sftp_upload.shutil,
        "which",
        lambda name: f"/tmp/fake-{name}" if name in {"ssh", "sftp"} else None,
    )

    with pytest.raises(ValueError, match="root must not be a symlink"):
        sftp_upload.upload_directory_sftp(
            symlink_dir,
            host="server",
            port=22,
            remote_dir="/datasets/accepted",
        )


def test_upload_directory_cleans_up_staging_after_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    local.mkdir()
    (local / "info.json").write_text("info")
    uploaded: list[PurePosixPath] = []
    fake_ssh = _FakeSshRunner(publish_error=46)

    monkeypatch.setattr(
        sftp_upload.shutil,
        "which",
        lambda name: f"/tmp/fake-{name}" if name in {"ssh", "sftp"} else None,
    )
    monkeypatch.setattr(sftp_upload.subprocess, "run", fake_ssh)
    monkeypatch.setattr(
        sftp_upload,
        "_upload_files_sftp",
        lambda _sftp, _args, _local_dir, remote_path, _files, _progress: uploaded.append(
            remote_path
        ),
    )

    with pytest.raises(RuntimeError, match="not published"):
        sftp_upload.upload_directory_sftp(
            local,
            host="server",
            port=22,
            remote_dir="/datasets/accepted",
        )

    assert len(fake_ssh.calls) == 2
    cleanup_call = fake_ssh.calls[1]
    assert cleanup_call["args"][-3:] == ["sh", "-s", "--"]
    assert str(uploaded[0]) not in cleanup_call["args"]
    assert f"set -- {sftp_upload.shlex.quote(str(uploaded[0]))}" in cleanup_call["input"]
    assert 'staging="$1"' in str(cleanup_call["input"])


def test_publish_remote_dataset_rejects_unexpected_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sftp_upload,
        "_run_ssh_script",
        lambda *_args, **_kwargs: "/datasets/accepted-not-copy",
    )
    monkeypatch.setattr(sftp_upload, "_copy_timestamp", lambda: "20260828T060553123456Z")

    with pytest.raises(RuntimeError, match="invalid destination"):
        sftp_upload._publish_remote_dataset(
            "server",
            22,
            None,
            PurePosixPath("/datasets/accepted"),
            PurePosixPath("/datasets/.accepted.staging-token"),
        )


def test_remote_publish_script_chooses_numeric_copy_suffix_without_touching_existing_dirs(
    tmp_path: Path,
) -> None:
    preferred = (tmp_path / "accepted").resolve()
    copy_base = Path(str(preferred) + ".copy_20260828T060553123456Z")
    staging = tmp_path / ".accepted.staging-token"
    preferred.mkdir()
    copy_base.mkdir()
    staging.mkdir()
    (preferred / "old.txt").write_text("old")
    (copy_base / "copy.txt").write_text("copy")
    (staging / "new.txt").write_text("new")

    result = subprocess.run(
        ["sh", "-s", "--"],
        check=True,
        capture_output=True,
        input=sftp_upload._ssh_payload(
            sftp_upload._remote_publish_script(),
            str(preferred),
            str(staging),
            str(copy_base),
        ),
        text=True,
    )

    published = Path(result.stdout.strip())
    assert published == Path(str(copy_base) + "_02")
    assert (preferred / "old.txt").read_text() == "old"
    assert (copy_base / "copy.txt").read_text() == "copy"
    assert (published / "new.txt").read_text() == "new"
    assert not staging.exists()


def test_upload_directory_uses_configured_sftp_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    (local / "meta").mkdir(parents=True)
    (local / "meta" / "info.json").write_bytes(b"info")
    identity = tmp_path / "identity"
    identity.write_text("key")
    fake_sftp = tmp_path / "sftp"
    fake_sftp.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    command = line.strip()\n"
        "    if command == 'pwd':\n"
        "        print('Remote working directory: /', flush=True)\n"
        "    elif command.startswith('put '):\n"
        "        print(' 25% 1B/s 00:01\\r 75% 1B/s 00:00', flush=True)\n"
        "    elif command == 'quit':\n"
        "        break\n"
    )
    fake_sftp.chmod(0o755)
    progress = []
    fake_ssh = _FakeSshRunner(published_target="/datasets/pick_place")

    def find_executable(name: str) -> str | None:
        if name == "sftp":
            return str(fake_sftp)
        if name == "ssh":
            return "/tmp/fake-ssh"
        return None

    monkeypatch.setattr(sftp_upload.shutil, "which", find_executable)
    monkeypatch.setattr(sftp_upload.subprocess, "run", fake_ssh)
    monkeypatch.setattr(sftp_upload, "_copy_timestamp", lambda: "20260828T060553123456Z")

    result = sftp_upload.upload_directory_sftp(
        local,
        host="10.0.0.8",
        port=8000,
        user="robot",
        identity_file=identity,
        remote_dir="/datasets/pick_place",
        progress_callback=progress.append,
    )

    assert result.destination == "robot@10.0.0.8"
    assert result.files == 1
    assert result.bytes == 4
    assert progress[0].bytes_completed == 0
    assert any(item.bytes_completed == 1 for item in progress)
    assert any(item.bytes_completed == 3 for item in progress)
    assert progress[-1].files_completed == 1
    assert progress[-1].bytes_completed == 4
    assert progress[-1].bytes_total == 4
