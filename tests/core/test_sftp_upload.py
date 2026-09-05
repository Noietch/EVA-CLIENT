from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import TypedDict

import pytest

from core.utils import sftp_upload
from core.utils.upload_plan import UploadProgress

pytestmark = pytest.mark.integration


class _SshCall(TypedDict):
    args: list[str]
    input: str


class _FakeSshRunner:
    def __init__(
        self,
        *,
        target_exists: bool = False,
        published_target: str = "/datasets/accepted",
        publish_state: str = "published",
        publish_error: int = 0,
        remote_root: str = "/datasets/accepted",
        remote_files: dict[str, str] | None = None,
    ) -> None:
        self.target_exists = target_exists
        self.publish_error = publish_error
        self.published_target = published_target
        self.publish_state = publish_state
        self.remote_root = remote_root
        self.remote_files = remote_files or {}
        self.calls: list[_SshCall] = []

    def __call__(self, args, *, check, capture_output, input, text):
        assert check is True
        assert capture_output is True
        assert text is True
        call = {"args": list(args), "input": input}
        self.calls.append(call)
        if 'find "$root" -type f' in str(input):
            stdout = "".join(
                f"{digest}  {self.remote_root}/{relative_path}\n"
                for relative_path, digest in self.remote_files.items()
            )
        elif 'target="$1"' in str(input):
            stdout = "exists\n" if self.target_exists else "missing\n"
        elif 'preferred="$1"' in str(input):
            if self.publish_error:
                raise subprocess.CalledProcessError(self.publish_error, args, stderr="publish")
            stdout = f"{self.publish_state}\n{self.published_target}\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


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

    assert len(fake_ssh.calls) == 4
    cleanup_call = fake_ssh.calls[3]
    assert cleanup_call["args"][-3:] == ["sh", "-s", "--"]
    assert str(uploaded[0]) not in cleanup_call["args"]
    assert f"set -- {sftp_upload.shlex.quote(str(uploaded[0]))}" in cleanup_call["input"]
    assert 'staging="$1"' in str(cleanup_call["input"])


def test_remote_publish_script_merges_changed_files_into_existing_target(
    tmp_path: Path,
) -> None:
    preferred = (tmp_path / "accepted").resolve()
    staging = tmp_path / ".accepted.staging-token"
    preferred.mkdir()
    staging.mkdir()
    (preferred / "old.txt").write_text("old")
    (staging / "new.txt").write_text("new")

    result = subprocess.run(
        ["sh", "-s", "--"],
        check=True,
        capture_output=True,
        input=sftp_upload._ssh_payload(
            sftp_upload._remote_publish_script(),
            str(preferred),
            str(staging),
        ),
        text=True,
    )

    assert result.stdout.splitlines() == ["merged", str(preferred)]
    assert (preferred / "old.txt").read_text() == "old"
    assert (preferred / "new.txt").read_text() == "new"
    assert not staging.exists()


def test_confirm_rejects_local_content_changed_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    local.mkdir()
    file_path = local / "info.json"
    file_path.write_text("before")
    fake_ssh = _FakeSshRunner(target_exists=False)
    monkeypatch.setattr(sftp_upload.subprocess, "run", fake_ssh)

    plan = sftp_upload.scan_directory_sftp(
        local,
        host="server",
        port=22,
        remote_dir="/datasets/accepted",
    )
    file_path.write_text("after")

    with pytest.raises(ValueError, match="local upload directory changed"):
        sftp_upload.upload_directory_sftp(
            local,
            host="server",
            port=22,
            remote_dir="/datasets/accepted",
            plan=plan,
        )


def test_confirm_rejects_remote_content_changed_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    local.mkdir()
    file_path = local / "info.json"
    file_path.write_text("local")
    fake_ssh = _FakeSshRunner(
        target_exists=True,
        remote_files={"info.json": sftp_upload.file_digest(file_path)},
    )
    monkeypatch.setattr(sftp_upload.subprocess, "run", fake_ssh)

    plan = sftp_upload.scan_directory_sftp(
        local,
        host="server",
        port=22,
        remote_dir="/datasets/accepted",
    )
    fake_ssh.remote_files["info.json"] = "0" * 32

    with pytest.raises(ValueError, match="remote upload directory changed"):
        sftp_upload.upload_directory_sftp(
            local,
            host="server",
            port=22,
            remote_dir="/datasets/accepted",
            plan=plan,
        )


def test_confirm_rejects_remote_only_content_changed_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    local.mkdir()
    fake_ssh = _FakeSshRunner(
        target_exists=True,
        remote_files={"stale.json": "1" * 32},
    )
    monkeypatch.setattr(sftp_upload.subprocess, "run", fake_ssh)

    plan = sftp_upload.scan_directory_sftp(
        local,
        host="server",
        port=22,
        remote_dir="/datasets/accepted",
    )
    fake_ssh.remote_files["stale.json"] = "2" * 32

    with pytest.raises(ValueError, match="remote upload directory changed"):
        sftp_upload.upload_directory_sftp(
            local,
            host="server",
            port=22,
            remote_dir="/datasets/accepted",
            plan=plan,
        )


def test_local_round_trip_upload_all_skip_same_then_only_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "accepted"
    (local / "meta").mkdir(parents=True)
    (local / "meta" / "info.json").write_text("info")
    (local / "episodes.jsonl").write_text("episode\n")
    remote_dir = (tmp_path / "remote" / "cup_set").as_posix()

    def run_local(_destination, _port, _identity, script, *args):
        result = subprocess.run(
            ["sh", "-s", "--"],
            check=True,
            capture_output=True,
            input=sftp_upload._ssh_payload(script, *args),
            text=True,
        )
        return result.stdout

    def copy_files(_sftp, _args, root, remote_path, files, progress_callback):
        target = Path(remote_path.as_posix())
        target.mkdir(parents=True, exist_ok=True)
        bytes_total = sum(path.stat().st_size for path in files)
        bytes_completed = 0
        if progress_callback is not None:
            progress_callback(UploadProgress(0, len(files), 0, bytes_total, ""))
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(root)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            bytes_completed += path.stat().st_size
            if progress_callback is not None:
                progress_callback(
                    sftp_upload.UploadProgress(
                        index,
                        len(files),
                        bytes_completed,
                        bytes_total,
                        relative.as_posix(),
                    )
                )

    monkeypatch.setattr(sftp_upload, "_run_ssh_script", run_local)
    monkeypatch.setattr(sftp_upload, "_upload_files_sftp", copy_files)
    monkeypatch.setattr(
        sftp_upload.shutil,
        "which",
        lambda name: "/usr/bin/true" if name == "sftp" else "/usr/bin/ssh",
    )

    first_plan = sftp_upload.scan_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir
    )
    first = sftp_upload.upload_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir, plan=first_plan
    )
    second_plan = sftp_upload.scan_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir
    )
    (local / "meta" / "tasks.jsonl").write_text("task\n")
    third_plan = sftp_upload.scan_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir
    )
    third = sftp_upload.upload_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir, plan=third_plan
    )
    stale_remote = Path(remote_dir) / "data" / "stale.parquet"
    stale_remote.parent.mkdir(parents=True)
    stale_remote.write_text("stale")
    prune_plan = sftp_upload.scan_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir
    )
    pruned = sftp_upload.upload_directory_sftp(
        local, host="local", port=22, remote_dir=remote_dir, plan=prune_plan
    )

    assert first.files == 2
    assert first_plan.new_files == 2
    assert second_plan.files_to_upload == ()
    assert second_plan.files_skipped == 2
    assert third.files == 1
    assert third_plan.new_files == 1
    assert third_plan.files_skipped == 2
    assert prune_plan.files_to_delete == 1
    assert pruned.files == 0
    assert pruned.files_deleted == 1
    assert not stale_remote.exists()
    assert (Path(remote_dir) / "meta" / "tasks.jsonl").read_text() == "task\n"
    assert not (Path(remote_dir) / "accepted").exists()
