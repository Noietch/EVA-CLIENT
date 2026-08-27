from __future__ import annotations

from pathlib import Path, PurePosixPath

from core.utils import sftp_upload


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


def test_upload_directory_uses_configured_sftp_and_reports_progress(
    tmp_path: Path, monkeypatch
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
    executable_lookups = []
    progress = []

    def find_executable(name):
        executable_lookups.append(name)
        return str(fake_sftp)

    monkeypatch.setattr(sftp_upload.shutil, "which", find_executable)

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
    assert executable_lookups == ["sftp"]
    assert progress[0].bytes_completed == 0
    assert any(item.bytes_completed == 1 for item in progress)
    assert any(item.bytes_completed == 3 for item in progress)
    assert progress[-1].files_completed == 1
    assert progress[-1].bytes_completed == 4
    assert progress[-1].bytes_total == 4


def test_upload_directory_reports_failed_file_upload(tmp_path: Path, monkeypatch) -> None:
    local = tmp_path / "accepted"
    local.mkdir()
    (local / "info.json").write_text("info")
    fake_sftp = tmp_path / "sftp"
    fake_sftp.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    command = line.strip()\n"
        "    if command == 'pwd':\n"
        "        print('Remote working directory: /', flush=True)\n"
        "    elif command.startswith('put '):\n"
        "        print('remote put: Failure', flush=True)\n"
        "        raise SystemExit(1)\n"
    )
    fake_sftp.chmod(0o755)
    monkeypatch.setattr(sftp_upload.shutil, "which", lambda _name: str(fake_sftp))

    try:
        sftp_upload.upload_directory_sftp(
            local,
            host="server",
            port=22,
            remote_dir="/datasets/already-there",
        )
    except RuntimeError as error:
        assert "SFTP command failed" in str(error)
    else:
        raise AssertionError("failed remote file upload was accepted")
