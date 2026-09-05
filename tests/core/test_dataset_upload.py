from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.utils import dataset_upload
from core.utils.loopback_upload import scan_directory_loopback

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("linked_root", ["source", "target"])
def test_loopback_rejects_symlink_roots(tmp_path, linked_root):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "episode.txt").write_text("data")
    link = tmp_path / "link"
    link.symlink_to(source if linked_root == "source" else target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinks"):
        scan_directory_loopback(
            link if linked_root == "source" else source,
            remote_dir=str(link if linked_root == "target" else target),
        )
    assert list(target.iterdir()) == []


def test_upload_receipt_tracks_current_accepted_source_episodes(tmp_path: Path) -> None:
    raw = tmp_path / "collection" / "cup_set" / "raw"
    accepted = raw.parent / "export" / "lerobot_v21" / "accepted"
    marker_path = accepted / "meta" / "quality_split.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps(
            {
                "subset": "accepted",
                "source_dir": str(raw),
                "source_episode_indices": [0, 2, 4],
                "dataset_format": "lerobot_v21",
            }
        )
    )

    receipt = dataset_upload.record_dataset_upload_receipt(
        accepted,
        destination="robot@host",
        remote_dir="/datasets/cup_set",
    )

    assert receipt == accepted.parent / "upload_receipt.json"
    assert dataset_upload.uploaded_source_episode_indices(raw) == {0, 2, 4}
    marker_path.write_text(marker_path.read_text().replace("[0, 2, 4]", "[0, 2]"))
    assert dataset_upload.uploaded_source_episode_indices(raw) == set()


def test_loopback_upload_scans_syncs_and_then_skips_matching_files(tmp_path: Path) -> None:
    local_dir = tmp_path / "accepted"
    (local_dir / "meta").mkdir(parents=True)
    (local_dir / "meta" / "info.json").write_text("one")
    (local_dir / "data.bin").write_bytes(b"12345")
    remote_root = tmp_path / "remote"
    specs = dataset_upload.resolve_dataset_uploads(
        {"loopback": {"remote_dir": str(remote_root)}},
        "cup_set",
    )
    progress = []

    assert [spec.backend for spec in specs] == ["loopback"]
    assert specs[0].target == str(remote_root / "cup_set")
    first_plan = dataset_upload.scan_dataset_directory(local_dir, specs)
    first_result = dataset_upload.upload_dataset_directory(
        local_dir,
        specs,
        plan=first_plan,
        progress_callback=progress.append,
    )

    target = remote_root / "cup_set"
    assert first_plan.new_files == 2
    assert first_plan.changed_files == 0
    assert first_plan.files_skipped == 0
    assert first_result.files == 2
    assert first_result.skipped is False
    assert (target / "meta" / "info.json").read_text() == "one"
    assert (target / "data.bin").read_bytes() == b"12345"
    assert progress[-1].files_completed == 2

    same_plan = dataset_upload.scan_dataset_directory(local_dir, specs)
    same_result = dataset_upload.upload_dataset_directory(
        local_dir,
        specs,
        plan=same_plan,
    )

    assert same_plan.files_to_upload == 0
    assert same_plan.files_skipped == 2
    assert same_result.skipped is True

    (local_dir / "meta" / "info.json").write_text("two")
    (local_dir / "new.txt").write_text("new")
    (target / "stale.txt").write_text("stale")
    changed_plan = dataset_upload.scan_dataset_directory(local_dir, specs)
    changed_result = dataset_upload.upload_dataset_directory(
        local_dir,
        specs,
        plan=changed_plan,
    )

    assert changed_plan.new_files == 1
    assert changed_plan.changed_files == 1
    assert changed_plan.files_skipped == 1
    assert changed_plan.files_to_delete == 1
    assert changed_result.files == 2
    assert changed_result.files_deleted == 1
    assert (target / "meta" / "info.json").read_text() == "two"
    assert (target / "new.txt").read_text() == "new"
    assert not (target / "stale.txt").exists()


def test_loopback_upload_rejects_source_changes_after_scan(tmp_path: Path) -> None:
    local_dir = tmp_path / "accepted"
    local_dir.mkdir()
    source_file = local_dir / "data.bin"
    source_file.write_bytes(b"before")
    specs = dataset_upload.resolve_dataset_uploads(
        {"loopback": {"remote_dir": str(tmp_path / "remote")}},
        "cup_set",
    )
    plan = dataset_upload.scan_dataset_directory(local_dir, specs)
    source_file.write_bytes(b"after")

    with pytest.raises(RuntimeError, match="changed after scan"):
        dataset_upload.upload_dataset_directory(local_dir, specs, plan=plan)
