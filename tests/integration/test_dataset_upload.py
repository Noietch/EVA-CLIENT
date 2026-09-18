from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.utils import dataset_upload
from core.utils.loopback_upload import scan_directory_loopback

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("linked_root", ["source"])
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


def test_upload_receipt_tracks_the_delivered_export(tmp_path: Path) -> None:
    source = tmp_path / "cup_set"
    exported = source.with_name(f"{source.name}_lerobot_v3")
    for directory in (source, exported):
        (directory / "meta").mkdir(parents=True)
        (directory / "meta" / "episodes.jsonl").write_text(
            "".join(json.dumps({"episode_index": index}) + "\n" for index in (0, 2, 4))
        )

    receipt = dataset_upload.record_dataset_upload_receipt(
        exported,
        source_dir=source,
        destination="robot@host",
        remote_dir="/datasets/cup_set",
    )

    assert receipt == exported.with_name(f"{exported.name}_upload_receipt.json")
    assert dataset_upload.uploaded_source_episode_indices(source) == {0, 2, 4}
    # Collecting again makes the export stale, so its receipt stops counting.
    (exported / "data").mkdir()
    (exported / "data" / "episode_000000.parquet").write_bytes(b"new episode")
    assert dataset_upload.uploaded_source_episode_indices(source) == set()
