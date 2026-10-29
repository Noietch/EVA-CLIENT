"""Task-set reference images transfer without network access."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.datasets import hf_task_sets

pytestmark = pytest.mark.unit


def test_publish_task_set_includes_configured_reference_images(tmp_path, monkeypatch):
    task_set = tmp_path / "task_sets" / "set_a"
    (task_set / "ref_imgs").mkdir(parents=True)
    for name in ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv"):
        (task_set / name).write_text("x\n", encoding="utf-8")
    (task_set / "layout.yaml").write_text("ref_imgs: ref_imgs\n", encoding="utf-8")
    (task_set / "ref_imgs/reference.png").write_bytes(b"reference")
    (task_set / "ref_imgs/notes.txt").write_text("metadata\n", encoding="utf-8")
    uploaded = {}

    class FakeApi:
        def __init__(self, token):
            pass

        def upload_folder(self, **kwargs):
            uploaded.update(kwargs)
            return SimpleNamespace(oid="revision")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))

    hf_task_sets.publish_task_set(
        tmp_path, task_set, {"huggingface": {"repo_id": "team/data"}}
    )

    assert uploaded["allow_patterns"] == [
        *hf_task_sets.TASK_FILES,
        "ref_imgs/reference.png",
    ]


def test_fetch_task_set_copies_nested_references_and_keeps_existing_files(tmp_path, monkeypatch):
    requested = []

    def fake_snapshot_download(**kwargs):
        requested.append(kwargs["allow_patterns"])
        source = Path(kwargs["local_dir"]) / "task_sets/set_a"
        (source / "ref_imgs/nested").mkdir(parents=True)
        (source / "tasks.csv").write_text("task\n", encoding="utf-8")
        (source / "layout.yaml").write_text("ref_imgs: ref_imgs\n", encoding="utf-8")
        (source / "ref_imgs/nested/reference.webp").write_bytes(b"reference")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    destination = tmp_path / "local"
    existing = destination / "set_a/keep.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep\n", encoding="utf-8")

    hf_task_sets.fetch_task_set(
        tmp_path,
        "set_a",
        destination,
        {"huggingface": {"repo_id": "team/data"}},
    )

    assert requested == [["task_sets/set_a/*"]]
    assert existing.read_text(encoding="utf-8") == "keep\n"
    assert (destination / "set_a/ref_imgs/nested/reference.webp").read_bytes() == b"reference"
