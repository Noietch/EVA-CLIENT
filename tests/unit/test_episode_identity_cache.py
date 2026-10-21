"""Episode lookup and QC reuse identities without retaining simulation metadata."""

import json
import os
from pathlib import Path

import pytest

from core.utils.lerobot import LeRobotDatasetIO

pytestmark = pytest.mark.unit


def test_repeated_lookup_and_qc_do_not_reread_episode_metadata(tmp_path, monkeypatch):
    meta = tmp_path / "meta"
    meta.mkdir()
    path = meta / "episodes.jsonl"
    original = json.dumps({
        "episode_index": 7,
        "file_stem": "capture-seven",
        "sim_metadata": {"objects": [{"pose": [0] * 7}] * 1000},
    }) + "\n"
    path.write_text(original)
    (meta / "info.json").write_text(json.dumps({
        "data_path": "data/{episode_file}.parquet",
    }))
    opens = 0
    original_open = Path.open

    def track_open(self, *args, **kwargs):
        nonlocal opens
        if self == path:
            opens += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    for verdict in ("pass", "fail", "unreviewed"):
        io = LeRobotDatasetIO(tmp_path)
        assert io.episode_parquet(7) == tmp_path / "data/capture-seven.parquet"
        assert io.mark_qc(7, verdict)
        assert not io.mark_qc(99, verdict)
    assert opens == 1
    assert path.read_text() == original
    assert json.loads((meta / "qc.jsonl").read_text())["qc_verdict"] == "unreviewed"


def test_identity_cache_follows_creation_append_retake_and_removal(tmp_path):
    io = LeRobotDatasetIO(tmp_path)
    assert io.episode_file_stem(0) == "episode_000000"
    assert not io.mark_qc(0, "pass")
    path = tmp_path / "meta/episodes.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text('{"episode_index": 0, "file_stem": "take-a"}\n')
    assert io.episode_file_stem(0) == "take-a"
    with path.open("a") as stream:
        stream.write('{"episode_index": 1}\n')
    assert io.mark_qc(1, "pass")
    assert io.episode_file_stem(1) == "episode_000001"

    # A retake can replace a same-sized file while preserving its timestamp.
    stat = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_text(path.read_text().replace("take-a", "take-b"))
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    assert io.episode_file_stem(0) == "take-b"
    assert io.mark_qc(0, "fail")

    path.write_text('{"episode_index": 1}\n')
    assert io.episode_file_stem(0) == "episode_000000"
    assert not io.mark_qc(0, "pass")
    path.unlink()
    assert not io.mark_qc(1, "fail")


def test_identity_cache_is_scoped_to_dataset_and_keeps_last_duplicate(tmp_path):
    for dataset, stem in (("a", "first"), ("b", "second")):
        meta = tmp_path / dataset / "meta"
        meta.mkdir(parents=True)
        (meta / "episodes.jsonl").write_text(
            '{"episode_index": 0, "file_stem": "old"}\n\n'
            + json.dumps({"episode_index": 0, "file_stem": stem}) + "\n"
        )
    assert LeRobotDatasetIO(tmp_path / "a").episode_file_stem(0) == "first"
    assert LeRobotDatasetIO(tmp_path / "b").episode_file_stem(0) == "second"
