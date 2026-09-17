import hashlib
from types import SimpleNamespace

import pytest

from tools.datasets.hf_compare import compare_files, compare_qc, qc_entries, qc_summary

pytestmark = pytest.mark.unit


def test_hash_comparison_detects_missing_extra_and_same_size_changes(tmp_path):
    same = tmp_path / "same"
    same.write_bytes(b"abc")
    changed = tmp_path / "changed"
    changed.write_bytes(b"abd")
    git = hashlib.sha1(b"blob 3\0abc").hexdigest()
    sha = hashlib.sha256(b"abc").hexdigest()
    remote = {
        "same": SimpleNamespace(size=3, blob_id=git, lfs=None),
        "changed": SimpleNamespace(size=3, blob_id="", lfs={"sha256": sha}),
        "missing": SimpleNamespace(size=3, blob_id=git, lfs=None),
    }
    assert compare_files({"same": same, "changed": changed, "extra": same}, remote) == {
        "state": "different",
        "missing_local": ["missing"],
        "local_only": ["extra"],
        "changed": ["changed"],
        "unknown": [],
    }


def test_qc_comparison_overlays_reviews_and_compares_episode_identity():
    episodes = [
        {"episode_index": 3, "qc_verdict": "fail", "quality": "green"},
        {"episode_index": 4, "quality": "red"},
    ]
    qc = [{"episode_index": 3, "qc_verdict": "pass"}]
    assert qc_summary(episodes, qc) == {"accept": 1, "fail": 1, "unreviewed": 0}
    local = qc_entries(episodes, qc)
    assert compare_qc(local, qc_entries(list(reversed(episodes)), qc))["state"] == "same"
    assert compare_qc(local, qc_entries(episodes, []))["changed"] == ["3"]
    assert compare_qc({}, {})["state"] == "absent"
