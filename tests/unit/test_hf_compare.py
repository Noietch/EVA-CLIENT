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
    # Nothing recorded on either side is a match, not a separate state.
    assert compare_qc({}, {})["state"] == "same"


@pytest.mark.parametrize(
    ("verdict", "auto_verdict", "replay_required", "replay_verified", "expected"),
    [
        ("unreviewed", "fail", False, None, "fail"),
        ("pass", "fail", False, None, "accept"),
        ("pass", "", True, None, "unreviewed"),
        ("pass", "", True, False, "fail"),
        ("pass", "", True, True, "accept"),
    ],
)
def test_qc_summary_uses_machine_verdict_and_replay_gate(
    verdict, auto_verdict, replay_required, replay_verified, expected
):
    episodes = [
        {
            "episode_index": 0,
            "quality": "green",
            "replay_required": replay_required,
            "replay_verified": replay_verified,
        }
    ]
    qc = [{"episode_index": 0, "qc_verdict": verdict, "qc_auto_verdict": auto_verdict}]
    counts = {"accept": 0, "fail": 0, "unreviewed": 0}
    counts[expected] = 1
    assert qc_summary(episodes, qc) == counts


@pytest.mark.parametrize("metadata", [{"sim_metadata": {}}, {"render_report": "replay.json"}])
def test_raw_sim_metadata_keeps_human_pass_pending_without_a_validated_report(metadata):
    episodes = [{"episode_index": 0, "quality": "green", **metadata}]
    qc = [{"episode_index": 0, "qc_verdict": "pass"}]
    assert qc_summary(episodes, qc) == {"accept": 0, "fail": 0, "unreviewed": 1}
