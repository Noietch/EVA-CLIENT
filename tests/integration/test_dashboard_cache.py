"""Dashboard metadata reuse must preserve live QC and episode updates."""

import json
from unittest.mock import patch

import pytest

from core.app.console import dashboard

pytestmark = pytest.mark.integration


def test_qc_changes_reuse_sources_and_new_episodes_invalidate_them(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    episodes_path = meta / "episodes.jsonl"
    episodes_path.write_text(json.dumps({"episode_index": 0, "length": 30}) + "\n")
    (meta / "tasks.jsonl").write_text(json.dumps({"task": "task", "required_episodes": 10}))
    (meta / "info.json").write_text(json.dumps({"fps": 30}))
    qc_path = meta / "qc.jsonl"
    _, _, records = dashboard._load_dataset_records(tmp_path, "collection")
    assert records[0]["valid"] is True

    with (
        patch.object(dashboard, "_read_jsonl", wraps=dashboard._read_jsonl) as read_jsonl,
        patch.object(dashboard, "_read_json", wraps=dashboard._read_json) as read_json,
    ):
        for verdict, valid in [("fail", False), ("pass", True)]:
            # Match the QC writer's atomic replacement, even on coarse-mtime filesystems.
            replacement = meta / "qc.next"
            replacement.write_text(json.dumps({"episode_index": 0, "qc_verdict": verdict}) + "\n")
            replacement.replace(qc_path)
            _, _, records = dashboard._load_dataset_records(tmp_path, "collection")
            assert records[0]["valid"] is valid
        assert [call.args[0] for call in read_jsonl.call_args_list] == [qc_path, qc_path]
        read_json.assert_not_called()

        with episodes_path.open("a") as stream:
            stream.write(json.dumps({"episode_index": 1, "length": 60}) + "\n")
        _, _, records = dashboard._load_dataset_records(tmp_path, "collection")
        assert [row["episode_index"] for row in records] == [0, 1]
        assert episodes_path in [call.args[0] for call in read_jsonl.call_args_list]

    key = (str(tmp_path.resolve()), "collection")
    assert sum(cache_key == key for cache_key in dashboard._DATASET_CACHE) == 1
