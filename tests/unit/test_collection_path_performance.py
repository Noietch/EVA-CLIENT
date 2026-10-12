"""Mounted paths and recent history retain their semantics after caching."""

import json

import pytest
import yaml

from core.cfg import ConfigDict
from core.recorder.episode import EpisodeLogger
from core.utils.collection_paths import dataset_location, task_set_root

pytestmark = pytest.mark.unit


def test_dataset_location_keeps_invalid_yaml_visible(tmp_path):
    plan = tmp_path / "dataset"
    plan.mkdir()
    (plan / "tasks.csv").touch()
    (plan / "info.yaml").write_text("collection_dir: [unterminated\n")
    config = ConfigDict(collection=dict(task_set_dir=str(plan)))
    with pytest.raises(yaml.YAMLError):
        dataset_location(config, "dataset")


def test_dataset_location_refreshes_changed_and_replaced_info(tmp_path):
    plan = tmp_path / "task_sets" / "dataset"
    plan.mkdir(parents=True)
    (plan / "tasks.csv").touch()
    info = plan / "info.yaml"
    info.write_text("collection_dir: datasets/first/dataset\n")
    config = ConfigDict(collection=dict(task_set_dir=str(plan)))

    assert dataset_location(config, "dataset")[1] == "datasets/first/dataset"
    info.write_text("collection_dir: datasets/second/dataset\n")
    assert dataset_location(config, "dataset")[1] == "datasets/second/dataset"
    replacement = plan / "replacement.yaml"
    replacement.write_text("collection_dir: datasets/third/dataset\n")
    replacement.replace(info)
    assert dataset_location(config, "dataset")[1] == "datasets/third/dataset"


def test_task_set_root_preserves_mount_priority(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second" / "dataset"
    earlier = first / "larybenchv2.v1.2" / "dataset"
    unrelated = tmp_path / "other-dataset"
    for root in (earlier, second, unrelated):
        root.mkdir(parents=True)
        (root / "tasks.csv").touch()
    config = ConfigDict(collection=dict(task_set_dir=[str(unrelated), str(first), str(second)]))

    assert task_set_root(config, "dataset") == earlier


def test_recent_history_preserves_fallback_indices_and_qc_updates(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    episodes = meta / "episodes.jsonl"
    rows = [{"length": index, "qc_verdict": "unreviewed"} for index in range(205)]
    episodes.write_text("invalid\n" + "\n".join(json.dumps(row) for row in rows))
    recorder = EpisodeLogger.__new__(EpisodeLogger)
    recorder._log_dir = tmp_path

    history = recorder._load_collection_history()
    assert len(history) == 200
    assert history[0]["episode_index"] == 5
    assert history[-1]["episode_index"] == 204
    rows[-1]["qc_verdict"] = "pass"
    episodes.write_text("\n".join(json.dumps(row) for row in rows))
    assert recorder._load_collection_history()[-1]["qc_verdict"] == "pass"
