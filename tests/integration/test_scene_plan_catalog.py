"""Evaluation scene catalogs preserve versioned names and explicit missing roots."""

import os
from pathlib import Path

import pytest

from core.config import ConfigDict
from core.utils.scene_plan import evaluation_task_sets

pytestmark = pytest.mark.integration


def _task_set(root):
    root.mkdir(parents=True)
    (root / "tasks.csv").write_text("task_id,prompt_en\nTASK,pick up the cup\n")
    return root


def _config(root, *, evaluation="", collection=""):
    return ConfigDict(
        eval=ConfigDict(task_set_dir=evaluation),
        collection=ConfigDict(
            task_set_dir=collection,
            storage=ConfigDict(data_root=str(root)),
        ),
    )


@pytest.mark.parametrize("as_list", [False, True], ids=["scalar", "list"])
def test_explicit_evaluation_roots_take_priority(tmp_path, as_list):
    explicit = _task_set(tmp_path / "explicit/eval")
    _task_set(tmp_path / "task_sets/automatic")
    mounted = _task_set(tmp_path / "mounted/train")
    configured = [str(explicit)] if as_list else str(explicit)
    catalog = evaluation_task_sets(
        _config(tmp_path, evaluation=configured, collection=str(mounted))
    )

    assert list(catalog) == ["eval"]
    assert catalog == {"eval": explicit}


def test_catalog_distinguishes_versions_and_bounds_discovery(tmp_path):
    first = _task_set(tmp_path / "task_sets/larybenchv2.v1.1/eval")
    second = _task_set(tmp_path / "task_sets/larybenchv2.v1.2/eval")
    unique = _task_set(tmp_path / "task_sets/train")
    _task_set(tmp_path / "task_sets/larybenchv2.v1.0/eval")
    _task_set(tmp_path / "task_sets/nested/deeper/hidden")
    catalog = evaluation_task_sets(_config(tmp_path))

    assert catalog == {
        "larybenchv2.v1.1/eval": first,
        "larybenchv2.v1.2/eval": second,
        "train": unique,
    }


def test_explicit_parent_root_discovers_its_task_sets(tmp_path):
    parent = tmp_path / "catalog"
    first = _task_set(parent / "larybenchv2.v1.1/eval")
    second = _task_set(parent / "larybenchv2.v1.2/eval")
    catalog = evaluation_task_sets(_config(tmp_path, evaluation=str(parent)))

    assert catalog["larybenchv2.v1.1/eval"] == first
    assert catalog["larybenchv2.v1.2/eval"] == second


@pytest.mark.parametrize("as_list", [False, True], ids=["scalar", "list"])
def test_catalog_falls_back_to_mounted_collection_sets(tmp_path, as_list):
    mounted = _task_set(tmp_path / "mounted/train")
    configured = [str(mounted)] if as_list else str(mounted)
    catalog = evaluation_task_sets(_config(tmp_path, collection=configured))

    assert catalog == {"train": mounted}


def test_missing_explicit_directory_is_listed_without_blocking_other_sets(tmp_path):
    missing = tmp_path / "missing"
    available = _task_set(tmp_path / "available")
    catalog = evaluation_task_sets(_config(tmp_path, evaluation=[str(missing), str(available)]))

    assert list(catalog) == ["missing", "available"]
    assert catalog["available"] == available
    assert catalog["missing"] == missing


def test_explicit_duplicate_names_use_parent_and_reject_remaining_ambiguity(tmp_path):
    first = _task_set(tmp_path / "first/batch")
    second = _task_set(tmp_path / "second/batch")
    catalog = evaluation_task_sets(_config(tmp_path, evaluation=[str(first), str(second)]))
    assert catalog == {"first/batch": first, "second/batch": second}

    conflicting = _task_set(tmp_path / "other/first/batch")
    with pytest.raises(ValueError, match="duplicate evaluation task-set name first/batch"):
        evaluation_task_sets(_config(tmp_path, evaluation=[str(first), str(conflicting)]))


def test_unconfigured_empty_catalog_returns_no_sets(tmp_path):
    assert evaluation_task_sets(_config(tmp_path)) == {}


def test_relative_explicit_paths_are_relative_to_the_repository(tmp_path):
    root = _task_set(tmp_path / "catalog/eval")
    relative = os.path.relpath(root, Path(__file__).resolve().parents[2])
    catalog = evaluation_task_sets(_config(tmp_path, evaluation=relative))

    assert catalog["eval"].resolve() == root
