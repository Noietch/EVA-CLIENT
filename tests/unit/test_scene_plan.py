import json

import pytest

from core.utils.scene_plan import plan_scenes, read_plan_yaml

pytestmark = pytest.mark.unit


def test_yaml_cache_reuses_parsing_and_reloads_changed_files(tmp_path, monkeypatch):
    from core.utils import scene_plan

    path = tmp_path / "info.yaml"
    path.write_text("collection_dir: datasets/first\n")
    original_load = scene_plan.yaml.load
    calls = []

    def load(*args, **kwargs):
        calls.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(scene_plan.yaml, "load", load)
    assert read_plan_yaml(tmp_path, "info.yaml")["collection_dir"] == "datasets/first"
    assert read_plan_yaml(tmp_path, "info.yaml")["collection_dir"] == "datasets/first"
    assert len(calls) == 1
    path.write_text("collection_dir: datasets/second\n")
    assert read_plan_yaml(tmp_path, "info.yaml")["collection_dir"] == "datasets/second"
    assert len(calls) == 2


def test_missing_yaml_is_not_cached(tmp_path):
    assert read_plan_yaml(tmp_path, "info.yaml") == {}
    (tmp_path / "info.yaml").write_text("task_order: spatial\n")
    assert read_plan_yaml(tmp_path, "info.yaml") == {"task_order": "spatial"}


def test_scene_placements_without_random_preserve_all_configured_positions():
    rows = [
        {
            "scene_id": "SC-001",
            "placements": json.dumps(
                [
                    {"object_id": "rack", "position_ids": ["P1", "P2", "P3"]},
                    {"object_id": "cup", "position_ids": ["P4"]},
                    {"object_id": "plate", "position_ids": ["P4"]},
                ]
            ),
        }
    ]
    (scene,) = plan_scenes(rows, {}, "unverified")

    assert [(item["object_id"], item["position_id"]) for item in scene["placements"]] == [
        ("rack", "P1"),
        ("rack", "P2"),
        ("rack", "P3"),
        ("cup", "P4"),
        ("plate", "P4"),
    ]
    assert len(scene["placement_groups"]) == 3
    assert "randomization" not in scene
    assert all("random" not in item for item in scene["placements"])
    assert all("random" not in item for item in scene["placement_groups"])


def test_scene_placements_preserve_the_configured_pose_and_fixed_flag():
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    rows = [
        {
            "scene_id": "SC-001",
            "placements": json.dumps(
                [
                    {"object_id": "cup", "position_ids": ["P1"], "pose_xyzw": pose, "fixed": False},
                ]
            ),
        }
    ]
    (scene,) = plan_scenes(rows, {}, "unverified")
    for placement in [*scene["placements"], *scene["placement_groups"]]:
        assert placement["pose_xyzw"] == pose
        assert placement["fixed"] is False


def test_scene_reference_image_names_are_split_trimmed_and_merged():
    rows = [
        {"scene_id": "SC-001", "ref_imgs": "front.jpg; side.jpg ; front.jpg"},
        {"scene_id": "SC-001", "ref_imgs": ";top.png; "},
        {"scene_id": "SC-002", "placements": "[]"},
    ]

    scenes = plan_scenes(rows, {}, "unverified")

    assert scenes[0]["reference_image_names"] == ["front.jpg", "side.jpg", "top.png"]
    assert scenes[1]["reference_image_names"] == []
