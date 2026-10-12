import json

import pytest

from core.utils.scene_plan import plan_scenes

pytestmark = pytest.mark.unit


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
