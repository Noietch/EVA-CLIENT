"""Scene catalog compatibility and object photo delivery."""

import csv
import json
from types import SimpleNamespace

import pytest

from core.app.console import server

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "id_column,name_column",
    [
        ("object_id", "object_name_zh"),
        ("物品代码", "名称"),
    ],
)
def test_scene_catalog_keeps_legacy_placements_and_serves_photos(
    tmp_path, monkeypatch, id_column, name_column
):
    root = tmp_path / "plans" / "example"
    root.mkdir(parents=True)
    assets = tmp_path / "assets"
    photo_dir = assets / "object_photos" / "cup"
    photo_dir.mkdir(parents=True)
    photo = photo_dir / "cup top.png"
    photo.write_bytes(b"photo fixture")
    (root / "info.yaml").write_text("objects_file: ../../assets/objects.csv\n")
    with (assets / "objects.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column, name_column, "photo_dir"])
        writer.writerow(["AST-001", "Cup", "cup"])
    with (root / "scene.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", "placements"])
        writer.writerow(
            [
                "SC-001",
                json.dumps(
                    [
                        {"object_id": "AST-001", "position_ids": ["P1", "P2"]},
                        {"object_id": "AST-001", "position_ids": ["P3"]},
                    ]
                ),
            ]
        )

    plan = server._build_scene_plan(root)
    scene = plan["scenes"][0]
    assert len(scene["placements"]) == 3
    assert all("random" not in group for group in scene["placement_groups"])
    for placement in scene["placements"] + scene["placement_groups"]:
        assert placement["name"] == "Cup"
        assert placement["photo_url"] == "/api/scene_plan/object-photo/AST-001/cup%20top.png?set=example"

    monkeypatch.setattr(server, "_scene_plan_root", lambda config, dataset=None: root)
    delivered = []
    rejected = []
    handler = SimpleNamespace(
        ctx=SimpleNamespace(config=None, runtime=SimpleNamespace(active_config=None)),
        _query_str=lambda key: "example",
        _send_static=delivered.append,
        _send_empty=rejected.append,
    )
    server.ConsoleRequestHandler._send_scene_plan_photo(handler, "AST-001/cup%20top.png")
    assert delivered == [photo]
    assert rejected == []
    server.ConsoleRequestHandler._send_scene_plan_photo(handler, "AST-001/..%2F..%2Fobjects.csv")
    assert rejected == [404]


def test_scene_placements_reject_invalid_position_ids():
    placements = [{"object_id": "cup", "position_ids": "P1"}]
    assert server._scene_plan_placements(json.dumps(placements)) == []
