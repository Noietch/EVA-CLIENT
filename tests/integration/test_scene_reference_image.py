"""Task-set reference images remain independent of scenes and rendering."""

import csv

import pytest
from PIL import Image

from core.config import ConfigDict
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def test_reference_images_follow_task_set_and_scope(tmp_path):
    roots = [tmp_path / "collect", tmp_path / "eval"]
    for root, color in zip(roots, ["red", "blue"], strict=True):
        (root / "ref_imgs").mkdir(parents=True)
        Image.new("RGB", (8, 6), color).save(root / "ref_imgs/摆放照片.png")
        (root / "layout.yaml").write_text("ref_imgs: ref_imgs\n")
        (root / "tasks.csv").write_text(
            "task_id,prompt_en,scene_ids,scene_epsiodes_count,total_epsiodes_count\n"
            "TASK-1,put cup,SC/1,1,1\n"
        )
        with (root / "scene.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scene_id", "placements"])
            writer.writerow(["SC/1", "[]"])
    Image.new("RGB", (8, 6), "green").save(tmp_path / "outside.png")
    config = console_config(collection={"task_set_dir": str(roots[0])})
    config.eval = ConfigDict(task_set_dir=str(roots[1]))
    with serve_console(config) as console:
        for root, scope in zip(roots, ["collect", "eval"], strict=True):
            plan = console.get(f"/api/scene_plan?set={root.name}&scope={scope}")
            assert plan.status == 200
            image, = plan.json["reference_images"]
            assert image["name"] == "摆放照片.png"
            response = console.get(image["url"])
            assert response.status == 200
            assert response.headers["content-type"] == "image/png"
            assert response.raw == (root / "ref_imgs/摆放照片.png").read_bytes()
            for filename in ["missing.png", "..%2Foutside.png"]:
                response = console.get(
                    f"/api/scene_plan/reference-image/{filename}"
                    f"?set={root.name}&scope={scope}"
                )
                assert response.status == 404


def test_reference_image_folder_changes_invalidate_plan(tmp_path):
    (tmp_path / "tasks.csv").write_text("task_id,prompt_en\nT1,put cup\n")
    config = console_config(collection={"task_set_dir": str(tmp_path)})
    with serve_console(config) as console:
        url = f"/api/scene_plan?set={tmp_path.name}"
        original = console.get(url)
        assert original.json["reference_images"] == []
        (tmp_path / "ref_imgs").mkdir()
        (tmp_path / "layout.yaml").write_text("ref_imgs: ref_imgs\n")
        empty = console.get(url)
        assert empty.json["reference_images"] == []
        assert empty.json["version"] != original.json["version"]
        Image.new("RGB", (8, 6), "red").save(tmp_path / "ref_imgs/photo.png")
        added = console.get(url)
        assert [image["name"] for image in added.json["reference_images"]] == ["photo.png"]
        assert added.json["version"] != empty.json["version"]
        (tmp_path / "ref_imgs/notes.txt").write_text("Reference folder, no rendering required.")
        assert console.get(url).json["version"] == added.json["version"]
