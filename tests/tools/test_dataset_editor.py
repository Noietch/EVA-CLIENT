"""Exercise task-plan batches, shared assets, and collection review."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from core.config import load_collection_task_set
from tools.datasets.app import (
    create_app,
)
from tools.datasets.assets import ObjectCatalog
from tools.datasets.store import TaskSetStore

pytestmark = pytest.mark.integration

BATCH = "bench_batch"


def _task_set(
    root: Path,
    dataset_name: str = "bench",
    objects_file: str = "objects.csv",
    write_objects: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "info.yaml").write_text(
        "dataset_name: " + dataset_name + "\n"
        "robot_type: arx_x5\n"
        "collection_dir: bench/raw\n"
        "task_count: 1\n"
        "target_episodes: 2\n"
        f"objects_file: {objects_file}\n",
        encoding="utf-8",
    )
    (root / "layout.yaml").write_text(
        "coordinate_frame: table_top_left\n"
        "unit: mm\n"
        "bounds: {width: 500, height: 500}\n"
        "sampling_points:\n"
        "- {position_id: P1, x: 0, y: 0}\n"
        "- {position_id: P2, x: 500, y: 500}\n",
        encoding="utf-8",
    )
    if write_objects:
        (root / "objects.csv").write_text(
            "\ufeffobject_id,object_name,object_name_zh,color\nOBJ-1,cup,杯子,white\n",
            encoding="utf-8",
        )
    (root / "scene.csv").write_text(
        '\ufeffscene_id,placements\nSC-1,"[{""position_ids"":[""P1""],""object_id"":""OBJ-1""}]"\n',
        encoding="utf-8",
    )
    (root / "tasks.csv").write_text(
        "\ufefftask_id,action,category,operation_object_ids,prompt_en,prompt_zh,"
        "total_epsiodes_count,scene_ids,scene_epsiodes_count\n"
        "TASK-1,pickplace,cup,OBJ-1,pick up cup,拿起杯子,2,SC-1,2\n",
        encoding="utf-8",
    )
    return root


def _workspace(tmp_path: Path):
    plans = tmp_path / "task_sets"
    _task_set(plans / BATCH, objects_file="../../assets/objects.csv", write_objects=False)
    assets = tmp_path / "assets"
    ObjectCatalog(
        assets,
        [{"object_id": "OBJ-1", "object_name": "cup", "object_name_zh": "杯子"}],
    )
    collection = tmp_path / "collection"
    app = create_app(plans, assets, collection)
    app.config["TESTING"] = True
    client = app.test_client()
    client.environ_base["HTTP_X_EVA_DATASET_EDITOR"] = "1"
    client.environ_base["HTTP_X_EVA_EDIT_MODE"] = "1"
    return client, plans, assets, collection


def _episode_dataset(collection: Path) -> Path:
    root = collection / "bench" / "raw"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint_a", "joint_b"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["action_a", "action_b"],
            },
        },
        "robot_type": "arx_x5",
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    episode = {
        "episode_index": 7,
        "length": 3,
        "task_id": "TASK-1",
        "scene_id": "SC-1",
        "scene_round": 0,
        "slot_id": "TASK-1:SC-1:0",
        "quality": "green",
    }
    (root / "meta/episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    vector = pa.list_(pa.float32(), 2)
    table = pa.table(
        {
            "observation.state": pa.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], vector),
            "action": pa.array([[1.0, 1.1], [1.0, 1.1], [1.4, 1.5]], vector),
            "timestamp": pa.array([0.0, 0.1, 0.2]),
        }
    )
    pq.write_table(table, root / "data/chunk-000/episode_000007.parquet")
    return root


def test_unknown_robot_does_not_block_catalog_or_filtering(tmp_path):
    client, plans, _, _ = _workspace(tmp_path)
    unknown = _task_set(plans / "unknown_robot", dataset_name="unknown")
    info = unknown / "info.yaml"
    info.write_text(info.read_text().replace("robot_type: arx_x5", "robot_type: agile_x"))

    response = client.get("/api/state")
    assert response.status_code == 200
    state = response.get_json()
    batches = {batch["batch_id"]: batch for batch in state["batches"]}
    assert batches["unknown_robot"]["robot_type"] == "agile_x"
    assert batches["unknown_robot"]["cameras"] == []
    assert batches[BATCH]["cameras"]
    assert state["objects"]

    filtered = client.get("/api/state?robot_type=agile_x")
    assert filtered.status_code == 200
    assert [batch["batch_id"] for batch in filtered.get_json()["batches"]] == ["unknown_robot"]
    assert filtered.get_json()["tasks"]


def test_object_measurements_round_trip_and_read_only(tmp_path):
    client, plans, assets, collection = _workspace(tmp_path)
    # Existing catalogs without measurement columns remain readable without migration.
    legacy = "object_id,object_name,object_name_zh,color,photo_dir\nOBJ-1,cup,杯子,white,杯子\n"
    (assets / "objects.csv").write_text(legacy, encoding="utf-8")
    catalog = ObjectCatalog(assets, [])
    assert catalog.records()[0]["mass_g"] == ""
    assert (assets / "objects.csv").read_text(encoding="utf-8") == legacy
    payload = {
        "object_id": "OBJ-1",
        "object_name": "cup",
        "length_cm": "20.5",
        "width_cm": "15",
        "height_cm": "3.2",
        "mass_g": "180",
        "modeling_method": "A",
    }
    assert client.put("/api/objects/OBJ-1", json=payload).status_code == 200
    saved = ObjectCatalog(assets, []).records()[0]
    for key, value in payload.items():
        assert saved[key] == value
    for method in ("B", "C", "D", ""):
        payload["modeling_method"] = method
        payload["height_cm"] = ""
        assert client.put("/api/objects/OBJ-1", json=payload).status_code == 200
        saved = ObjectCatalog(assets, []).records()[0]
        assert saved["modeling_method"] == method
        assert saved["height_cm"] == ""
    before = (assets / "objects.csv").read_bytes()
    readonly = create_app(plans, assets, collection, read_only=True).test_client()
    response = readonly.put(
        "/api/objects/OBJ-1",
        json=payload,
        headers={
            "X-EVA-Dataset-Editor": "1",
            "X-EVA-Edit-Mode": "1",
        },
    )
    assert response.status_code == 403
    assert (assets / "objects.csv").read_bytes() == before


@pytest.mark.parametrize("field", ["length_cm", "width_cm", "height_cm", "mass_g"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc"])
def test_object_rejects_invalid_measurements(tmp_path, field, value):
    client, _, assets, _ = _workspace(tmp_path)
    before = (assets / "objects.csv").read_bytes()
    response = client.put(
        "/api/objects/OBJ-1",
        json={
            "object_id": "OBJ-1",
            "object_name": "cup",
            field: value,
        },
    )
    assert response.status_code == 400
    assert (assets / "objects.csv").read_bytes() == before


def test_object_rejects_unknown_modeling_method(tmp_path):
    client, _, assets, _ = _workspace(tmp_path)
    before = (assets / "objects.csv").read_bytes()
    response = client.put(
        "/api/objects/OBJ-1",
        json={
            "object_id": "OBJ-1",
            "object_name": "cup",
            "modeling_method": "E",
        },
    )
    assert response.status_code == 400
    assert (assets / "objects.csv").read_bytes() == before


def test_batch_filter_and_plan_crud_round_trip(tmp_path):
    client, plans, _, _ = _workspace(tmp_path)
    _task_set(plans / "other_batch", "other")
    state = client.get("/api/state?batch=" + BATCH).get_json()
    assert [plan["batch_id"] for plan in state["plans"]] == [BATCH]

    assert (
        client.post(
            "/api/objects",
            json={
                "object_id": "OBJ-2",
                "object_name": "plate",
                "object_name_zh": "盘子",
                "color": "black",
                "photo_dir": "盘子",
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/batches/" + BATCH + "/scenes",
            json={
                "scene_id": "SC-2",
                "placements": [{"object_id": "OBJ-2", "position_ids": ["P2"]}],
            },
        ).status_code
        == 200
    )
    response = client.post(
        "/api/batches/" + BATCH + "/tasks",
        json={
            "task_id": "TASK-2",
            "action": "pickplace",
            "category": "plate",
            "operation_object_ids": ["OBJ-2"],
            "prompt_en": "move plate",
            "prompt_zh": "移动盘子",
            "scene_ids": ["SC-2"],
            "scene_epsiodes_count": [3],
            "total_epsiodes_count": 3,
        },
    )
    assert response.status_code == 200
    state = response.get_json()
    assert len(state["tasks"]) == 2
    assert len(state["tasks"][1]["slots"]) == 3
    assert state["scenes"][-1]["placements"] == [{"position_ids": ["P2"], "object_id": "OBJ-2"}]
    assert TaskSetStore(plans / BATCH).state()["info"]["target_episodes"] == 5


def test_reference_conflicts_and_invalid_plan_do_not_mutate(tmp_path):
    client, plans, _, _ = _workspace(tmp_path)
    before = (plans / BATCH / "tasks.csv").read_bytes()
    assert client.delete("/api/objects/OBJ-1").status_code == 409
    assert client.delete("/api/batches/" + BATCH + "/scenes/SC-1").status_code == 409
    invalid = {
        "task_id": "TASK-BAD",
        "action": "pickplace",
        "category": "",
        "operation_object_ids": ["OBJ-404"],
        "prompt_en": "bad reference",
        "prompt_zh": "",
        "scene_ids": ["SC-1"],
        "scene_epsiodes_count": [2],
        "total_epsiodes_count": 3,
    }
    response = client.post("/api/batches/" + BATCH + "/tasks", json=invalid)
    assert response.status_code == 400
    assert "Unknown objects OBJ-404" in response.get_json()["error"]
    assert (plans / BATCH / "tasks.csv").read_bytes() == before


def test_episode_matches_slot_and_review_returns_series(tmp_path):
    client, _, _, collection = _workspace(tmp_path)
    _episode_dataset(collection)
    state = client.get("/api/state?batch=" + BATCH).get_json()
    task = state["tasks"][0]
    assert task["counts"] == {"complete": 1, "pending": 1, "repair": 0}
    assert task["slots"][0]["episode"]["episode_index"] == 7
    assert task["slots"][1]["episode"] is None

    client.application.config["DATASET_READ_ONLY"] = True
    response = client.get(
        "/api/review?batch=" + BATCH + "&slot_id=TASK-1:SC-1:0",
    )
    review = response.get_json()
    assert response.status_code == 200
    assert review["series"]["state_names"] == ["joint_a", "joint_b"]
    assert review["series"]["action"][2] == pytest.approx([1.4, 1.5])
    assert review["series"]["frame_labels"] == ["static", "static", "non-static"]
    assert review["fps"] == 10
    transforms = client.get("/api/batches/" + BATCH + "/episodes/7/transforms?start=0&count=2")
    assert transforms.status_code == 200
    assert transforms.data.startswith(b"EVAXFRM1")
    assert transforms.headers["X-EVA-Transform-Total"] == "3"


def test_dashboard_summary_excludes_episode_without_matching_slot(tmp_path):
    client, _, _, collection = _workspace(tmp_path)
    root = _episode_dataset(collection)
    episodes_path = root / "meta/episodes.jsonl"
    orphan = {
        "episode_index": 8,
        "length": 5,
        "task_id": "TASK-1",
        "scene_id": "SC-OTHER",
        "scene_round": 0,
        "slot_id": "TASK-1:SC-OTHER:0",
        "quality": "green",
    }
    with episodes_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(orphan) + "\n")

    state = client.get("/api/state?batch=" + BATCH).get_json()
    summary = next(item for item in state["all_batches"] if item["batch_id"] == BATCH)
    assert summary["collected"] == 1
    assert summary["pending"] == 1
    assert summary["frames"] == 3


def test_middle_static_frames_auto_fail_and_expose_trim_boundaries(tmp_path):
    client, _, _, collection = _workspace(tmp_path)
    root = _episode_dataset(collection)
    vector = pa.list_(pa.float32(), 1)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array([[0], [1], [1], [2]], vector),
                "action": pa.array([[0], [1], [1], [2]], vector),
                "timestamp": pa.array([0.0, 0.1, 0.2, 0.3]),
            }
        ),
        root / "data/chunk-000/episode_000007.parquet",
    )
    row_path = root / "meta/episodes.jsonl"
    row = json.loads(row_path.read_text())
    row["length"] = 4
    row_path.write_text(json.dumps(row) + "\n")

    state = client.get("/api/state?batch=" + BATCH).get_json()
    slot = state["tasks"][0]["slots"][0]
    assert slot["qc_state"] == "failed"
    assert slot["episode"]["qc_reason"] == "static_frames_excessive"
    review = client.get("/api/review?batch=" + BATCH + "&slot_id=TASK-1:SC-1:0").get_json()
    analysis = review["series"]["frame_label_analysis"]
    assert analysis["trim_start_frame"] == 1
    assert analysis["trim_end_frame"] == 4
    assert analysis["middle_static_frames"] == 1


def test_trim_episode_updates_parquet_and_episode_metadata(tmp_path):
    client, _, _, collection = _workspace(tmp_path)
    root = _episode_dataset(collection)
    response = client.post(
        "/api/batches/" + BATCH + "/episodes/7/trim",
        json={"start_frame": 1, "end_frame": 3},
    )
    assert response.status_code == 200
    assert response.get_json()["length"] == 2
    table = pq.read_table(root / "data/chunk-000/episode_000007.parquet")
    assert table.num_rows == 2
    assert table["timestamp"].to_pylist() == [0.0, 0.1]
    row = json.loads((root / "meta/episodes.jsonl").read_text())
    assert row["length"] == 2
    assert row["trim_start_frame"] == 1
    assert row["trim_end_frame"] == 3


@pytest.mark.parametrize("quality,verdict", [("red", ""), ("RED", "pass"), ("green", "fail")])
def test_rejected_episode_requires_repair(tmp_path, quality, verdict):
    client, _, _, collection = _workspace(tmp_path)
    root = _episode_dataset(collection)
    path = root / "meta/episodes.jsonl"
    episode = json.loads(path.read_text())
    episode.update(quality=quality, qc_verdict=verdict)
    path.write_text(json.dumps(episode) + "\n")

    state = client.get("/api/state?batch=" + BATCH).get_json()

    assert state["tasks"][0]["slots"][0]["state"] == "repair"
    assert state["tasks"][0]["counts"] == {"complete": 0, "pending": 1, "repair": 1}


def test_qc_updates_episode_and_slot_state(tmp_path):
    client, _, _, collection = _workspace(tmp_path)
    root = _episode_dataset(collection)
    response = client.put(
        "/api/batches/" + BATCH + "/episodes/7/qc",
        json={"verdict": "fail", "note": "grasp missed"},
    )
    assert response.status_code == 200
    slot = response.get_json()["tasks"][0]["slots"][0]
    assert slot["state"] == "repair"
    qc = json.loads((root / "meta/qc.jsonl").read_text().strip())
    assert qc["qc_verdict"] == "fail"
    assert qc["qc_note"] == "grasp missed"


def test_object_photo_upload_and_placeholder(tmp_path):
    client, _, _, _ = _workspace(tmp_path)
    state = client.get("/api/state").get_json()
    assert state["objects"][0]["photos"] == []
    content = io.BytesIO()
    Image.effect_noise((800, 600), 100).convert("RGB").save(content, format="PNG")
    rejected = client.post(
        "/api/objects/OBJ-1/photos",
        data={
            "photos": [
                (io.BytesIO(content.getvalue()), "partial.png"),
                (io.BytesIO(b"not an image"), "invalid.txt"),
            ]
        },
        content_type="multipart/form-data",
    )
    assert rejected.status_code == 400
    assert not (Path(state["assets_root"]) / "object_photos/杯子/partial.png").exists()
    response = client.post(
        "/api/objects/OBJ-1/photos",
        data={"photos": (io.BytesIO(content.getvalue()), "front.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["photos"] == ["front.png"]
    photo = client.get("/api/objects/OBJ-1/photos/front.png")
    assert photo.status_code == 200
    assert photo.mimetype == "image/png"
    thumbnail = client.get("/api/objects/OBJ-1/previews/thumb/front.png")
    assert thumbnail.status_code == 200
    assert thumbnail.mimetype == "image/webp"
    with Image.open(io.BytesIO(thumbnail.data)) as image:
        assert max(image.size) == 192
    display = client.get("/api/objects/OBJ-1/previews/display/front.png")
    assert display.status_code == 200
    with Image.open(io.BytesIO(display.data)) as image:
        assert image.size == (800, 600)
    assert len(thumbnail.data) < len(content.getvalue())


def test_zip_export_import_and_metadata_update(tmp_path):
    client, plans, _, _ = _workspace(tmp_path)
    response = client.put(
        "/api/batches/" + BATCH + "/info",
        json={
            "dataset_name": "exported set",
            "robot_type": "arx_x5",
            "collection_dir": "bench/raw",
        },
    )
    assert response.status_code == 200
    exported = client.get("/api/batches/" + BATCH + "/export")
    assert exported.status_code == 200
    assert "filename=bench_batch.zip" in exported.headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(exported.data)) as archive:
        assert set(archive.namelist()) == {
            "info.yaml",
            "layout.yaml",
            "scene.csv",
            "tasks.csv",
        }
        info = yaml.safe_load(archive.read("info.yaml"))
        assert info["dataset_name"] == "exported set"
        assert info["collection_dir"] == "bench/raw"

    _task_set(plans / "destination", "old")
    imported = client.post(
        "/api/batches/destination/import",
        data={"file": (io.BytesIO(exported.data), "task-set.zip")},
        content_type="multipart/form-data",
    )
    assert imported.status_code == 200
    assert imported.get_json()["plans"][0]["info"]["dataset_name"] == "exported set"


def test_invalid_archive_and_write_guard(tmp_path):
    client, _, _, _ = _workspace(tmp_path)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("info.yaml", "dataset_name: incomplete\n")
    response = client.post(
        "/api/batches/" + BATCH + "/import",
        data={"file": (io.BytesIO(stream.getvalue()), "bad.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "missing" in response.get_json()["error"]

    plans = tmp_path / "guard/task_sets"
    _task_set(plans / BATCH)
    app = create_app(
        plans,
        tmp_path / "guard/assets",
        tmp_path / "guard/collection",
    )
    app.config["TESTING"] = True
    guarded_client = app.test_client()
    guarded = guarded_client.post(
        "/api/objects",
        json={"object_id": "OBJ-2", "object_name": "plate", "object_name_zh": "盘子"},
    )
    assert guarded.status_code == 403
    assert guarded.is_json
    guarded_client.environ_base["HTTP_X_EVA_DATASET_EDITOR"] = "1"
    assert (
        guarded_client.post(
            "/api/objects",
            json={"object_id": "OBJ-2", "object_name": "plate", "object_name_zh": "盘子"},
        ).status_code
        == 403
    )
    assert guarded_client.get("/api/batches/" + BATCH + "/validate").status_code == 200


def test_store_import_rolls_back_on_replace_failure(tmp_path, monkeypatch):
    source_root = _task_set(tmp_path / "source")
    destination_root = _task_set(tmp_path / "destination")
    source = TaskSetStore(source_root)
    destination = TaskSetStore(destination_root)
    source.update_info(
        {
            "dataset_name": "replacement",
            "robot_type": "arx_x5",
            "collection_dir": "bench/raw",
        }
    )
    archive = source.export_zip().getvalue()
    filenames = ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv")
    before = {name: (destination_root / name).read_bytes() for name in filenames}
    original = destination._atomic_write
    failed = False

    def fail_scene_once(filename, content):
        nonlocal failed
        if filename == "scene.csv" and not failed:
            failed = True
            raise OSError("injected failure")
        original(filename, content)

    monkeypatch.setattr(destination, "_atomic_write", fail_scene_once)
    with pytest.raises(OSError, match="injected failure"):
        destination.import_zip(archive)
    assert {name: (destination_root / name).read_bytes() for name in filenames} == before


def test_task_set_round_trips_into_production_loader(tmp_path):
    source = TaskSetStore(_task_set(tmp_path / "source"))
    shared_assets = ObjectCatalog(tmp_path / "assets", source.state()["objects"])
    imported = TaskSetStore(tmp_path / "official-copy", shared_assets.path)
    state = imported.import_zip(source.export_zip().getvalue())
    assert (len(state["objects"]), len(state["scenes"]), len(state["tasks"])) == (1, 1, 1)
    assert state["issues"] == []
    production_tasks = load_collection_task_set(imported.root)
    assert sum(target for _, target in production_tasks[imported.root.name]) == 2
