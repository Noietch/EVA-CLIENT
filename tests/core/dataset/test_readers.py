from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcap.reader import make_reader

from core.datasets import EpisodeData, VideoRef, detect_data_format, open_dataset


def test_public_containers_have_expected_defaults(tmp_path) -> None:
    video = VideoRef(path=tmp_path / "clip.mp4")
    episode = EpisodeData(columns={"action": np.zeros((1, 2), dtype=np.float32)})

    assert video.start_time == 0.0
    assert video.end_time is None
    assert episode.task == ""
    assert episode.fps is None
    assert episode.videos == {}
    assert episode.images == {}


def test_dataset_writer_requires_explicit_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit write format"):
        open_dataset(tmp_path, "auto", fps=30)


def test_hdf5_writer_streams_one_episode_and_reopens(tmp_path) -> None:
    dataset = open_dataset(tmp_path, "hdf5", fps=30)
    frames_yielded = 0

    def frames():
        nonlocal frames_yielded
        for value in (1, 2):
            frames_yielded += 1
            yield np.full((3, 4, 3), value, dtype=np.uint8)

    dataset.write(
        columns={
            "observation.state": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "action": np.asarray([[3.0], [4.0]], dtype=np.float32),
        },
        row={"episode_index": 0, "tasks": ["pick cup"], "length": 2},
        videos={"observation.images.front": frames()},
    )

    assert frames_yielded == 2
    reopened = open_dataset(tmp_path)
    episode = reopened.load(0)
    assert episode.task == "pick cup"
    assert episode.fps == 30.0
    np.testing.assert_array_equal(
        episode.images["observation.images.front"],
        np.asarray(
            [
                np.full((3, 4, 3), 1, dtype=np.uint8),
                np.full((3, 4, 3), 2, dtype=np.uint8),
            ]
        ),
    )
    np.testing.assert_array_equal(
        reopened.load_image_frame(0, "observation.images.front", 1),
        np.full((3, 4, 3), 2, dtype=np.uint8),
    )
    assert reopened.load_image_frame(0, "observation.images.front", 2) is None
    assert reopened.episode_rows() == [
        {"episode_index": 0, "length": 2, "tasks": ["pick cup"], "task": "pick cup", "fps": 30.0}
    ]


def test_open_dataset_reads_lerobot_v2_and_infers_ui_keys(tmp_path) -> None:
    dataset_dir = tmp_path / "lerobot_v2"
    _write_lerobot_v2_dataset(dataset_dir)

    reader = open_dataset(dataset_dir)
    episode = reader.load(0)

    assert reader.format == "lerobot_v21"
    assert reader.count_episodes() == 1
    assert episode.task == "pick up cup"
    assert episode.fps == 30.0
    np.testing.assert_allclose(
        episode.columns["observations.state.qpos"],
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        episode.columns["action"],
        np.asarray([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
    )
    assert episode.videos["observation.images.cam_high"].path == (
        dataset_dir / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    )
    inferred = reader.infer_keys()
    assert inferred["image"]["default"] == "observation.images.cam_high"
    assert inferred["state"]["default"] == "observations.state.qpos"
    assert inferred["action"]["default"] == "action"


def test_open_dataset_reads_lerobot_v3_shared_shard_with_filter_and_video_offsets(
    tmp_path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "lerobot_v3"
    shared_data_path = _write_lerobot_v3_dataset(dataset_dir)
    original_read_table = pq.read_table
    calls: list[tuple[Path, object]] = []

    def tracking_read_table(source, *args, **kwargs):
        calls.append((Path(source), kwargs.get("filters")))
        return original_read_table(source, *args, **kwargs)

    monkeypatch.setattr("core.datasets.lerobot.pq.read_table", tracking_read_table)

    reader = open_dataset(dataset_dir, format="lerobot3.0")
    episode = reader.load(1)

    assert reader.format == "lerobot_v3"
    assert reader.count_episodes() == 2
    assert episode.task == "place cube"
    np.testing.assert_allclose(
        episode.columns["observation.state"],
        np.asarray([[20.0], [21.0], [22.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        episode.columns["action"],
        np.asarray([[30.0], [31.0], [32.0]], dtype=np.float32),
    )
    video = episode.videos["observation.images.front"]
    assert video.path == dataset_dir / "videos/observation.images.front/chunk-000/file-000.mp4"
    assert video.start_time == 1.0
    assert video.end_time == 2.5
    inferred = reader.infer_keys()
    assert inferred["image"]["default"] == "observation.images.front"
    assert inferred["state"]["default"] == "observation.state"
    assert inferred["action"]["default"] == "action"

    assert any(
        path == shared_data_path and filters == [("episode_index", "=", 1)]
        for path, filters in calls
    )


def test_open_dataset_reads_lerobot_v3_with_legacy_tasks_jsonl(tmp_path) -> None:
    dataset_dir = tmp_path / "lerobot_v3_legacy"
    _write_lerobot_v3_dataset(dataset_dir, write_tasks_parquet=False)

    episode = open_dataset(dataset_dir).load(0)

    assert episode.task == "pick cube"


def test_open_dataset_reads_lerobot_v3_official_default_paths(tmp_path) -> None:
    dataset_dir = tmp_path / "lerobot_v3_defaults"
    _write_lerobot_v3_dataset(dataset_dir, include_path_templates=False)

    episode = open_dataset(dataset_dir).load(1)

    assert episode.columns["action"].shape == (3, 1)
    assert episode.videos["observation.images.front"].path == (
        dataset_dir / "videos/observation.images.front/chunk-000/file-000.mp4"
    )


def test_lerobot_v3_write_streams_episodes_into_row_groups(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "dataset_format": "lerobot_v3",
                "codebase_version": "v3.0",
                "total_episodes": 2,
                "fps": 30,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [1]},
                    "action": {"dtype": "float32", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "index": {"dtype": "int64", "shape": [1]},
                },
            }
        )
    )
    dataset = open_dataset(tmp_path, "lerobot_v3", fps=30)
    for episode_index in range(2):
        start = episode_index * 2
        dataset.write(
            {
                "observation.state": np.asarray([[start], [start + 1]], dtype=np.float32),
                "action": np.asarray([[start + 2], [start + 3]], dtype=np.float32),
                "episode_index": np.full(2, episode_index, dtype=np.int64),
                "index": np.asarray([start, start + 1], dtype=np.int64),
            },
            {"episode_index": episode_index, "tasks": ["task"], "length": 2},
            {},
        )
    dataset.finalize()

    data_path = tmp_path / "data/chunk-000/file-000.parquet"
    metadata_path = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    assert pq.ParquetFile(data_path).num_row_groups == 2
    assert pq.read_table(metadata_path).num_rows == 2
    assert not any("episode" in key and key not in {"_episode_writer"} for key in dataset.__dict__)
    np.testing.assert_array_equal(open_dataset(tmp_path).load(1).columns["action"], [[4], [5]])


def test_lerobot_v21_patch_episode_and_annotation_roundtrip(tmp_path) -> None:
    dataset_dir = tmp_path / "lerobot_v2_patch"
    _write_lerobot_v2_dataset(dataset_dir)
    (dataset_dir / "meta/episodes.jsonl").unlink()

    dataset = open_dataset(dataset_dir)
    assert dataset.episode_rows() == [{"episode_index": 0, "tasks": ["pick up cup"], "length": 2}]
    assert dataset.patch_episode(0, {"score": 1.0}) is True
    assert dataset.mark_qc(0, "pass", "checked") is True
    assert dataset.annotate_language(0, "spoken instruction") is True

    reopened = open_dataset(dataset_dir)
    rows = reopened.episode_rows()
    assert rows[0]["score"] == 1.0
    assert rows[0]["qc_verdict"] == "pass"
    assert rows[0]["qc_note"] == "checked"
    assert reopened.read_annotation(0) == "spoken instruction"

    table = pq.read_table(dataset_dir / "data/chunk-000/episode_000000.parquet")
    assert table.column("language_annotation").to_pylist() == ["spoken instruction"] * 2
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    assert info["features"]["language_annotation"] == {
        "dtype": "string",
        "shape": [1],
        "names": None,
    }


def test_hdf5_and_mcap_metadata_patch_api_is_explicitly_unsupported(tmp_path) -> None:
    hdf5 = open_dataset(tmp_path / "hdf5_meta", "hdf5", fps=10)
    hdf5.write(
        {"state": np.asarray([[1.0]], dtype=np.float32)},
        {"episode_index": 0, "length": 1, "tasks": ["hdf5 task"]},
        {},
    )
    assert hdf5.patch_episode(0, {"score": 1.0}) is False
    assert hdf5.mark_qc(0, "pass", "ok") is False
    assert hdf5.annotate_language(0, "note") is False
    assert hdf5.read_annotation(0) == ""

    mcap = open_dataset(tmp_path / "mcap_meta", "mcap", fps=10)
    mcap.write(
        {"state": np.asarray([[1.0]], dtype=np.float32)},
        {"episode_index": 0, "length": 1, "tasks": ["mcap task"]},
        {},
    )
    assert mcap.patch_episode(0, {"score": 1.0}) is False
    assert mcap.mark_qc(0, "pass", "ok") is False
    assert mcap.annotate_language(0, "note") is False
    assert mcap.read_annotation(0) == ""


def test_mcap_episode_roundtrip_preserves_streamed_images_and_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "mcap"
    columns = {
        "observations.state": np.arange(9, dtype=np.float32).reshape(3, 3),
        "action": np.array([1, 2, 3], dtype=np.int16),
    }
    yielded = 0

    def frames():
        nonlocal yielded
        for frame_index in range(3):
            yielded += 1
            yield np.full((2, 2, 3), frame_index + 1, dtype=np.uint8)

    row = {
        "episode_index": 0,
        "length": 3,
        "tasks": ["close drawer"],
        "task": "close drawer",
        "score": np.float32(0.75),
        "notes": np.array(["good", "fast"], dtype=object),
    }

    open_dataset(root, "mcap", fps=30).write(
        columns,
        row,
        {"observation.images.front": frames()},
    )
    path = root / "data/chunk-000/episode_000000.mcap"

    raw = path.read_bytes()
    magic = b"\x89MCAP0\r\n"
    assert raw[: len(magic)] == magic
    assert raw[-len(magic) :] == magic
    assert detect_data_format(path) == "mcap"
    assert yielded == 3

    reopened = open_dataset(path)
    episode = reopened.load(0)
    np.testing.assert_array_equal(
        episode.columns["observations.state"],
        columns["observations.state"],
    )
    np.testing.assert_array_equal(episode.columns["action"], columns["action"])
    np.testing.assert_array_equal(
        episode.images["observation.images.front"],
        np.asarray(
            [
                np.full((2, 2, 3), 1, dtype=np.uint8),
                np.full((2, 2, 3), 2, dtype=np.uint8),
                np.full((2, 2, 3), 3, dtype=np.uint8),
            ]
        ),
    )
    assert episode.task == row["task"]
    np.testing.assert_array_equal(
        reopened.load_image_frame(0, "observation.images.front", 1),
        np.full((2, 2, 3), 2, dtype=np.uint8),
    )
    assert reopened.load_image_frame(0, "observation.images.front", 3) is None

    rows = reopened.episode_rows()
    assert rows[0]["episode_index"] == 0
    assert rows[0]["length"] == 3
    assert rows[0]["tasks"] == ["close drawer"]
    assert rows[0]["task"] == "close drawer"
    assert rows[0]["score"] == pytest.approx(0.75)
    np.testing.assert_array_equal(rows[0]["notes"], np.array(["good", "fast"], dtype=object))
    assert rows[0]["fps"] == 30.0

    image_times = []
    with path.open("rb") as stream:
        for _schema, channel, message in make_reader(stream).iter_messages():
            if channel.topic == "episode/image/observation.images.front":
                image_times.append((message.log_time, message.publish_time))
    assert image_times == [
        (0, 0),
        (33_333_333, 33_333_333),
        (66_666_667, 66_666_667),
    ]


def test_mcap_writer_asserts_strictly_increasing_frame_timestamps(tmp_path: Path) -> None:
    dataset = open_dataset(tmp_path / "mcap_ns_assert", "mcap", fps=2_000_000_000)
    with pytest.raises(AssertionError, match="strictly increase"):
        dataset.write(
            {"state": np.asarray([[0.0], [1.0]], dtype=np.float32)},
            {"episode_index": 0, "length": 2, "tasks": ["fast"]},
            {"observation.images.front": np.zeros((2, 2, 2, 3), dtype=np.uint8)},
        )


def _write_lerobot_v2_dataset(dataset_dir: Path) -> None:
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data/chunk-000"
    video_path = dataset_dir / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"v2")
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observations.state.qpos": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.images.cam_high": {"dtype": "video", "shape": [4, 4, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up cup"}) + "\n"
    )
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick up cup"], "length": 2}) + "\n"
    )
    table = pa.table(
        {
            "observations.state.qpos": [[0.0, 1.0], [2.0, 3.0]],
            "action": [[10.0, 11.0], [12.0, 13.0]],
            "timestamp": [0.0, 1.0 / 30.0],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [0, 1],
            "task_index": [0, 0],
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")


def _write_lerobot_v3_dataset(
    dataset_dir: Path,
    write_tasks_parquet: bool = True,
    include_path_templates: bool = True,
) -> Path:
    meta_dir = dataset_dir / "meta"
    episode_meta_dir = meta_dir / "episodes/chunk-000"
    data_dir = dataset_dir / "data/chunk-000"
    video_path = dataset_dir / "videos/observation.images.front/chunk-000/file-000.mp4"
    episode_meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"v3")
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1]},
            "action": {"dtype": "float32", "shape": [1]},
            "observation.images.front": {"dtype": "video", "shape": [4, 4, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    if include_path_templates:
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "pick cube"}),
                json.dumps({"task_index": 1, "task": "place cube"}),
            ]
        )
        + "\n"
    )
    if write_tasks_parquet:
        pq.write_table(
            pa.table(
                {
                    "task_index": [0, 1],
                    "task": ["pick cube", "place cube"],
                }
            ),
            meta_dir / "tasks.parquet",
        )

    shared_data_path = data_dir / "file-000.parquet"
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0], [1.0], [20.0], [21.0], [22.0]],
                "action": [[10.0], [11.0], [30.0], [31.0], [32.0]],
                "timestamp": [0.0, 0.5, 1.0, 1.5, 2.0],
                "episode_index": [0, 0, 1, 1, 1],
                "task_index": [0, 0, 1, 1, 1],
            }
        ),
        shared_data_path,
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "tasks": [["pick cube"], ["place cube"]],
                "length": [2, 3],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 5],
                "videos/observation.images.front/chunk_index": [0, 0],
                "videos/observation.images.front/file_index": [0, 0],
                "videos/observation.images.front/from_timestamp": [0.0, 1.0],
                "videos/observation.images.front/to_timestamp": [1.0, 2.5],
            }
        ),
        episode_meta_dir / "file-000.parquet",
    )
    return shared_data_path
