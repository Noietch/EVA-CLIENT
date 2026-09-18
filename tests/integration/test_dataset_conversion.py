from __future__ import annotations

import json
import pickle
import shutil
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    from mcap.reader import make_reader
except ModuleNotFoundError:
    make_reader = None

if "openpi_client" not in sys.modules:

    class _FakeMsgpackPacker:
        def pack(self, value: object) -> bytes:
            return pickle.dumps(value)

    module = types.ModuleType("openpi_client")
    module.msgpack_numpy = types.SimpleNamespace(
        Packer=_FakeMsgpackPacker,
        packb=pickle.dumps,
        unpackb=pickle.loads,
    )
    sys.modules["openpi_client"] = module

from openpi_client import msgpack_numpy

if make_reader is None and "mcap" not in sys.modules:
    mcap_module = types.ModuleType("mcap")
    writer_module = types.ModuleType("mcap.writer")

    class _CompressionType:
        NONE = "none"

    class _Writer:
        def __init__(self, stream, compression=None) -> None:
            self.stream = stream
            self._next_id = 1

        def start(self, **_: object) -> None:
            self.stream.write(b"\x89MCAP0\r\n")

        def register_schema(self, **_: object) -> int:
            schema_id = self._next_id
            self._next_id += 1
            return schema_id

        def register_channel(self, **_: object) -> int:
            channel_id = self._next_id
            self._next_id += 1
            return channel_id

        def add_message(self, *, data: bytes, **_: object) -> None:
            self.stream.write(data[:1] or b"\0")

        def finish(self) -> None:
            self.stream.write(b"\x89MCAP0\r\n")

    writer_module.CompressionType = _CompressionType
    writer_module.Writer = _Writer
    mcap_module.writer = writer_module
    sys.modules["mcap"] = mcap_module
    sys.modules["mcap.writer"] = writer_module

from tools.conversion import export_dataset

pytestmark = pytest.mark.integration


class _FakeVideoWriter:
    def __init__(self, path: str, **_: object) -> None:
        self.path = Path(path)
        self.frames: list[np.ndarray] = []

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame))

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.frames))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read_fake_video(path: Path) -> Iterator[np.ndarray]:
    for frame in pickle.loads(path.read_bytes()):
        yield np.asarray(frame)


@pytest.fixture(autouse=True)
def _patch_video_io(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.conversion.hdf5 as hdf5_module
    import tools.conversion.lerobot_v3 as lerobot_v3_module
    import tools.conversion.mcap as mcap_module
    import tools.conversion.native as native_module

    monkeypatch.setattr(hdf5_module, "video_frames", _read_fake_video)
    monkeypatch.setattr(lerobot_v3_module, "video_frames", _read_fake_video)
    monkeypatch.setattr(mcap_module, "video_frames", _read_fake_video)
    monkeypatch.setattr(lerobot_v3_module.imageio, "get_writer", _FakeVideoWriter)

    def copy_video(source: Path, target: Path, **_: object) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    monkeypatch.setattr(native_module, "_transcode_dataset_video", copy_video)


def _source_dataset(
    root: Path,
    *,
    video_keys: tuple[str, ...] = ("observation.images.cam",),
) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    features: dict[str, dict[str, object]] = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    }
    for key in video_keys:
        features[key] = {"dtype": "video", "shape": [8, 8, 3]}

    # Build a minimal v2.1 source dataset that conversion can split and rewrite.
    info = {
        "codebase_version": "v2.1",
        "collection_started_at": "2026-08-31T09:30:00+08:00",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 6,
        "total_videos": 2 * len(video_keys),
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [0, 1],
                    "max": [12, 13],
                    "mean": [6, 7],
                    "std": [1, 1],
                }
            }
        )
    )
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "test"}])
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": 0, "length": 3, "video_keys": list(video_keys)},
            {"episode_index": 1, "length": 3, "video_keys": list(video_keys)},
        ],
    )
    _write_jsonl(meta_dir / "episodes_stats.jsonl", [])

    for episode_index in range(2):
        values = np.arange(6, dtype=np.float32).reshape(3, 2) + episode_index * 8
        table = pa.table(
            {
                "observation.state": values.tolist(),
                "action": (values + 1).tolist(),
                "frame_index": np.arange(3, dtype=np.int64),
                "episode_index": np.full(3, episode_index, dtype=np.int64),
                "index": np.arange(episode_index * 3, episode_index * 3 + 3),
                "task_index": np.zeros(3, dtype=np.int64),
            }
        )
        data_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)
        for key_index, key in enumerate(video_keys):
            video_path = root / "videos" / "chunk-000" / key / f"episode_{episode_index:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            frames = [
                np.full(
                    (8, 8, 3),
                    episode_index * 40 + key_index * 20 + frame_index,
                    dtype=np.uint8,
                )
                for frame_index in range(3)
            ]
            video_path.write_bytes(pickle.dumps(frames))


@pytest.mark.parametrize("dataset_format", ["lerobot_v3", "hdf5", "mcap"])
def test_export_converts_each_supported_format(tmp_path: Path, dataset_format: str) -> None:
    source = tmp_path / "source"
    _source_dataset(source)
    output = source.with_name(f"{source.name}_{dataset_format}")

    progress: list[object] = []
    summary = export_dataset(
        source,
        output,
        dataset_format=dataset_format,
        progress_callback=progress.append,
    )

    assert summary.source_dir == str(source.resolve())
    assert summary.output_dir == str(output.resolve())
    assert summary.dataset_format == dataset_format
    assert summary.episodes == 2
    assert progress[-1].episodes_completed == progress[-1].episodes_total == 2
    episodes = [
        json.loads(line)
        for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(episodes) == 2

    if dataset_format == "lerobot_v3":
        assert (output / "data/chunk-000/file-000.parquet").is_file()
        episode_table = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
        assert episode_table["videos/observation.images.cam/from_timestamp"][0].as_py() == 0.0
        assert episode_table["videos/observation.images.cam/to_timestamp"][0].as_py() == 0.15
    elif dataset_format == "hdf5":
        with h5py.File(output / "data/chunk-000/episode_000000.hdf5", "r") as handle:
            assert len(handle["columns"]) > 0
            assert len(handle["images"]) == 1
    else:
        payload = (output / "data/chunk-000/episode_000000.mcap").read_bytes()
        assert payload.startswith(b"\x89MCAP0\r\n")
        assert payload.endswith(b"\x89MCAP0\r\n")
        if make_reader is not None:
            with (output / "data/chunk-000/episode_000000.mcap").open("rb") as stream:
                records = list(make_reader(stream).iter_messages())

            topics = [channel.topic for _, channel, _ in records]
            timestamps = [message.log_time for _, _, message in records]
            metadata = msgpack_numpy.unpackb(records[0][2].data)
            frame = msgpack_numpy.unpackb(records[1][2].data)

            assert topics[0] == "episode"
            assert topics[1:] == ["episode/image/observation.images.cam"] * 3
            assert timestamps == sorted(timestamps)
            assert len(timestamps) == len(set(timestamps))
            assert metadata["metadata"]["episode_index"] == 0
            assert set(metadata) == {"columns", "images", "metadata"}
            assert frame.shape == (8, 8, 3)
            assert int(frame[0, 0, 0]) == 0


def test_export_refuses_to_overwrite_an_existing_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_dataset(source)
    output = source.with_name(f"{source.name}_hdf5")
    export_dataset(source, output, dataset_format="hdf5")

    with pytest.raises(FileExistsError, match="output directory already exists"):
        export_dataset(source, output, dataset_format="hdf5")


def test_export_rejects_misaligned_video_frames(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = source.with_name(f"{source.name}_hdf5")
    _source_dataset(source)
    video = source / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    video.write_bytes(pickle.dumps([np.zeros((8, 8, 3), dtype=np.uint8)] * 2))

    with pytest.raises(ValueError, match=r"HDF5 episode 0 .* has 2 frames; expected 3"):
        export_dataset(source, output, dataset_format="hdf5")

    assert not output.exists()


def test_export_failure_keeps_the_previous_copy(tmp_path: Path, monkeypatch) -> None:
    import tools.conversion.hdf5 as hdf5_module

    source = tmp_path / "source"
    output = source.with_name(f"{source.name}_hdf5")
    _source_dataset(source)
    export_dataset(source, output, dataset_format="hdf5")
    before = sorted(path.name for path in output.rglob("*"))

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("conversion interrupted")

    monkeypatch.setattr(hdf5_module, "export_hdf5", explode)
    with pytest.raises(OSError, match="conversion interrupted"):
        export_dataset(source, output, dataset_format="hdf5", replace_existing=True)

    assert sorted(path.name for path in output.rglob("*")) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source", "source_hdf5"]


def test_mcap_interleaves_multi_camera_frames_by_frame_index(tmp_path: Path) -> None:
    if make_reader is None:
        pytest.skip("real mcap reader is unavailable")

    source = tmp_path / "source"
    output = source.with_name(f"{source.name}_mcap")
    video_keys = ("observation.images.left", "observation.images.right")
    _source_dataset(source, video_keys=video_keys)

    export_dataset(source, output, dataset_format="mcap")

    with (output / "data/chunk-000/episode_000000.mcap").open("rb") as stream:
        records = list(make_reader(stream).iter_messages())

    image_records = records[1:]
    topics = [channel.topic for _, channel, _ in image_records]
    sequences = [message.sequence for _, _, message in image_records]
    timestamps = [message.log_time for _, _, message in image_records]
    frames = [msgpack_numpy.unpackb(message.data) for _, _, message in image_records]
    expected_topics = [f"episode/image/{key}" for _ in range(3) for key in video_keys]

    assert topics == expected_topics
    assert sequences == [0, 0, 1, 1, 2, 2]
    assert timestamps == [
        50_000_000,
        50_000_000,
        100_000_000,
        100_000_000,
        150_000_000,
        150_000_000,
    ]
    assert int(frames[0][0, 0, 0]) == 0
    assert int(frames[1][0, 0, 0]) == 20
    assert int(frames[2][0, 0, 0]) == 1
    assert int(frames[3][0, 0, 0]) == 21
