from __future__ import annotations

import json
import pickle
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

from tools.conversion import (
    DATASET_EXPORT_FORMATS,
    DatasetExportProgress,
    export_dataset_by_quality,
)


class _FakeVideoWriter:
    def __init__(self, path: str, **_: object) -> None:
        self.path = Path(path)
        self.frames: list[np.ndarray] = []

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame))

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.frames))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read_fake_video(path: Path) -> Iterator[np.ndarray]:
    for frame in pickle.loads(path.read_bytes()):
        yield np.asarray(frame)


@pytest.fixture(autouse=True)
def _patch_video_io(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.conversion.hdf5 as hdf5_module
    import tools.conversion.lerobot_v3 as lerobot_v3_module
    import tools.conversion.mcap as mcap_module

    monkeypatch.setattr(hdf5_module, "video_frames", _read_fake_video)
    monkeypatch.setattr(lerobot_v3_module, "video_frames", _read_fake_video)
    monkeypatch.setattr(mcap_module, "video_frames", _read_fake_video)
    monkeypatch.setattr(lerobot_v3_module.imageio, "get_writer", _FakeVideoWriter)


def _source_dataset(root: Path, *, rejected_indices: set[int] | None = None) -> None:
    if rejected_indices is None:
        rejected_indices = {1}
    else:
        rejected_indices = set(rejected_indices)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 6,
        "total_videos": 2,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.images.cam": {"dtype": "video", "shape": [8, 8, 3]},
        },
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
    (meta_dir / "tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n')
    rows = [
        {
            "episode_index": 0,
            "length": 3,
            "quality": "red" if 0 in rejected_indices else "green",
            "video_keys": ["observation.images.cam"],
        },
        {
            "episode_index": 1,
            "length": 3,
            "qc_verdict": "fail" if 1 in rejected_indices else "pass",
            "video_keys": ["observation.images.cam"],
        },
    ]
    _write_jsonl(meta_dir / "episodes.jsonl", rows)
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
        video_path = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.cam"
            / f"episode_{episode_index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [
            np.full((8, 8, 3), episode_index * 20 + frame_index, dtype=np.uint8)
            for frame_index in range(3)
        ]
        video_path.write_bytes(pickle.dumps(frames))


def test_dataset_export_formats_are_registered() -> None:
    assert DATASET_EXPORT_FORMATS == ("lerobot_v21", "lerobot_v3", "hdf5", "mcap")


@pytest.mark.parametrize("dataset_format", DATASET_EXPORT_FORMATS)
def test_quality_export_converts_each_supported_format(
    tmp_path: Path,
    dataset_format: str,
) -> None:
    source = tmp_path / "source"
    accepted = tmp_path / dataset_format / "accepted"
    rejected = tmp_path / dataset_format / "rejected"
    _source_dataset(source)

    progress = []
    summary = export_dataset_by_quality(
        source,
        accepted,
        rejected,
        dataset_format=dataset_format,
        progress_callback=progress.append,
    )

    assert summary.source_dir == str(source.resolve())
    assert summary.accepted_dir == str(accepted.resolve())
    assert summary.rejected_dir == str(rejected.resolve())
    assert summary.dataset_format == dataset_format
    assert summary.source_episodes == 2
    assert summary.accepted_episodes == 1
    assert summary.rejected_episodes == 1
    assert summary.accepted_frames == 3
    assert summary.rejected_frames == 3
    assert summary.rejected_source_indices == (1,)
    assert progress[0] == DatasetExportProgress(0, 2, "", None)
    assert all(isinstance(item, DatasetExportProgress) for item in progress)
    assert progress[-1].episodes_completed == 2
    for subset, label, source_index in (
        (accepted, "accepted", 0),
        (rejected, "rejected", 1),
    ):
        marker = json.loads((subset / "meta" / "quality_split.json").read_text())
        assert marker["source_dir"] == str(source.resolve())
        assert marker["subset"] == label
        assert marker["dataset_format"] == dataset_format
        assert marker["source_episode_indices"] == [source_index]

    if dataset_format == "lerobot_v21":
        assert (accepted / "data/chunk-000/episode_000000.parquet").is_file()
    elif dataset_format == "lerobot_v3":
        assert pq.read_table(accepted / "data/chunk-000/file-000.parquet").num_rows == 3
        episode_table = pq.read_table(
            accepted / "meta/episodes/chunk-000/file-000.parquet"
        )
        assert episode_table["videos/observation.images.cam/from_timestamp"][0].as_py() == 0
        assert episode_table["videos/observation.images.cam/to_timestamp"][0].as_py() == 0.15
    elif dataset_format == "hdf5":
        with h5py.File(accepted / "data/chunk-000/episode_000000.hdf5", "r") as handle:
            assert len(handle["columns"]) > 0
            assert len(handle["images"]) == 1
    else:
        payload = (accepted / "data/chunk-000/episode_000000.mcap").read_bytes()
        assert payload.startswith(b"\x89MCAP0\r\n")
        assert payload.endswith(b"\x89MCAP0\r\n")
        if make_reader is not None:
            with (accepted / "data/chunk-000/episode_000000.mcap").open("rb") as stream:
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


def test_quality_export_rejects_unknown_format_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported dataset format"):
        export_dataset_by_quality(
            tmp_path / "source",
            tmp_path / "accepted",
            tmp_path / "rejected",
            dataset_format="unknown",
        )


@pytest.mark.parametrize(
    ("dataset_format", "format_name"),
    [("lerobot_v3", "LeRobot v3"), ("hdf5", "HDF5"), ("mcap", "MCAP")],
)
def test_quality_export_rejects_misaligned_video_frames(
    tmp_path: Path,
    dataset_format: str,
    format_name: str,
) -> None:
    source = tmp_path / "source"
    accepted = tmp_path / "accepted"
    rejected = tmp_path / "rejected"
    _source_dataset(source)
    video = source / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    video.write_bytes(pickle.dumps([np.zeros((8, 8, 3), dtype=np.uint8)] * 2))

    with pytest.raises(
        ValueError,
        match=rf"{format_name} episode 0 .* has 2 frames; expected 3",
    ):
        export_dataset_by_quality(
            source,
            accepted,
            rejected,
            dataset_format=dataset_format,
        )

    assert not accepted.exists()
    assert not rejected.exists()


def test_quality_export_supports_empty_rejected_subset_across_parents(tmp_path: Path) -> None:
    source = tmp_path / "source"
    accepted = tmp_path / "accepted-root" / "accepted"
    rejected = tmp_path / "rejected-root" / "rejected"
    _source_dataset(source, rejected_indices=set())

    summary = export_dataset_by_quality(
        source,
        accepted,
        rejected,
        dataset_format="lerobot_v3",
    )

    assert summary.accepted_episodes == 2
    assert summary.rejected_episodes == 0
    assert summary.accepted_dir == str(accepted.resolve())
    assert summary.rejected_dir == str(rejected.resolve())
    assert pq.read_table(accepted / "data/chunk-000/file-000.parquet").num_rows == 6
    assert json.loads((rejected / "meta" / "info.json").read_text())["total_episodes"] == 0
    marker = json.loads((rejected / "meta" / "quality_split.json").read_text())
    assert marker["subset"] == "rejected"
    assert not (rejected / "data").exists()
