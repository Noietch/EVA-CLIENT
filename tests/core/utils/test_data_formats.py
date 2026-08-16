from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.datasets import (
    CANONICAL_DATA_FORMATS,
    detect_data_format,
    normalize_data_format,
    open_dataset,
)


def test_normalize_data_format_aliases_and_invalid() -> None:
    assert CANONICAL_DATA_FORMATS == (
        "lerobot_v21",
        "lerobot_v3",
        "mcap",
        "hdf5",
    )
    assert normalize_data_format("LeRobot") == "lerobot_v21"
    assert normalize_data_format("lerobot-v2.1") == "lerobot_v21"
    assert normalize_data_format("LeRobot 3.0") == "lerobot_v3"
    assert normalize_data_format("H5") == "hdf5"
    assert normalize_data_format("mcap") == "mcap"

    with pytest.raises(ValueError, match="Unsupported data format"):
        normalize_data_format("lerobot4")


def test_detect_data_format_from_files_and_directories(tmp_path: Path) -> None:
    lerobot_v2 = tmp_path / "lerobot_v2"
    (lerobot_v2 / "meta").mkdir(parents=True)
    (lerobot_v2 / "meta" / "info.json").write_text('{"codebase_version":"v2.1"}', encoding="utf-8")
    assert detect_data_format(lerobot_v2) == "lerobot_v21"

    lerobot_v3 = tmp_path / "lerobot_v3"
    (lerobot_v3 / "meta").mkdir(parents=True)
    (lerobot_v3 / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}', encoding="utf-8")
    assert detect_data_format(lerobot_v3) == "lerobot_v3"

    lerobot_v2_by_episode = tmp_path / "lerobot_v2_by_episode" / "data" / "chunk-000"
    lerobot_v2_by_episode.mkdir(parents=True)
    (lerobot_v2_by_episode / "episode_000000.parquet").write_bytes(b"PAR1")
    assert detect_data_format(lerobot_v2_by_episode.parents[1]) == "lerobot_v21"

    lerobot_v3_by_files = tmp_path / "lerobot_v3_by_files" / "meta" / "episodes" / "chunk-000"
    lerobot_v3_by_files.mkdir(parents=True)
    (lerobot_v3_by_files / "file-000.parquet").write_bytes(b"PAR1")
    assert detect_data_format(lerobot_v3_by_files.parents[2]) == "lerobot_v3"

    hdf5_file = tmp_path / "offset_signature.hdf5"
    hdf5_file.write_bytes(b"\x00" * 1024 + b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)
    assert detect_data_format(hdf5_file) == "hdf5"

    hdf5_dir = tmp_path / "hdf5_dir"
    hdf5_dir.mkdir()
    (hdf5_dir / "episode_000001.h5").write_bytes(b"stub")
    assert detect_data_format(hdf5_dir) == "hdf5"

    mcap_file = tmp_path / "episode.mcap"
    mcap_magic = b"\x89MCAP0\r\n"
    mcap_file.write_bytes(mcap_magic + b"\x00" * 16 + mcap_magic)
    assert detect_data_format(mcap_file) == "mcap"

    mcap_dir = tmp_path / "mcap_dir"
    mcap_dir.mkdir()
    (mcap_dir / "episode_000001.mcap").write_bytes(b"stub")
    assert detect_data_format(mcap_dir) == "mcap"


def test_detect_data_format_prefers_declared_dataset_format(tmp_path: Path) -> None:
    declared = tmp_path / "declared_hdf5"
    (declared / "meta").mkdir(parents=True)
    (declared / "meta" / "info.json").write_text(
        '{"dataset_format":"hdf5","codebase_version":"custom"}',
        encoding="utf-8",
    )
    assert detect_data_format(declared) == "hdf5"


def test_hdf5_episode_roundtrip_preserves_keys_shapes_and_values(tmp_path: Path) -> None:
    root = tmp_path / "hdf5"
    columns = {
        "observations/state/qpos": np.arange(12, dtype=np.float32).reshape(3, 4),
        "action.scalar": np.int16(7),
    }
    images = {
        "observation.images/front_left": np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(
            2, 3, 4, 3
        ),
    }
    row = {
        "episode_index": 0,
        "length": 3,
        "tasks": ["pick the cup"],
        "task": "pick the cup",
    }

    open_dataset(root, "hdf5", fps=30).write(columns, row, images)
    path = root / "data/chunk-000/episode_000000.hdf5"

    assert path.is_file()
    assert detect_data_format(path) == "hdf5"

    episode = open_dataset(path).load(0)
    assert list(episode.columns) == list(columns)
    assert list(episode.images) == list(images)

    np.testing.assert_array_equal(
        episode.columns["observations/state/qpos"],
        columns["observations/state/qpos"],
    )
    assert episode.columns["action.scalar"] == columns["action.scalar"]
    np.testing.assert_array_equal(
        episode.images["observation.images/front_left"],
        images["observation.images/front_left"],
    )
    assert episode.task == row["task"]
    assert episode.fps == 30


def test_mcap_episode_roundtrip_preserves_streamed_images(tmp_path: Path) -> None:
    root = tmp_path / "mcap"
    columns = {
        "observations.state": np.arange(6, dtype=np.float32).reshape(2, 3),
        "action": np.array([1, 2, 3], dtype=np.int16),
    }
    images = {
        "observation.images.front": np.arange(2 * 2 * 3, dtype=np.uint8).reshape(1, 2, 2, 3),
    }
    row = {
        "episode_index": 0,
        "length": 2,
        "tasks": ["close drawer"],
        "task": "close drawer",
        "score": np.float32(0.75),
        "notes": np.array(["good", "fast"], dtype=object),
    }

    open_dataset(root, "mcap", fps=20).write(columns, row, images)
    path = root / "data/chunk-000/episode_000000.mcap"

    raw = path.read_bytes()
    magic = b"\x89MCAP0\r\n"
    assert raw[: len(magic)] == magic
    assert raw[-len(magic) :] == magic
    assert detect_data_format(path) == "mcap"

    episode = open_dataset(path).load(0)
    np.testing.assert_array_equal(
        episode.columns["observations.state"],
        columns["observations.state"],
    )
    np.testing.assert_array_equal(episode.columns["action"], columns["action"])
    np.testing.assert_array_equal(
        episode.images["observation.images.front"],
        images["observation.images.front"],
    )
    assert episode.task == row["task"]
