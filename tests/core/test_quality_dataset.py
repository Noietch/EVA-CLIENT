from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.utils.quality_dataset import is_rejected_episode, split_dataset_by_quality


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "total_episodes": 3,
        "total_frames": 9,
        "total_videos": 3,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1]},
            "observation.images.cam": {"dtype": "video", "shape": [2, 2, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": {"min": [0], "max": [22], "mean": [11], "std": [1]}})
    )
    (root / "meta" / "tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n')
    episodes = [
        {
            "episode_index": 0,
            "length": 3,
            "quality": "green",
            "video_keys": ["observation.images.cam"],
        },
        {
            "episode_index": 1,
            "length": 3,
            "quality": "red",
            "video_keys": ["observation.images.cam"],
        },
        {
            "episode_index": 2,
            "length": 3,
            "quality": "green",
            "qc_verdict": "fail",
            "video_keys": ["observation.images.cam"],
        },
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    stats_rows = []
    for episode in range(3):
        values = np.arange(3, dtype=np.float32) + episode * 10
        table = pa.table(
            {
                "observation.state": pa.array([[float(value)] for value in values]),
                "frame_index": pa.array(np.arange(3, dtype=np.int64)),
                "episode_index": pa.array(np.full(3, episode, dtype=np.int64)),
                "index": pa.array(np.arange(episode * 3, episode * 3 + 3, dtype=np.int64)),
                "task_index": pa.array(np.zeros(3, dtype=np.int64)),
            }
        )
        parquet = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet)
        video = (
            root / "videos" / "chunk-000" / "observation.images.cam" / f"episode_{episode:06d}.mp4"
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{episode}".encode())
        stats_rows.append(
            {
                "episode_index": episode,
                "stats": {
                    "observation.state": {
                        "min": [float(values.min())],
                        "max": [float(values.max())],
                        "mean": [float(values.mean())],
                        "std": [float(values.std())],
                        "count": [3],
                    }
                },
            }
        )
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats_rows)


def test_web_red_rule_includes_automatic_and_manual_failures() -> None:
    assert not is_rejected_episode({"quality": "green"})
    assert is_rejected_episode({"quality": "red"})
    assert is_rejected_episode({"quality": "green", "qc_verdict": "fail"})
    assert is_rejected_episode({"quality": "red", "qc_verdict": "pass"})


def test_split_dataset_exports_contiguous_accepted_and_rejected_subsets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    accepted = tmp_path / "accepted"
    rejected = tmp_path / "rejected"
    _dataset(source)

    progress = []
    summary = split_dataset_by_quality(
        source, accepted, rejected, progress_callback=progress.append
    )

    assert summary.accepted_episodes == 1
    assert summary.rejected_episodes == 2
    assert summary.rejected_source_indices == (1, 2)
    assert (source / "data/chunk-000/episode_000002.parquet").exists()

    accepted_info = json.loads((accepted / "meta/info.json").read_text())
    rejected_info = json.loads((rejected / "meta/info.json").read_text())
    assert accepted_info["total_episodes"] == 1
    assert accepted_info["total_frames"] == 3
    assert rejected_info["total_episodes"] == 2
    assert rejected_info["total_frames"] == 6
    assert len(list(rejected.glob("videos/**/*.mp4"))) == 2

    first_rejected = pq.read_table(rejected / "data/chunk-000/episode_000000.parquet")
    second_rejected = pq.read_table(rejected / "data/chunk-000/episode_000001.parquet")
    assert first_rejected["episode_index"].to_pylist() == [0, 0, 0]
    assert second_rejected["episode_index"].to_pylist() == [1, 1, 1]
    assert first_rejected["index"].to_pylist() == [0, 1, 2]
    assert second_rejected["index"].to_pylist() == [3, 4, 5]
    rejected_stats = json.loads((rejected / "meta/stats.json").read_text())
    assert rejected_stats["observation.state"]["min"] == [10.0]
    assert rejected_stats["observation.state"]["max"] == [22.0]
    assert [item.episodes_completed for item in progress] == [0, 1, 2, 3]
    assert all(item.episodes_total == 3 for item in progress)
    assert [item.subset for item in progress] == ["", "accepted", "rejected", "rejected"]
    assert [item.source_episode_index for item in progress] == [None, 0, 1, 2]


def test_split_dataset_allows_an_empty_rejected_subset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    accepted = tmp_path / "accepted"
    rejected = tmp_path / "rejected"
    _dataset(source)
    rows = [
        {
            "episode_index": index,
            "length": 3,
            "quality": "green",
            "video_keys": ["observation.images.cam"],
        }
        for index in range(3)
    ]
    _write_jsonl(source / "meta" / "episodes.jsonl", rows)

    summary = split_dataset_by_quality(source, accepted, rejected)

    assert summary.accepted_episodes == 3
    assert summary.rejected_episodes == 0
    assert json.loads((rejected / "meta/info.json").read_text())["total_episodes"] == 0
    assert (rejected / "meta/episodes.jsonl").read_text() == ""
    assert json.loads((rejected / "meta/stats.json").read_text()) == {}


def test_split_dataset_replaces_the_same_export_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export_root = tmp_path / "source_export"
    accepted = export_root / "accepted"
    rejected = export_root / "rejected"
    _dataset(source)

    split_dataset_by_quality(source, accepted, rejected)
    (accepted / "stale.txt").write_text("old export")

    summary = split_dataset_by_quality(
        source,
        accepted,
        rejected,
        replace_existing=True,
    )

    assert summary.accepted_dir == str(accepted.resolve())
    assert not (accepted / "stale.txt").exists()
    assert sorted(path.name for path in export_root.iterdir()) == ["accepted", "rejected"]
