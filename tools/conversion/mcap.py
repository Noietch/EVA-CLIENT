from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from mcap.writer import CompressionType, Writer

from .source import LeRobotV21Source, video_frames, write_common_metadata


def export_mcap(
    source_dir: Path,
    output_dir: Path,
    progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> None:
    source = LeRobotV21Source(source_dir)
    for completed, episode in enumerate(source.episodes(), start=1):
        episode_index = int(episode.row["episode_index"])
        frame_count = episode.table.num_rows
        if frame_count == 0:
            raise ValueError(f"MCAP episode {episode_index} has no frames")
        path = (
            output_dir
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.mcap"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            writer = Writer(stream, compression=CompressionType.NONE)
            writer.start(profile="eva-client", library="eva-client")
            schema_id = writer.register_schema(
                name="eva_client.episode",
                encoding="jsonschema",
                data=json.dumps({"type": "object"}).encode(),
            )
            episode_channel = writer.register_channel(
                topic="episode",
                message_encoding="messagepack",
                schema_id=schema_id,
            )
            writer.add_message(
                channel_id=episode_channel,
                log_time=0,
                publish_time=0,
                data=_pack(
                    {
                        "columns": source.columns_without_images(episode.table),
                        "images": source.inline_images(episode.table),
                        "metadata": {**episode.row, "fps": source.fps},
                    }
                ),
            )
            _write_video_messages(
                writer,
                episode.videos,
                source.fps,
                episode_index=episode_index,
                expected_frames=frame_count,
            )
            writer.finish()
        if progress_callback is not None:
            progress_callback(completed, episode.row)
    write_common_metadata(source, output_dir, "mcap")


def _write_video_messages(
    writer: Writer,
    videos: dict[str, Path],
    fps: float,
    *,
    episode_index: int,
    expected_frames: int,
) -> None:
    frame_period_ns = max(1, int(round(1_000_000_000 / fps)))
    channels = {
        key: writer.register_channel(
            topic=f"episode/image/{key}",
            message_encoding="messagepack",
            schema_id=0,
        )
        for key in videos
    }
    frames_by_key = {key: iter(video_frames(path)) for key, path in videos.items()}
    observed_frames = {key: 0 for key in videos}

    # Emit one aligned timestamp per frame index across all cameras.
    for frame_index in range(expected_frames):
        timestamp = (frame_index + 1) * frame_period_ns
        for key in videos:
            frame = _next_frame(
                frames_by_key[key],
                episode_index=episode_index,
                key=key,
                observed_frames=observed_frames[key],
                expected_frames=expected_frames,
            )
            writer.add_message(
                channel_id=channels[key],
                log_time=timestamp,
                publish_time=timestamp,
                sequence=frame_index,
                data=_pack(np.asarray(frame)),
            )
            observed_frames[key] += 1

    # Validate there are no trailing frames beyond the shared sequence length.
    for key in videos:
        _ensure_frame_count(
            frames_by_key[key],
            episode_index=episode_index,
            key=key,
            observed_frames=observed_frames[key],
            expected_frames=expected_frames,
        )


def _next_frame(
    frames: Any,
    *,
    episode_index: int,
    key: str,
    observed_frames: int,
    expected_frames: int,
) -> np.ndarray:
    try:
        return np.asarray(next(frames))
    except StopIteration as error:
        raise ValueError(
            f"MCAP episode {episode_index} video {key!r} has {observed_frames} frames; "
            f"expected {expected_frames}"
        ) from error


def _ensure_frame_count(
    frames: Any,
    *,
    episode_index: int,
    key: str,
    observed_frames: int,
    expected_frames: int,
) -> None:
    try:
        next(frames)
    except StopIteration:
        return

    observed_frames += 1
    for _ in frames:
        observed_frames += 1
    raise ValueError(
        f"MCAP episode {episode_index} video {key!r} has {observed_frames} frames; "
        f"expected {expected_frames}"
    )


def _pack(value: Any) -> bytes:
    from openpi_client import msgpack_numpy

    encoded = msgpack_numpy.packb(value)
    if not isinstance(encoded, bytes):
        raise TypeError("msgpack encoder did not return bytes")
    return encoded


__all__ = ["export_mcap"]
