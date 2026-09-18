from __future__ import annotations

import shutil
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np

from core.recorder.video_encoding import DATASET_VIDEO_GOP_SIZE, dataset_h264_ffmpeg_params

from .source import video_frames


def _transcode_dataset_video(
    source: Path,
    target: Path,
    *,
    fps: float,
    expected_frames: int,
) -> None:
    """Rewrite one LeRobot v2.1 video with the required dataset GOP."""
    if _dataset_video_is_compatible(source, fps=fps, expected_frames=expected_frames):
        _link_or_copy_video(source, target)
        return
    writer = imageio.get_writer(
        str(target),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=dataset_h264_ffmpeg_params(threads=None),
    )
    observed_frames = 0
    try:
        for frame in video_frames(source):
            writer.append_data(np.ascontiguousarray(frame))
            observed_frames += 1
    finally:
        writer.close()
    if observed_frames != expected_frames:
        raise ValueError(
            f"LeRobot v2.1 video {source} has {observed_frames} frames; expected {expected_frames}"
        )


def _dataset_video_is_compatible(
    source: Path,
    *,
    fps: float,
    expected_frames: int,
) -> bool:
    """Check frame count, rate, codec, and GOP from packets without decoding pixels."""
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            if stream.codec_context.name != "h264":
                return False
            rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
            if rate <= 0.0 or abs(rate - fps) > max(0.01, fps * 0.001):
                return False
            keyframes: list[int] = []
            packet_count = 0
            for packet in container.demux(stream):
                if packet.size <= 0:
                    continue
                if packet.is_keyframe:
                    keyframes.append(packet_count)
                packet_count += 1
    except (OSError, ValueError, av.error.FFmpegError):
        return False
    return packet_count == expected_frames and keyframes == list(
        range(0, expected_frames, DATASET_VIDEO_GOP_SIZE)
    )


def _link_or_copy_video(source: Path, target: Path) -> None:
    """Reuse source video bytes for an intermediate conversion dataset."""
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)
