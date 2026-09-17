from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np
import pytest

from core.recorder.episode import EpisodeLogger
from tools.conversion.native import _transcode_dataset_video

pytestmark = pytest.mark.integration


def test_collection_video_output_has_keyframe_every_30_frames(tmp_path: Path) -> None:
    logger = object.__new__(EpisodeLogger)
    logger._log_dir = tmp_path
    logger._fps = 30
    frames = [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(65)]

    logger._write_videos(
        episode_index=0,
        videos={"observation.images.cam": frames},
        fps=30,
    )

    path = tmp_path / "videos/chunk-000/observation.images.cam/episode_000000.mp4"

    keyframes = []
    packet_index = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size <= 0:
                continue
            if packet.is_keyframe:
                keyframes.append(packet_index)
            packet_index += 1

    assert keyframes == [0, 30, 60]


def test_lerobot_v21_export_rewrites_video_to_gop_30(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    frames = [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(65)]
    writer = imageio.get_writer(
        str(source),
        fps=30,
        codec="libx264",
        macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    _transcode_dataset_video(source, output, fps=30, expected_frames=len(frames))

    keyframes = []
    packet_index = 0
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size <= 0:
                continue
            if packet.is_keyframe:
                keyframes.append(packet_index)
            packet_index += 1

    assert keyframes == [0, 30, 60]
