from __future__ import annotations

DATASET_VIDEO_GOP_SIZE = 30


def dataset_h264_ffmpeg_params(*, threads: int | None = 1) -> list[str]:
    """Return the fixed H.264 output parameters required by EVA datasets."""
    params = ["-preset", "ultrafast"]
    if threads is not None:
        params.extend(["-threads", str(threads)])
    params.extend(
        [
            "-g",
            str(DATASET_VIDEO_GOP_SIZE),
            "-keyint_min",
            str(DATASET_VIDEO_GOP_SIZE),
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
        ]
    )
    return params


__all__ = ["DATASET_VIDEO_GOP_SIZE", "dataset_h264_ffmpeg_params"]
