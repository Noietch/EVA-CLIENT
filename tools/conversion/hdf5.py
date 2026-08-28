from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from openpi_client import msgpack_numpy

from .source import LeRobotV21Source, video_frames, write_common_metadata


def export_hdf5(
    source_dir: Path,
    output_dir: Path,
    progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> None:
    source = LeRobotV21Source(source_dir)
    for completed, episode in enumerate(source.episodes(), start=1):
        episode_index = int(episode.row["episode_index"])
        frame_count = episode.table.num_rows
        if frame_count == 0:
            raise ValueError(f"HDF5 episode {episode_index} has no frames")
        path = (
            output_dir
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.hdf5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            _write_mapping(
                handle.create_group("columns"),
                source.columns_without_images(episode.table),
            )
            images: dict[str, Iterable[np.ndarray]] = dict(source.inline_images(episode.table))
            images.update({key: video_frames(video) for key, video in episode.videos.items()})
            _write_images(
                handle.create_group("images"),
                images,
                episode_index=episode_index,
                expected_frames=frame_count,
            )
            metadata = {**episode.row, "fps": source.fps}
            _write_mapping(handle.create_group("metadata"), metadata, preserve_containers=True)
        if progress_callback is not None:
            progress_callback(completed, episode.row)
    write_common_metadata(source, output_dir, "hdf5")


def _write_mapping(
    group: Any,
    values: Mapping[str, Any],
    *,
    preserve_containers: bool = False,
) -> None:
    for index, (key, value) in enumerate(values.items()):
        item = group.create_group(f"item_{index:06d}")
        item.attrs["key"] = key
        _write_value(item, value, preserve_containers=preserve_containers)


def _write_value(item: Any, value: Any, *, preserve_containers: bool) -> None:
    if value is None:
        item.attrs["kind"] = "none"
        item.create_dataset("value", data=np.empty((0,), dtype=np.uint8))
        return
    if isinstance(value, str):
        item.attrs["kind"] = "string"
        item.create_dataset("value", data=value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    if isinstance(value, bytes):
        item.attrs["kind"] = "bytes"
        item.create_dataset("value", data=np.frombuffer(value, dtype=np.uint8))
        return
    if preserve_containers and isinstance(value, (Mapping, list, tuple)):
        _write_msgpack(item, value)
        return
    array = np.asarray(value)
    if array.dtype.kind not in {"O", "U"}:
        item.attrs["kind"] = "array"
        item.create_dataset("value", data=array)
        return
    _write_msgpack(item, value)


def _write_msgpack(item: Any, value: Any) -> None:
    encoded = msgpack_numpy.packb(value)
    if not isinstance(encoded, bytes):
        raise TypeError("msgpack encoder did not return bytes")
    item.attrs["kind"] = "msgpack"
    item.create_dataset("value", data=np.frombuffer(encoded, dtype=np.uint8))


def _write_images(
    group: Any,
    images: Mapping[str, Iterable[np.ndarray]],
    *,
    episode_index: int,
    expected_frames: int,
) -> None:
    for index, (key, frames) in enumerate(images.items()):
        item = group.create_group(f"item_{index:06d}")
        item.attrs.update(key=key, kind="array")
        dataset = None
        observed_frames = 0
        for frame in frames:
            observed_frames += 1
            array = np.asarray(frame)
            if dataset is None:
                dataset = item.create_dataset(
                    "value",
                    shape=(0, *array.shape),
                    maxshape=(None, *array.shape),
                    chunks=(1, *array.shape),
                    dtype=array.dtype,
                )
            if array.shape != dataset.shape[1:]:
                raise ValueError(
                    f"HDF5 episode {episode_index} image {key!r} changed shape to {array.shape}"
                )
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = array
        if dataset is None:
            item.create_dataset("value", data=np.empty((0,), dtype=np.uint8))
        if observed_frames != expected_frames:
            raise ValueError(
                f"HDF5 episode {episode_index} image {key!r} has {observed_frames} frames; "
                f"expected {expected_frames}"
            )


__all__ = ["export_hdf5"]
