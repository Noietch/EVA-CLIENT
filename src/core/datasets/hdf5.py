from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .base import (
    BaseDataset,
    EpisodeData,
    build_episode_metadata,
    coerce_episode_data,
    pack_value,
    unpack_value,
)


class HDF5Dataset(BaseDataset):
    format = "hdf5"
    _suffixes = (".h5", ".hdf5")

    def count_episodes(self) -> int:
        return len(self.episode_paths(self._suffixes))

    def episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fallback_index, path in enumerate(self.episode_paths(self._suffixes)):
            row = _metadata_row(path)
            episode_index = (
                int(path.stem.removeprefix("episode_"))
                if path.stem.startswith("episode_")
                else fallback_index
            )
            row.setdefault("episode_index", episode_index)
            rows.append(row)
        return rows

    def load(self, episode_index: int) -> EpisodeData:
        with h5py.File(self.episode_path(episode_index, self._suffixes), "r") as handle:
            return coerce_episode_data(
                {
                    "columns": _read_mapping(handle["columns"]),
                    "images": _read_mapping(handle["images"]),
                    "metadata": _read_mapping(handle["metadata"]),
                }
            )

    def load_columns(self, episode_index: int) -> dict[str, Any]:
        with h5py.File(self.episode_path(episode_index, self._suffixes), "r") as handle:
            return _read_mapping(handle["columns"])

    def load_image_frame(
        self,
        episode_index: int,
        image_key: str,
        frame_index: int,
    ) -> np.ndarray | None:
        if frame_index < 0:
            return None
        with h5py.File(self.episode_path(episode_index, self._suffixes), "r") as handle:
            item = _mapping_item(handle["images"], image_key)
            if item is None:
                return None
            dataset = item["value"]
            if dataset.ndim == 0 or frame_index >= dataset.shape[0]:
                return None
            return np.asarray(dataset[frame_index])

    def write(
        self,
        columns: dict[str, Any],
        row: dict[str, Any],
        videos: Mapping[str, Iterable[np.ndarray]],
    ) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        path = (
            self.path
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.hdf5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f"{path.name}.incomplete")
        with h5py.File(pending, "w") as handle:
            _write_mapping(handle.create_group("columns"), columns)
            _write_images(handle.create_group("images"), videos)
            _write_mapping(
                handle.create_group("metadata"),
                build_episode_metadata(row, self.require_fps()),
                preserve_containers=True,
            )
        pending.replace(path)
        return dict(row)


def _mapping_item(group: Any, key: str) -> Any | None:
    for name in sorted(group):
        item = group[name]
        if str(item.attrs["key"]) == key:
            return item
    return None


def _metadata_row(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        row = {
            key: _metadata_value(value)
            for key, value in _read_mapping(handle["metadata"]).items()
        }
        row.setdefault("length", _mapping_length(handle["columns"]))
    tasks = row.get("tasks") or []
    if tasks:
        row["tasks"] = [str(task) for task in tasks]
    elif row.get("task"):
        row["tasks"] = [str(row["task"])]
    else:
        row["tasks"] = []
    return row


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return value


def _mapping_length(group: Any) -> int:
    for name in sorted(group):
        value = group[name]["value"]
        if value.ndim == 0:
            continue
        return int(value.shape[0])
    return 0


def _read_mapping(group: Any) -> dict[str, Any]:
    return {str(group[name].attrs["key"]): _read_value(group[name]) for name in sorted(group)}


def _read_value(item: Any) -> Any:
    kind = str(item.attrs["kind"])
    if kind == "none":
        return None
    dataset = item["value"]
    if kind == "bytes":
        return dataset[()].tobytes()
    if kind == "string":
        return dataset.asstr()[()]
    if kind == "msgpack":
        return unpack_value(dataset[()].tobytes())
    value = dataset[()]
    return value.item() if isinstance(value, np.generic) else value


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
        item.attrs["kind"] = "msgpack"
        item.create_dataset("value", data=np.frombuffer(pack_value(value), dtype=np.uint8))
        return
    array = np.asarray(value)
    if array.dtype.kind not in {"O", "U"}:
        item.attrs["kind"] = "array"
        item.create_dataset("value", data=array)
        return
    item.attrs["kind"] = "msgpack"
    item.create_dataset("value", data=np.frombuffer(pack_value(value), dtype=np.uint8))


def _write_images(group: Any, images: Mapping[str, Iterable[np.ndarray]]) -> None:
    for index, (key, frames) in enumerate(images.items()):
        item = group.create_group(f"item_{index:06d}")
        item.attrs.update(key=key, kind="array")
        dataset = None
        for frame in frames:
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
                raise ValueError(f"HDF5 image {key!r} changed shape to {array.shape}")
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = array
        if dataset is None:
            item.create_dataset("value", data=np.empty((0,), dtype=np.uint8))


__all__ = ["HDF5Dataset"]
