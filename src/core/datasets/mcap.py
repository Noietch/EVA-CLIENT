from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from .base import (
    BaseDataset,
    EpisodeData,
    build_episode_metadata,
    coerce_episode_data,
    pack_value,
    unpack_value,
)


class MCAPDataset(BaseDataset):
    format = "mcap"
    _suffixes = (".mcap",)

    def count_episodes(self) -> int:
        return len(self.episode_paths(self._suffixes))

    def episode_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fallback_index, path in enumerate(self.episode_paths(self._suffixes)):
            row = _episode_row(path)
            episode_index = (
                int(path.stem.removeprefix("episode_"))
                if path.stem.startswith("episode_")
                else fallback_index
            )
            row.setdefault("episode_index", episode_index)
            rows.append(row)
        return rows

    def load(self, episode_index: int) -> EpisodeData:
        path = self.episode_path(episode_index, self._suffixes)
        payload: dict[str, Any] | None = None
        frames: dict[str, list[np.ndarray]] = {}
        with path.open("rb") as stream:
            for _schema, channel, message in make_reader(stream).iter_messages():
                if channel.topic == "episode":
                    value = unpack_value(message.data)
                    if not isinstance(value, dict):
                        raise ValueError(f"Invalid episode payload in {path}")
                    payload = value
                elif channel.topic.startswith("episode/image/"):
                    key = channel.topic.removeprefix("episode/image/")
                    frames.setdefault(key, []).append(np.asarray(unpack_value(message.data)))
        if payload is None:
            raise ValueError(f"No episode message found in MCAP file: {path}")
        payload["images"] = {
            **dict(payload.get("images", {})),
            **{key: np.stack(values) for key, values in frames.items()},
        }
        return coerce_episode_data(payload)

    def load_columns(self, episode_index: int) -> dict[str, Any]:
        path = self.episode_path(episode_index, self._suffixes)
        with path.open("rb") as stream:
            for _schema, channel, message in make_reader(stream).iter_messages(topics=["episode"]):
                if channel.topic == "episode":
                    payload = unpack_value(message.data)
                    if isinstance(payload, dict):
                        return coerce_episode_data(payload).columns
                    break
        raise ValueError(f"No valid episode message found in MCAP file: {path}")

    def load_image_frame(
        self,
        episode_index: int,
        image_key: str,
        frame_index: int,
    ) -> np.ndarray | None:
        if frame_index < 0:
            return None
        path = self.episode_path(episode_index, self._suffixes)
        target_topic = f"episode/image/{image_key}"
        with path.open("rb") as stream:
            seen = 0
            for _schema, channel, message in make_reader(stream).iter_messages(
                topics=[target_topic]
            ):
                if channel.topic != target_topic:
                    continue
                if seen == frame_index:
                    return np.asarray(unpack_value(message.data))
                seen += 1
        return None

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
            / f"episode_{episode_index:06d}.mcap"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f"{path.name}.incomplete")
        with pending.open("wb") as stream:
            writer = Writer(stream, compression=CompressionType.NONE)
            writer.start(profile="eva-client", library="eva-client")
            schema = writer.register_schema(
                name="eva_client.episode",
                encoding="jsonschema",
                data=json.dumps({"type": "object"}).encode(),
            )
            channel = writer.register_channel(
                topic="episode",
                message_encoding="messagepack",
                schema_id=schema,
            )
            writer.add_message(
                channel_id=channel,
                log_time=0,
                publish_time=0,
                data=pack_value(
                    {
                        "columns": columns,
                        "images": {},
                        "metadata": build_episode_metadata(row, self.require_fps()),
                    }
                ),
            )
            for key, values in videos.items():
                image_channel = writer.register_channel(
                    topic=f"episode/image/{key}",
                    message_encoding="messagepack",
                    schema_id=0,
                )
                previous_timestamp = -1
                for frame_index, frame in enumerate(values):
                    timestamp = _frame_timestamp_ns(frame_index, self.require_fps())
                    assert timestamp > previous_timestamp, (
                        f"MCAP frame timestamps must strictly increase for {key!r}: "
                        f"{previous_timestamp} -> {timestamp}"
                    )
                    writer.add_message(
                        channel_id=image_channel,
                        log_time=timestamp,
                        publish_time=timestamp,
                        sequence=frame_index,
                        data=pack_value(np.asarray(frame)),
                    )
                    previous_timestamp = timestamp
            writer.finish()
        pending.replace(path)
        return dict(row)


def _episode_row(path: Any) -> dict[str, Any]:
    with path.open("rb") as stream:
        for _schema, channel, message in make_reader(stream).iter_messages(topics=["episode"]):
            if channel.topic != "episode":
                continue
            payload = unpack_value(message.data)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid episode payload in {path}")
            metadata = dict(payload.get("metadata", {}))
            tasks = metadata.get("tasks") or []
            if tasks:
                metadata["tasks"] = [str(task) for task in tasks]
            elif metadata.get("task"):
                metadata["tasks"] = [str(metadata["task"])]
            else:
                metadata["tasks"] = []
            columns = payload.get("columns", {})
            if "length" not in metadata:
                first_column = next(iter(columns.values()), ())
                metadata["length"] = len(first_column)
            return metadata
    raise ValueError(f"No episode message found in MCAP file: {path}")


def _frame_timestamp_ns(frame_index: int, fps: float) -> int:
    return int(round(frame_index * 1_000_000_000 / fps))


__all__ = ["MCAPDataset"]
