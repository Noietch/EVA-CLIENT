"""Dataset transport for offline inference and replay."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Silence ffmpeg's stderr spam (e.g. "av1 ... Missing Sequence Header") emitted
# while OpenCV probes a codec it cannot decode before falling back to PyAV. Must be
# set before cv2/av initialize their ffmpeg backends. AV_LOG_QUIET = -8.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import av
import av.container
import av.logging
import cv2
import numpy as np

from core.config import ConfigDict, resolve_video_key
from core.datasets import EpisodeData, open_dataset
from core.registry import TRANSPORT_REGISTRY
from core.types import Observation
from robots.base import ActuatorGroup, Robot
from transport.base import TransportBridge

av.logging.set_level(av.logging.PANIC)

logger = logging.getLogger(__name__)
_DEFAULT_FPS = 30
_TWO_FINGER_STROKE_M = 0.04
_EEF_DIMS = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")


def _is_eef_key(key: str) -> bool:
    normalized = key.lower().replace(".", "_").replace("/", "_")
    return "eef" in normalized


def _eef_dimension_names(robot: Robot, vector_dim: int) -> list[str]:
    per_arm_dim = vector_dim // len(robot.arm_groups)
    return [f"{group.name}.{dim}" for group in robot.arm_groups for dim in _EEF_DIMS[:per_arm_dim]]


class _FrameSource:
    """Random-access video frame reader with an OpenCV fast path and a PyAV fallback.

    OpenCV handles common codecs (h264) quickly. When OpenCV cannot decode the
    stream (e.g. av1 on platforms without a system av1 decoder), this transparently
    falls back to PyAV, which ships a software av1 decoder (libdav1d) in its wheel.

    Frames are returned BGR (OpenCV-native) so downstream color handling — which
    expects cv2.VideoCapture output — is unchanged regardless of backend.
    """

    def __init__(self, path: Path) -> None:
        self._path = str(path)
        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(self._path)
        self._use_av = not self._probe_opencv()
        # PyAV state: a forward-only decode iterator + the next frame index it will yield.
        self._container: av.container.InputContainer | None = None
        self._iter = None
        self._next_idx = 0
        if self._use_av:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()
            self._cap = None
            self._open_av(0)
            logger.info("av1/PyAV decode fallback for %s", self._path)

    def _probe_opencv(self) -> bool:
        # OpenCV may "open" a container it cannot actually decode; require a real frame.
        if self._cap is None or not self._cap.isOpened():
            return False
        ok, _ = self._cap.read()
        if ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return ok

    def _open_av(self, start_idx: int) -> None:
        if self._container is not None:
            self._container.close()
        container = av.open(self._path)
        self._container = container
        self._iter = container.decode(container.streams.video[0])
        self._next_idx = 0
        # Skip forward to start_idx without materializing intermediate frames.
        for _ in range(start_idx):
            next(self._iter, None)
            self._next_idx += 1

    def read_at(self, idx: int) -> np.ndarray | None:
        """Return frame ``idx`` as a BGR uint8 array, or None on failure."""
        if self._use_av:
            return self._read_av(idx)
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def _read_av(self, idx: int) -> np.ndarray | None:
        # Sequential reads just advance the iterator; backward jumps reopen from 0.
        if self._iter is None or idx < self._next_idx:
            self._open_av(idx)
        assert self._iter is not None
        frame = None
        while self._next_idx <= idx:
            frame = next(self._iter, None)
            self._next_idx += 1
            if frame is None:
                return None
        if frame is None:
            return None
        rgb = frame.to_ndarray(format="rgb24")
        return rgb[:, :, ::-1].copy()

    def release(self) -> None:
        """Release the OpenCV capture and/or PyAV container backing this source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._container is not None:
            self._container.close()
            self._container = None
            self._iter = None


class DatasetTransport(TransportBridge):
    """Reads supported robotics datasets for offline inference and replay."""

    def __init__(
        self,
        config: ConfigDict,
        robot: Robot,
        dataset_dir: str | Path,
        episode_id: int = 0,
    ) -> None:
        self._config = config
        self._robot = robot
        self._dataset_dir = Path(dataset_dir)
        requested_format = str(config.transport.get("dataset_format", "auto"))
        self._io = open_dataset(self._dataset_dir, format=requested_format)
        self._shutdown = threading.Event()
        self._lock = threading.Lock()

        schema = robot.observation_schema
        disabled_cameras = set(config.transport.disabled_cameras)
        self._disabled_groups = set(config.transport.disabled_groups)
        self._camera_keys = [
            cam.observation_key
            for cam in schema.cameras
            if cam.observation_key not in disabled_cameras
        ]
        self._group_initial_qpos: dict[str, np.ndarray] = {}
        self._enabled_groups: list[ActuatorGroup] = []
        offset = 0
        for group in robot.actuator_groups:
            self._group_initial_qpos[group.name] = np.asarray(
                robot.initial_qpos[offset : offset + group.dof], dtype=np.float32
            )
            if group.name not in self._disabled_groups:
                self._enabled_groups.append(group)
            offset += group.dof
        self._robot_qpos_dim = offset
        self._total_episodes = self._io.count_episodes()
        self._fallback_fps = max(
            1, int(round(float(config.transport.get("fps", _DEFAULT_FPS))))
        )
        self._fps = self._fallback_fps
        self._timestamps: np.ndarray | None = None
        self._caps: dict[str, _FrameSource] = {}
        self._video_paths: dict[str, Path] = {}
        self._video_offsets: dict[str, float] = {}
        self._embedded_images: dict[str, np.ndarray] = {}
        self._episode_id = episode_id
        self._load_episode(episode_id)

    @property
    def fps(self) -> int:
        """Recorded capture rate; replay plays back at this rate."""
        return self._fps

    @property
    def data_format(self) -> str:
        """Canonical format identifier selected by the dataset reader."""
        return self._io.format

    def _load_timestamps(self, columns: Mapping[str, Any]) -> np.ndarray | None:
        if "timestamp" not in columns:
            return None
        timestamps = np.asarray(columns["timestamp"], dtype=np.float32).reshape(-1)
        return timestamps if timestamps.shape[0] == self._n_steps else None

    def _fps_from_timestamps(self, timestamps: np.ndarray | None) -> int | None:
        if timestamps is None or timestamps.shape[0] < 2:
            return None
        deltas = np.diff(timestamps.astype(np.float64))
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            return None
        return max(1, int(round(1.0 / float(np.median(deltas)))))

    def _load_episode(self, episode_id: int) -> None:
        episode: EpisodeData = self._io.load(episode_id)
        columns = episode.columns
        keys = self._config.transport.dataset_keys
        missing = [key for key in (keys.state_key, keys.action_key) if key not in columns]
        if missing:
            available = ", ".join(sorted(columns))
            raise KeyError(f"Dataset columns missing {missing}; available columns: {available}")
        self._obs_state = np.asarray(columns[keys.state_key], dtype=np.float32)
        self._actions = np.asarray(columns[keys.action_key], dtype=np.float32)
        if "control_source" in columns:
            self._control_source = [str(value) for value in columns["control_source"]]
        elif "intervention" in columns:
            self._control_source = [
                "intervention" if value else "policy" for value in columns["intervention"]
            ]
        else:
            self._control_source = ["policy"] * len(self._actions)
        self._intervention = [value == "intervention" for value in self._control_source]
        self._intervention_segment_index = (
            np.asarray(columns["intervention_segment_index"], dtype=np.int64).reshape(-1).tolist()
            if "intervention_segment_index" in columns
            else [-1] * len(self._actions)
        )
        series_state_key = keys.get("series_state_key", "")
        self._series_state = (
            np.asarray(columns[series_state_key], dtype=np.float32)
            if series_state_key and series_state_key in columns
            else self._obs_state
        )
        self._state_names = (
            _eef_dimension_names(self._robot, self._series_state.shape[1])
            if _is_eef_key(series_state_key) and self._series_state.ndim > 1
            else []
        )
        self._action_names = (
            _eef_dimension_names(self._robot, self._actions.shape[1])
            if _is_eef_key(keys.action_key) and self._actions.ndim > 1
            else []
        )
        self._n_steps = self._obs_state.shape[0]
        self._timestamps = self._load_timestamps(columns)
        self._fps = self._fallback_fps
        if episode.fps:
            self._fps = max(1, int(round(episode.fps)))
        timestamp_fps = self._fps_from_timestamps(self._timestamps)
        if timestamp_fps is not None:
            self._fps = timestamp_fps
        self._current_task = episode.task

        for cap in self._caps.values():
            cap.release()
        self._caps = {}
        self._video_paths = {}
        self._video_offsets = {}
        self._embedded_images = {}
        video_keys = {
            cam_key: video_key
            for cam_key in self._camera_keys
            if (video_key := resolve_video_key(keys, cam_key)) is not None
        }
        for cam_key, data_key in video_keys.items():
            video = episode.videos.get(data_key)
            if video is not None:
                self._video_paths[cam_key] = video.path
                self._video_offsets[cam_key] = video.start_time
                continue
            images = episode.images.get(data_key)
            if images is not None:
                self._embedded_images[cam_key] = np.asarray(images)
                continue
            logger.warning("Image stream not found for camera %s (%s)", cam_key, data_key)

        self._episode_id = episode_id
        self._frame_index = 0
        self._qpos = (
            self._obs_state[0].copy() if self._n_steps > 0 else self._robot.initial_qpos.copy()
        )
        self._recorded_qpos = []
        # Avoid re-decoding when the frame index is unchanged between get_frame calls.
        self._frame_cache_idx = -1
        self._frame_cache = {}

        logger.info(
            "Dataset transport ready: %s ep=%d, %d steps, cameras=%s",
            self._dataset_dir,
            episode_id,
            self._n_steps,
            sorted(set(self._video_paths) | set(self._embedded_images)),
        )

    def reload_episode(self, episode_id: int) -> None:
        """Swap the active episode in place and reopen its media."""
        # Swap the active episode in place without recreating the transport object.
        with self._lock:
            self._load_episode(episode_id)

    @property
    def n_episodes(self) -> int:
        """Total number of episodes in the dataset."""
        return self._total_episodes

    def has_episode(self, episode_id: int) -> bool:
        """True if episode_id is a valid index into the dataset."""
        return 0 <= episode_id < self._total_episodes

    @property
    def episode_id(self) -> int:
        """Index of the currently loaded episode."""
        return self._episode_id

    @property
    def current_task(self) -> str:
        """Prompt text of the currently loaded episode ("" if none recorded)."""
        return self._current_task

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the currently loaded episode."""
        return self._n_steps

    def available_camera_keys(self) -> tuple[str, ...]:
        """Camera observation keys that have video or embedded image frames."""
        available = set(self._video_paths) | set(self._embedded_images)
        return tuple(cam_key for cam_key in self._camera_keys if cam_key in available)

    def _frame_source(self, cam_key: str) -> _FrameSource | None:
        cap = self._caps.get(cam_key)
        if cap is not None:
            return cap
        video_path = self._video_paths.get(cam_key)
        if video_path is None:
            return None
        cap = _FrameSource(video_path)
        self._caps[cam_key] = cap
        return cap

    def _read_camera_frame(self, cam_key: str, index: int) -> np.ndarray | None:
        images = self._embedded_images.get(cam_key)
        if images is not None:
            if 0 <= index < len(images):
                frame = np.asarray(images[index])
                return frame.copy() if frame.ndim == 3 else None
            return None
        cap = self._frame_source(cam_key)
        if cap is None:
            return None
        offset = int(round(self._video_offsets.get(cam_key, 0.0) * self._fps))
        return cap.read_at(index + offset)

    def get_action_trajectory(self) -> np.ndarray:
        """Full recorded action sequence [n_steps, action_dim] float32 for replay."""
        # Full recorded action sequence (n_steps, action_dim) for replay playback.
        return self._actions.copy()

    def series(self) -> dict:
        """Whole-episode state/action time series for the console replay charts.

        Mirrors EpisodeLogger.load_episode_series so the frontend reuses one chart
        path. The dataset parquet has no timestamp column, so it is synthesized from
        the dataset fps to drive the scrub clock.

        Returns:
            {"timestamp": [n], "state": [n, Ds], "action": [n, Da]} as plain lists.
        """
        n = self._n_steps
        fps = max(1, self._fps)
        if self._timestamps is not None:
            timestamps = self._timestamps.astype(float).tolist()
        else:
            timestamps = [i / fps for i in range(n)]
        return {
            "timestamp": timestamps,
            "state": self._series_state.tolist(),
            "action": self._actions.tolist(),
            "state_names": self._state_names,
            "action_names": self._action_names,
            "control_source": self._control_source,
            "intervention": self._intervention,
            "intervention_segment_index": self._intervention_segment_index,
        }

    def video_paths(self) -> dict[str, Path]:
        """Map each available camera key to its on-disk mp4 for the loaded episode.

        Only cameras whose video file actually exists are included, so the console can
        stream the recorded footage (native <video>) instead of re-decoding per frame.
        """
        with self._lock:
            return dict(self._video_paths)

    def video_offsets(self) -> dict[str, float]:
        """Map camera keys to their episode start offsets within shared video files."""
        with self._lock:
            return dict(self._video_offsets)

    @property
    def video_mode(self) -> str:
        """Use frame endpoints when an episode stores any camera images inline."""
        return "frames" if self._embedded_images else "native"

    @property
    def frame_index(self) -> int:
        """Index of the frame the next get_frame() will return."""
        with self._lock:
            return self._frame_index

    @property
    def recorded_qpos(self) -> list[np.ndarray]:
        """Copy of the qpos vectors published so far during this replay run."""
        with self._lock:
            return list(self._recorded_qpos)

    def seek(self, index: int) -> None:
        """Move the playback cursor to ``index``, clamped to [0, n_steps - 1]."""
        with self._lock:
            self._frame_index = max(0, min(index, self._n_steps - 1))

    def advance(self) -> bool:
        """Step the cursor forward one frame; return False if already at the end."""
        with self._lock:
            if self._frame_index >= self._n_steps - 1:
                return False
            self._frame_index += 1
            return True

    def get_obs_state(self, index: int | None = None) -> np.ndarray:
        """Recorded state vector at ``index`` (current frame if None), float32."""
        idx = index if index is not None else self._frame_index
        return self._obs_state[idx].copy()

    def _fit_group_qpos(self, group: ActuatorGroup, qpos: np.ndarray) -> np.ndarray:
        if qpos.shape[0] == group.dof:
            return qpos.astype(np.float32, copy=True)

        out = self._group_initial_qpos[group.name].copy()
        n = min(group.dof, qpos.shape[0])
        out[:n] = qpos[:n]
        if group.gripper_index is not None and qpos.shape[0] == group.dof + 1:
            fingers = qpos[group.gripper_index : group.gripper_index + 2]
            out[group.gripper_index] = np.clip(
                float(np.mean(fingers) / _TWO_FINGER_STROKE_M), 0.0, 1.0
            )
        return out

    def _expand_robot_qpos(self, qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float32)
        if q.shape[0] == self._robot_qpos_dim or not self._disabled_groups:
            return q.copy()

        parts: list[np.ndarray] = []
        if len(self._enabled_groups) == 1:
            active_group = self._enabled_groups[0]
            for group in self._robot.actuator_groups:
                if group.name in self._disabled_groups:
                    parts.append(self._group_initial_qpos[group.name])
                elif group.name == active_group.name:
                    parts.append(self._fit_group_qpos(group, q))
            return np.concatenate(parts, axis=0)

        offset = 0
        for group in self._robot.actuator_groups:
            if group.name in self._disabled_groups:
                parts.append(self._group_initial_qpos[group.name])
                continue
            part = q[offset : offset + group.dof]
            parts.append(self._fit_group_qpos(group, part))
            offset += group.dof
        return np.concatenate(parts, axis=0)

    def get_scene_qpos(self, index: int | None = None) -> np.ndarray:
        """Recorded state at ``index`` expanded to the robot's full qpos layout.

        Re-inserts disabled groups (filled with their initial qpos) so the result
        is a complete [robot_qpos_dim] float32 vector for scene visualization,
        regardless of which groups this deployment enabled.
        """
        return self._expand_robot_qpos(self.get_obs_state(index))

    def _frame_timestamp(self, index: int) -> float:
        if self._timestamps is not None:
            return float(self._timestamps[index])
        return float(index) / float(max(1, self._fps))

    def get_frame(self) -> Observation | None:
        """Decode the observation at the current frame index.

        Returns:
            Observation with one BGR uint8 image [H, W, 3] per camera (cameras the
            dataset lacks are filled with black frames) and the recorded state
            [state_dim] float32, or None if shut down or past the last frame.
        """
        with self._lock:
            if self._shutdown.is_set():
                return None
            idx = self._frame_index
            if idx >= self._n_steps:
                return None
            state = self._obs_state[idx].copy()
            # Reuse the cache only when it holds every camera; get_camera_frame may
            # have filled it with a single view for the MJPEG feed.
            if idx == self._frame_cache_idx and all(
                k in self._frame_cache for k in self._camera_keys
            ):
                return Observation(
                    images={k: self._frame_cache[k].copy() for k in self._camera_keys},
                    state_qpos=state,
                    timestamp=self._frame_timestamp(idx),
                )

            h = self._config.transport.image_height
            w = self._config.transport.image_width
            images: dict[str, np.ndarray] = {}
            for cam_key in self._camera_keys:
                frame = self._read_camera_frame(cam_key, idx)
                if frame is not None:
                    images[cam_key] = frame
                else:
                    images[cam_key] = np.zeros((h, w, 3), dtype=np.uint8)

            self._frame_cache_idx = idx
            self._frame_cache = {k: v.copy() for k, v in images.items()}
        return Observation(images=images, state_qpos=state, timestamp=self._frame_timestamp(idx))

    def get_camera_frame(self, key: str) -> np.ndarray | None:
        """Decode a single camera at the current frame index for the live MJPEG feed.

        The console opens one stream per camera; routing each through get_frame()
        would re-decode and copy *every* camera per stream. This decodes only ``key``
        and reuses the shared per-frame cache so a paused replay never re-decodes.

        Args:
            key: camera observation key (e.g. "cam_high").

        Returns:
            BGR uint8 array [H, W, 3], or None if the camera/frame is unavailable.
            The returned array is read-only (caller must not mutate it).
        """
        with self._lock:
            if self._shutdown.is_set():
                return None
            idx = self._frame_index
            if idx >= self._n_steps:
                return None
            if idx != self._frame_cache_idx:
                # Frame advanced (or first read): drop the stale per-camera cache.
                self._frame_cache_idx = idx
                self._frame_cache = {}
            cached = self._frame_cache.get(key)
            if cached is not None:
                return cached
            frame = self._read_camera_frame(key, idx)
            if frame is None:
                h = self._config.transport.image_height
                w = self._config.transport.image_width
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            self._frame_cache[key] = frame
            return frame

    def get_camera_frame_at(self, key: str, index: int) -> np.ndarray | None:
        """Decode one camera at an explicit episode frame without moving playback."""
        with self._lock:
            if self._shutdown.is_set() or not 0 <= index < self._n_steps:
                return None
            return self._read_camera_frame(key, index)

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        """Record the action as the latest qpos and advance the playback cursor.

        Replay sink: instead of driving a robot it stores the action [action_dim]
        float32 for get_latest_qpos(), appends it to the recorded history, and
        steps the frame index forward (stopping at the last frame).
        """
        qpos = np.asarray(action, dtype=np.float32).copy()
        with self._lock:
            self._qpos = qpos
            self._recorded_qpos.append(qpos)
            if self._frame_index < self._n_steps - 1:
                self._frame_index += 1
        logger.debug(
            "Dataset publish [%s] step=%d frame=%d: %s",
            target,
            len(self._recorded_qpos),
            self._frame_index,
            qpos[:4],
        )

    def get_latest_qpos(self) -> np.ndarray | None:
        """Last published qpos expanded to the robot's full layout [qpos_dim] float32."""
        with self._lock:
            return self._expand_robot_qpos(self._qpos)

    def is_offline(self) -> bool:
        """Always True — a dataset replay has no live link."""
        return True

    def close(self) -> None:
        """Mark shut down and release all per-camera video captures."""
        self._shutdown.set()
        with self._lock:
            for cap in self._caps.values():
                cap.release()
            self._caps.clear()
        logger.debug("Dataset transport closed")

    def is_shutdown(self) -> bool:
        """Return True once close() has been called."""
        return self._shutdown.is_set()


@TRANSPORT_REGISTRY.register("dataset")
def _build_dataset(config: ConfigDict, robot: Robot) -> TransportBridge:
    dataset_dir = config.transport.dataset_dir
    if not dataset_dir:
        raise ValueError("transport.dataset_dir is required for dataset transport")
    return DatasetTransport(config, robot, dataset_dir, config.transport.episode_id)
