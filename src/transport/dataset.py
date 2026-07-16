"""Dataset transport — reads LeRobot v2 dataset (parquet + MP4) for offline inference and replay."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

# Silence ffmpeg's stderr spam (e.g. "av1 ... Missing Sequence Header") emitted
# while OpenCV probes a codec it cannot decode before falling back to PyAV. Must be
# set before cv2/av initialize their ffmpeg backends. AV_LOG_QUIET = -8.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import av
import av.container
import av.logging
import cv2
import numpy as np
import pyarrow.parquet as pq

from core.config import ConfigDict, resolve_video_key
from core.registry import TRANSPORT_REGISTRY
from core.types import Observation
from core.utils.lerobot import LeRobotDatasetIO
from robots.base import ActuatorGroup, Robot
from transport.base import TransportBridge

av.logging.set_level(av.logging.PANIC)

logger = logging.getLogger(__name__)
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
    """Reads observations from a LeRobot v2 dataset directory.

    Expected layout (LeRobot v2.1)::

        dataset_dir/
            meta/info.json
            data/chunk-000/episode_000000.parquet
            videos/chunk-000/observation.images.cam_high/episode_000000.mp4

    Provides step-by-step frame access for offline inference and replay.
    """

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
        self._io = LeRobotDatasetIO(self._dataset_dir)
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
        self._tasks_by_index = self._io.tasks_by_index()
        self._total_episodes = self._io.count_episodes()
        self._fps = self._load_fps()
        self._timestamps: np.ndarray | None = None
        self._caps: dict[str, _FrameSource] = {}
        self._video_paths: dict[str, Path] = {}
        self._episode_id = episode_id
        self._load_episode(episode_id)

    def _load_fps(self) -> int:
        # Recorded capture rate from info.json; replay plays back at this rate so the
        # motion matches the original data collection instead of the publish rate.
        info_path = self._dataset_dir / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                fps = json.load(f).get("fps")
            if fps:
                return int(fps)
        return 10

    @property
    def fps(self) -> int:
        """Recorded capture rate; replay plays back at this rate."""
        return self._fps

    def _resolve_task(self, table: object) -> str:
        return self._io.resolve_task(table, self._tasks_by_index)

    def _load_timestamps(self, table: object) -> np.ndarray | None:
        if "timestamp" not in table.column_names:  # type: ignore[attr-defined]
            return None
        timestamps = np.asarray(
            table.column("timestamp").to_pylist(),  # type: ignore[attr-defined]
            dtype=np.float32,
        ).reshape(-1)
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
        # (Re)open parquet + per-camera video captures for one episode in place.
        parquet_path = self._io.episode_parquet(episode_id)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        keys = self._config.transport.dataset_keys
        table = pq.read_table(str(parquet_path))
        self._obs_state = np.array(table.column(keys.state_key).to_pylist(), dtype=np.float32)
        self._actions = np.array(table.column(keys.action_key).to_pylist(), dtype=np.float32)
        if "control_source" in table.column_names:
            self._control_source = [
                str(value) for value in table.column("control_source").to_pylist()
            ]
        elif "intervention" in table.column_names:
            self._control_source = [
                "intervention" if value else "policy"
                for value in table.column("intervention").to_pylist()
            ]
        else:
            self._control_source = ["policy"] * len(self._actions)
        self._intervention = [value == "intervention" for value in self._control_source]
        self._intervention_segment_index = (
            [int(value) for value in table.column("intervention_segment_index").to_pylist()]
            if "intervention_segment_index" in table.column_names
            else [-1] * len(self._actions)
        )
        series_state_key = keys.get("series_state_key", "")
        self._series_state = (
            np.array(table.column(series_state_key).to_pylist(), dtype=np.float32)
            if series_state_key
            else self._obs_state
        )
        self._state_names = (
            _eef_dimension_names(self._robot, self._series_state.shape[1])
            if _is_eef_key(series_state_key)
            else []
        )
        self._action_names = (
            _eef_dimension_names(self._robot, self._actions.shape[1])
            if _is_eef_key(keys.action_key)
            else []
        )
        self._n_steps = self._obs_state.shape[0]
        self._timestamps = self._load_timestamps(table)
        timestamp_fps = self._fps_from_timestamps(self._timestamps)
        if timestamp_fps is not None:
            self._fps = timestamp_fps
        self._current_task = self._resolve_task(table)

        for cap in self._caps.values():
            cap.release()
        self._caps = {}
        self._video_paths = {}
        video_keys = {
            cam_key: video_key
            for cam_key in self._camera_keys
            if (video_key := resolve_video_key(keys, cam_key)) is not None
        }
        resolved = self._io.episode_video_paths(episode_id, video_keys)
        for cam_key in self._camera_keys:
            if cam_key not in video_keys:
                # Dataset does not provide this camera (robot declares more views than
                # the dataset recorded); get_frame fills it with a black frame.
                continue
            video_path = resolved.get(cam_key)
            if video_path is None:
                logger.warning("Video not found for camera %s", cam_key)
                continue
            self._video_paths[cam_key] = video_path

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
            list(self._video_paths.keys()),
        )

    def reload_episode(self, episode_id: int) -> None:
        """Swap the active episode in place (thread-safe), reopening parquet+videos."""
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
        """Camera observation keys that actually have a video stream this episode."""
        return tuple(cam_key for cam_key in self._camera_keys if cam_key in self._video_paths)

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
        if self._shutdown.is_set():
            return None
        with self._lock:
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
                cap = self._frame_source(cam_key)
                frame = cap.read_at(idx) if cap is not None else None
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
        if self._shutdown.is_set():
            return None
        with self._lock:
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
            cap = self._frame_source(key)
            frame = cap.read_at(idx) if cap is not None else None
            if frame is None:
                h = self._config.transport.image_height
                w = self._config.transport.image_width
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            self._frame_cache[key] = frame
            return frame

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
