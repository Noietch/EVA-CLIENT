"""HTTP lifecycle and streaming validation for simulator recordings."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from mcap.exceptions import McapError
from mcap.reader import make_reader

_SIM_OPENER = build_opener(ProxyHandler({}))
_SIM_EXECUTION_TOPIC = "/sim_execution"
_SIM_EXECUTION_ENCODING = "json"


class SimRecordingError(RuntimeError):
    """Raised when simulator recording state or data violates the contract."""


@dataclass(frozen=True)
class SimRecordingHandle:
    episode_id: str
    path: Path
    scene_revision: int


@dataclass(frozen=True)
class SimRecordingReply:
    episode_id: str
    path: Path
    status: str
    steps: int


@dataclass(frozen=True)
class SimRecordingFrame:
    """One observe-before-action frame extracted without retaining raw state."""

    step_id: int
    sim_time: float
    state_qpos: np.ndarray
    action_qpos: np.ndarray


@dataclass(frozen=True)
class SimExecutionRecord:
    path: Path
    episode_id: str
    schema_version: int
    steps: int
    control_dt: float
    physics_dt: float
    sim_metadata: Mapping[str, Any]
    initial_qpos: np.ndarray
    frames: tuple[SimRecordingFrame, ...]
    final_state_qpos: np.ndarray

    def iter_frames(self, expected_dim: int | None = None) -> Iterator[SimRecordingFrame]:
        """Stream compact observe-before-action frames after MCAP validation."""
        for frame in self.frames:
            if expected_dim is not None:
                next_state = (
                    self.frames[frame.step_id + 1].state_qpos
                    if frame.step_id + 1 < self.steps
                    else self.final_state_qpos
                )
                _validate_dimension(next_state, expected_dim, "state.qpos")
                _validate_dimension(frame.action_qpos, expected_dim, "action")
            yield SimRecordingFrame(
                step_id=frame.step_id,
                sim_time=frame.sim_time,
                state_qpos=frame.state_qpos.copy(),
                action_qpos=frame.action_qpos.copy(),
            )


class SimRecordingClient:
    """Small synchronous client for the simulator recording API."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def get_status(self, timeout: float | None = None) -> Mapping[str, Any]:
        """Return the simulator status data used to pin scene_revision."""
        url = f"{self._base_url}/api/status"
        request = Request(url, method="GET")
        try:
            with _SIM_OPENER.open(
                request, timeout=self._timeout if timeout is None else timeout
            ) as response:
                raw = response.read()
        except (HTTPError, URLError, OSError) as error:
            raise SimRecordingError(f"SIM status RPC failed: {error}") from error
        payload = _decode_response(raw)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise SimRecordingError("SIM status response has no data object")
        return data

    def resolve_scene_revision(self, expected_metadata: Mapping[str, Any] | None = None) -> int:
        return _scene_revision(self.get_status(), expected_metadata)

    def recording_steps(self, episode_id: str) -> int:
        """Read the live control-step count for one active recording."""
        data = self.get_status(timeout=min(self._timeout, 1.0))
        recording = data.get("recording")
        if not isinstance(recording, Mapping):
            raise SimRecordingError("SIM status has no active recording summary")
        if recording.get("episode_id") != episode_id:
            raise SimRecordingError("SIM status recording episode_id does not match")
        try:
            steps = int(recording["steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise SimRecordingError("SIM status recording steps is invalid") from error
        if steps < 0:
            raise SimRecordingError("SIM status recording steps is negative")
        return steps

    def start(
        self,
        handle: SimRecordingHandle,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> SimRecordingReply:
        """Pin the current scene and begin recording at the exclusive path."""
        status = self.get_status()
        scene_revision = _scene_revision(status, expected_metadata)
        if scene_revision != handle.scene_revision:
            raise SimRecordingError(
                "SIM scene_revision changed before recording start "
                f"({handle.scene_revision} != {scene_revision})"
            )
        payload = self._post(
            "/api/recording/start",
            {
                "episode_id": handle.episode_id,
                "path": str(handle.path),
                "scene_revision": scene_revision,
            },
        )
        data = _required_data(payload)
        reply = _recording_reply(data, "recording", handle.episode_id, handle.path)
        if reply.steps != 0:
            raise SimRecordingError("SIM start response must report steps=0")
        return reply

    def stop(self, handle: SimRecordingHandle) -> SimRecordingReply:
        payload = self._post("/api/recording/stop", {"episode_id": handle.episode_id})
        return _recording_reply(_required_data(payload), "complete", handle.episode_id, handle.path)

    def cancel(self, handle: SimRecordingHandle) -> SimRecordingReply:
        payload = self._post("/api/recording/cancel", {"episode_id": handle.episode_id})
        return _recording_reply(
            _required_data(payload), "cancelled", handle.episode_id, handle.path
        )

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self._base_url}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _SIM_OPENER.open(request, timeout=self._timeout) as response:
                raw = response.read()
        except (HTTPError, URLError, OSError) as error:
            raise SimRecordingError(f"SIM {endpoint} RPC failed: {error}") from error
        return _decode_response(raw)


def create_sim_recording_handle(dataset_dir: Path, scene_revision: int) -> SimRecordingHandle:
    """Create the raw directory and reserve an absolute UUID pathname for SIM."""
    raw_dir = dataset_dir.resolve() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    episode_id = str(uuid.uuid4())
    path = raw_dir / f"{episode_id}.mcap"
    if path.exists():
        raise SimRecordingError(f"generated SIM recording path already exists: {path}")
    return SimRecordingHandle(episode_id, path, int(scene_revision))


def read_sim_execution(
    path: Path,
    expected_episode_id: str,
    expected_steps: int | None = None,
) -> SimExecutionRecord:
    """Validate MCAP framing and retain only compact qpos/action frames."""
    payloads = _iter_mcap_payloads(path)
    header = next(payloads, None)
    if header is None:
        raise SimRecordingError("SIM recording is empty or missing a header")
    episode_id, control_dt, physics_dt, sim_metadata, initial_qpos = _validate_header(
        header, expected_episode_id, include_values=True
    )
    seen_steps = 0
    footer: dict[str, Any] | None = None
    previous_qpos = initial_qpos.copy()
    frames: list[SimRecordingFrame] = []
    for message_number, payload in enumerate(payloads, start=2):
        payload_type = payload.get("type")
        if payload_type == "step":
            if footer is not None:
                raise SimRecordingError("step found after footer")
            frame_state, action, _ = _validate_step(payload, seen_steps, control_dt)
            frames.append(
                SimRecordingFrame(
                    step_id=seen_steps,
                    sim_time=seen_steps * control_dt,
                    state_qpos=previous_qpos.copy(),
                    action_qpos=action,
                )
            )
            previous_qpos = frame_state
            seen_steps += 1
        elif payload_type == "footer":
            if footer is not None:
                raise SimRecordingError("duplicate footer")
            footer = payload
        else:
            raise SimRecordingError(f"unexpected MCAP record type at message {message_number}")
    _validate_footer(footer, seen_steps, expected_steps)
    if seen_steps <= 0:
        raise SimRecordingError("SIM recording contains no control steps")
    return SimExecutionRecord(
        path=path.resolve(),
        episode_id=episode_id,
        schema_version=1,
        steps=seen_steps,
        control_dt=control_dt,
        physics_dt=physics_dt,
        sim_metadata=sim_metadata,
        initial_qpos=initial_qpos,
        frames=tuple(frames),
        final_state_qpos=previous_qpos,
    )


def _iter_mcap_payloads(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            for message_number, (_, channel, message) in enumerate(
                reader.iter_messages(log_time_order=False), start=1
            ):
                if message.sequence != message_number - 1:
                    raise SimRecordingError(
                        f"MCAP message sequence gap at message {message_number}"
                    )
                if channel.topic != _SIM_EXECUTION_TOPIC:
                    raise SimRecordingError(f"unexpected MCAP topic at message {message_number}")
                if channel.message_encoding != _SIM_EXECUTION_ENCODING:
                    raise SimRecordingError(f"unexpected MCAP encoding at message {message_number}")
                try:
                    payload = json.loads(message.data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SimRecordingError(
                        f"invalid MCAP JSON at message {message_number}"
                    ) from error
                if not isinstance(payload, dict):
                    raise SimRecordingError(f"MCAP message {message_number} is not an object")
                yield payload
    except FileNotFoundError as error:
        raise SimRecordingError(f"SIM recording file is missing: {path}") from error
    except SimRecordingError:
        raise
    except (McapError, EOFError, OSError, ValueError) as error:
        raise SimRecordingError(f"invalid MCAP recording: {path}") from error


def _decode_response(raw: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimRecordingError("SIM RPC returned invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise SimRecordingError("SIM RPC returned ok=false")
    return payload


def _required_data(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise SimRecordingError("SIM recording response has no data object")
    return data


def _recording_reply(
    data: Mapping[str, Any],
    status: str,
    episode_id: str,
    path: Path,
) -> SimRecordingReply:
    if data.get("episode_id") != episode_id or data.get("status") != status:
        raise SimRecordingError(f"SIM recording response is not {status}")
    response_path = data.get("path")
    if response_path is not None and Path(str(response_path)).resolve() != path.resolve():
        raise SimRecordingError("SIM recording response path does not match CLIENT path")
    try:
        steps = int(data.get("steps", 0))
    except (TypeError, ValueError) as error:
        raise SimRecordingError("SIM recording response has invalid steps") from error
    if steps < 0:
        raise SimRecordingError("SIM recording response has negative steps")
    return SimRecordingReply(episode_id, path, status, steps)


def _scene_revision(status: Mapping[str, Any], expected_metadata: Mapping[str, Any] | None) -> int:
    _check_scene_metadata(status, expected_metadata)
    try:
        revision = int(status["scene_revision"])
    except (KeyError, TypeError, ValueError) as error:
        raise SimRecordingError("SIM status has no valid scene_revision") from error
    if revision < 0:
        raise SimRecordingError("SIM scene_revision must be non-negative")
    return revision


def _check_scene_metadata(
    status: Mapping[str, Any], expected_metadata: Mapping[str, Any] | None
) -> None:
    if not expected_metadata:
        return
    actual: dict[str, Any] = dict(status)
    for key in ("metadata", "sim_metadata"):
        nested = status.get(key)
        if isinstance(nested, Mapping):
            actual.update(nested)
    expected_sim = expected_metadata.get("sim_metadata")
    if isinstance(expected_sim, Mapping):
        expected = {**expected_metadata, **expected_sim}
        task_item = expected_sim.get("task_item")
        if isinstance(task_item, Mapping) and task_item.get("task_item_id") is not None:
            expected["task_id"] = task_item["task_item_id"]
    else:
        expected = expected_metadata
    aliases = {
        "task_id": ("task_id",),
        "layout": ("layout", "scene_id"),
        "scene_id": ("scene_id", "layout"),
        "scene_round": ("scene_round",),
    }
    for actual_key, expected_keys in aliases.items():
        actual_value = actual.get(actual_key)
        if actual_value is None:
            continue
        if actual_key == "layout" and isinstance(actual_value, Mapping):
            if not isinstance(expected.get("layout"), Mapping):
                continue
        expected_value = next(
            (expected[key] for key in expected_keys if expected.get(key) is not None),
            None,
        )
        if expected_value is not None and str(actual_value) != str(expected_value):
            raise SimRecordingError(
                f"SIM scene metadata mismatch for {actual_key}: "
                f"{actual_value!r} != {expected_value!r}"
            )


def _validate_header(
    header: Mapping[str, Any],
    expected_episode_id: str,
    include_values: bool = False,
) -> Any:
    if header.get("type") != "header" or header.get("schema_version") != 1:
        raise SimRecordingError("SIM recording header schema is invalid")
    if header.get("episode_id") != expected_episode_id:
        raise SimRecordingError("SIM recording episode_id does not match")
    sim_metadata = header.get("sim_metadata")
    if not isinstance(sim_metadata, Mapping):
        raise SimRecordingError("SIM recording header sim_metadata is invalid")
    control_dt = _positive_float(header.get("control_dt"), "control_dt")
    physics_dt = _positive_float(header.get("physics_dt"), "physics_dt")
    initial_qpos = _validate_state(header.get("initial_state"), "initial_state")[0]
    if include_values:
        return expected_episode_id, control_dt, physics_dt, dict(sim_metadata), initial_qpos


def _validate_step(
    payload: Mapping[str, Any], expected_step_id: int, control_dt: float
) -> tuple[np.ndarray, np.ndarray, float]:
    if payload.get("step_id") != expected_step_id:
        raise SimRecordingError(
            f"SIM step sequence gap: expected {expected_step_id}, got {payload.get('step_id')}"
        )
    if "source_frame_id" not in payload:
        raise SimRecordingError("SIM step is missing source_frame_id")
    sim_time = _positive_float(payload.get("sim_time"), "sim_time")
    expected_time = (expected_step_id + 1) * control_dt
    if not math.isclose(sim_time, expected_time, rel_tol=1e-6, abs_tol=1e-7):
        raise SimRecordingError(
            f"SIM step {expected_step_id} sim_time is not control-clock aligned"
        )
    action = _vector(payload.get("action"), "action")
    state_qpos, _, _, _ = _validate_state(payload.get("state"), "state")
    return state_qpos, action, sim_time


def _validate_state(
    value: Any, field: str
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise SimRecordingError(f"SIM {field} is not an object")
    qpos = _vector(value.get("qpos"), f"{field}.qpos")
    parts: list[Mapping[str, Any]] = []
    for name in ("scene_state", "objects", "liquids"):
        part = value.get(name)
        if not isinstance(part, Mapping):
            raise SimRecordingError(f"SIM {field}.{name} is not an object")
        parts.append(part)
    return qpos, parts[0], parts[1], parts[2]


def _vector(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or not value:
        raise SimRecordingError(f"SIM {field} must be a non-empty array")
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise SimRecordingError(f"SIM {field} is not numeric") from error
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise SimRecordingError(f"SIM {field} must be a finite vector")
    return vector


def _validate_dimension(vector: np.ndarray, expected_dim: int, field: str) -> None:
    if vector.shape != (expected_dim,):
        raise SimRecordingError(
            f"SIM {field} dimension mismatch: {vector.shape} != {(expected_dim,)}"
        )


def _positive_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SimRecordingError(f"SIM {field} is not numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise SimRecordingError(f"SIM {field} must be positive and finite")
    return result


def _validate_footer(
    footer: Mapping[str, Any] | None, seen_steps: int, expected_steps: int | None
) -> None:
    if footer is None or footer.get("type") != "footer" or footer.get("complete") is not True:
        raise SimRecordingError("SIM recording is incomplete")
    if footer.get("steps") != seen_steps:
        raise SimRecordingError("SIM footer step count does not match MCAP")
    if expected_steps is not None and expected_steps != seen_steps:
        raise SimRecordingError(
            f"SIM stop reported {expected_steps} steps but MCAP has {seen_steps}"
        )
