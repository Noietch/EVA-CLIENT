"""Measure state-only latency through ZMQ and the EVAClient console.

Synthetic mode starts an in-process 60 Hz publisher and measures the same reader
and URDF transform path used by the live console. Recording mode deliberately
stalls collection, then verifies state/action/timestamps in the saved Parquet.
Live mode compares a fast ZMQ reference reader against EVAClient's HTTP endpoints
while EVA Sim and EVAClient are already running.

Examples:
    python tests/manual/benchmark_state_only_latency.py --mode synthetic
    python tests/manual/benchmark_state_only_latency.py --mode recording
    python tests/manual/benchmark_state_only_latency.py --mode live \
        --obs-endpoint tcp://127.0.0.1:5555 \
        --console-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
import zmq

import robots  # noqa: F401
from core.app.collection_capture import start_collection_capture, stop_collection_capture
from core.app.handlers.recording import COLLECT_STEP_MAX_RAW_SNAPSHOTS
from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger, sanitize_path_component
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene
from transport.zmq import WireObservation, _ObservationReader, pack_observation

_ALIGNMENT_OFFSET = 10.0


def _reader(endpoint: str, robot, preserve_backlog: bool = False) -> _ObservationReader:
    config = SimpleNamespace(
        transport=SimpleNamespace(
            sub_endpoint=endpoint,
            disabled_cameras=[],
            disabled_groups=[],
        )
    )
    return _ObservationReader(
        cast(ConfigDict, config),
        robot,
        zmq,
        preserve_collection_backlog=preserve_backlog,
    )


def _summary(name: str, values: list[float], unit: str) -> dict[str, float]:
    if not values:
        raise RuntimeError(f"No samples collected for {name}")
    array = np.asarray(values, dtype=np.float64)
    result = {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }
    print(
        f"{name}: n={len(array)} p50={result['p50']:.2f}{unit} "
        f"p95={result['p95']:.2f}{unit} max={result['max']:.2f}{unit}"
    )
    return result


def _state(robot, frame_id: int) -> dict[str, np.ndarray]:
    values = 0.2 * np.sin(np.arange(robot.total_action_dim) + frame_id * 0.03)
    state = {}
    offset = 0
    for group in robot.actuator_groups:
        state[group.name] = values[offset : offset + group.dof].astype(np.float32)
        offset += group.dof
    return state


def _free_endpoint() -> str:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"tcp://127.0.0.1:{port}"


def _run_synthetic(args) -> bool:
    endpoint = _free_endpoint()
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(endpoint)
    robot = ROBOT_REGISTRY.build(args.robot)
    reader = _reader(endpoint, robot)
    scene = UrdfScene(robot, gripper_open=1.0, gripper_close=0.0)
    sent_at: dict[int, float] = {}
    latest_frame = [-1]
    stop = threading.Event()
    lock = threading.Lock()

    def publish() -> None:
        frame_id = 0
        period = 1.0 / args.source_hz
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            observation = WireObservation(
                t=frame_id / args.source_hz,
                images={},
                state=_state(robot, frame_id),
                frame_id=frame_id,
                camera_resolution=(480, 640),
            )
            publisher.send(pack_observation(observation))
            with lock:
                sent_at[frame_id] = started
                latest_frame[0] = frame_id
            frame_id += 1
            remaining = period - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        stop.set()

    time.sleep(0.3)
    thread = threading.Thread(target=publish, name="state-latency-publisher")
    thread.start()
    time.sleep(0.1)
    wire_age_ms: list[float] = []
    frame_lag: list[float] = []
    console_path_ms: list[float] = []
    period = 1.0 / args.poll_hz
    while not stop.is_set():
        started = time.perf_counter()
        frame = reader.get_frame()
        camera_keys = reader.get_camera_keys()
        if frame is not None and frame.frame_id is not None:
            json.dumps({"available": True, "arms": scene.transforms(frame.state_qpos)})
            finished = time.perf_counter()
            with lock:
                wire_age_ms.append((finished - sent_at[frame.frame_id]) * 1000.0)
                frame_lag.append(float(latest_frame[0] - frame.frame_id))
            console_path_ms.append((finished - started) * 1000.0)
            if camera_keys:
                raise RuntimeError(f"State-only stream exposed cameras: {camera_keys}")
        remaining = period - (time.perf_counter() - started)
        if remaining > 0.0:
            time.sleep(remaining)

    thread.join()
    reader.close()
    publisher.close(linger=0)
    print("mode=synthetic")
    age = _summary("wire_age", wire_age_ms, "ms")
    lag = _summary("frame_lag", frame_lag, " frames")
    path = _summary("console_path", console_path_ms, "ms")
    return (
        age["p95"] <= args.max_latency_ms
        and lag["p95"] <= 1.0
        and path["p95"] <= args.max_latency_ms
    )


def _recording_logger(root: Path, robot, fps: int) -> EpisodeLogger:
    camera_keys = [camera.observation_key for camera in robot.observation_schema.cameras]
    return EpisodeLogger(
        root,
        robot,
        fps=fps,
        dataset_keys=ConfigDict(
            state_key="observation.qpos",
            eef_key="observation.eef",
            action_key="action",
            video_keys={},
        ),
        collection=ConfigDict(
            enabled=True,
            schema=ConfigDict(
                robot_type=robot.name,
                min_episode_frames=1,
                arms={group.name: group.name for group in robot.arm_groups},
                cameras={key: f"observation.images.{key}" for key in camera_keys},
                columns={"qpos": "observation.qpos", "action_qpos": "action"},
            ),
        ),
    )


def _run_recording(args) -> bool:
    endpoint = _free_endpoint()
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(endpoint)
    robot = ROBOT_REGISTRY.build(args.robot)
    reader = _reader(endpoint, robot, preserve_backlog=True)
    sent_count = [0]

    def publish() -> None:
        period = 1.0 / args.source_hz
        deadline = time.perf_counter() + args.duration
        frame_id = 0
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            capture_time = frame_id / args.source_hz
            state = np.full(robot.total_action_dim, capture_time, dtype=np.float32)
            publisher.send(
                pack_observation(
                    WireObservation(
                        t=capture_time,
                        images={},
                        state=_split_state(robot, state),
                        action=state + _ALIGNMENT_OFFSET,
                        frame_id=frame_id,
                        camera_resolution=(480, 640),
                    )
                )
            )
            sent_count[0] += 1
            frame_id += 1
            remaining = period - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)

    time.sleep(0.3)
    for warmup_time in (60.0, 0.0):
        publisher.send(
            pack_observation(
                WireObservation(
                    t=warmup_time,
                    images={},
                    state=_state(robot, -1),
                    action=np.zeros(robot.total_action_dim, dtype=np.float32),
                    camera_resolution=(480, 640),
                )
            )
        )
    time.sleep(0.1)
    cutoff = reader.clear_collection_backlog()

    with TemporaryDirectory(prefix="eva-state-recording-") as temp_dir:
        root = Path(temp_dir)
        episode_logger = _recording_logger(root, robot, args.record_fps)
        episode_logger.start_episode("state alignment", collection_min_capture_time=cutoff)
        runtime = SimpleNamespace(episode_logger=episode_logger, transport=reader)
        publisher_thread = threading.Thread(target=publish, name="state-recording-publisher")
        publisher_thread.start()
        time.sleep(min(args.capture_stall, args.duration))
        start_collection_capture(
            runtime,
            fps=float(args.record_fps),
            max_raw_snapshots_per_tick=COLLECT_STEP_MAX_RAW_SNAPSHOTS,
        )
        publisher_thread.join()

        catchup_deadline = time.perf_counter() + args.catchup_timeout
        while (
            episode_logger.active_frame_count < sent_count[0]
            and time.perf_counter() < catchup_deadline
        ):
            time.sleep(0.01)
        stop_collection_capture(runtime)
        received_count = episode_logger.active_frame_count
        saved = episode_logger.end_episode()

        task_dir = root / sanitize_path_component("state alignment")
        parquet_path = task_dir / "data" / "chunk-000" / "episode_000000.parquet"
        if not saved or not parquet_path.exists():
            print(f"mode=recording sent={sent_count[0]} received={received_count} saved_rows=0")
            passed = False
        else:
            table = pq.read_table(parquet_path)
            state = np.asarray(table.column("observation.qpos").to_pylist(), dtype=np.float64)
            action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
            timestamp = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
            capture_time = np.asarray(table.column("capture_time").to_pylist(), dtype=np.float64)
            frame_index = table.column("frame_index").to_pylist()
            expected_timestamp = np.arange(table.num_rows, dtype=np.float64) / args.record_fps
            state_action_error = float(np.max(np.abs(action - state - _ALIGNMENT_OFFSET)))
            state_clock_error = float(np.max(np.abs(state[:, 0] - capture_time)))
            timestamp_error = float(np.max(np.abs(timestamp - expected_timestamp)))
            capture_step_error = (
                0.0
                if table.num_rows < 2
                else float(np.max(np.abs(np.diff(capture_time) - 1.0 / args.record_fps)))
            )
            print(
                f"mode=recording stall={args.capture_stall:.2f}s sent={sent_count[0]} "
                f"received={received_count} saved_rows={table.num_rows}"
            )
            print(
                f"state_action_error={state_action_error:.3e} "
                f"state_clock_error={state_clock_error:.3e} "
                f"timestamp_error={timestamp_error:.3e} "
                f"capture_step_error={capture_step_error:.3e}"
            )
            passed = (
                received_count == sent_count[0]
                and table.num_rows > 0
                and frame_index == list(range(table.num_rows))
                and state_action_error <= args.alignment_tolerance
                and state_clock_error <= args.alignment_tolerance
                and timestamp_error <= args.alignment_tolerance
                and capture_step_error <= args.alignment_tolerance
            )

    reader.close()
    publisher.close(linger=0)
    return passed


def _split_state(robot, values: np.ndarray) -> dict[str, np.ndarray]:
    state = {}
    offset = 0
    for group in robot.actuator_groups:
        state[group.name] = values[offset : offset + group.dof]
        offset += group.dof
    return state


def _run_live(args) -> bool:
    robot = ROBOT_REGISTRY.build(args.robot)
    reference = _reader(args.obs_endpoint, robot)
    latest_reference = [-1]
    stop = threading.Event()
    lock = threading.Lock()

    def read_reference() -> None:
        while not stop.is_set():
            observation = reference._drain_latest()
            if observation is not None and observation.frame_id is not None:
                with lock:
                    latest_reference[0] = observation.frame_id
            stop.wait(0.005)

    thread = threading.Thread(target=read_reference, name="state-latency-reference")
    thread.start()
    deadline = time.perf_counter() + 5.0
    while latest_reference[0] < 0 and time.perf_counter() < deadline:
        time.sleep(0.05)
    if latest_reference[0] < 0:
        stop.set()
        thread.join()
        reference.close()
        raise RuntimeError(f"No observations received from {args.obs_endpoint}")

    frame_http_ms: list[float] = []
    scene_http_ms: list[float] = []
    frame_lag: list[float] = []
    freeze_ms: list[float] = []
    last_frame: int | None = None
    last_change = time.perf_counter()
    period = 1.0 / args.poll_hz
    deadline = time.perf_counter() + args.duration
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        with urlopen(f"{args.console_url}/api/frame", timeout=2.0) as response:
            frame_payload = json.load(response)
        frame_finished = time.perf_counter()
        with urlopen(f"{args.console_url}/api/scene", timeout=2.0) as response:
            json.load(response)
        finished = time.perf_counter()
        frame_http_ms.append((frame_finished - started) * 1000.0)
        scene_http_ms.append((finished - frame_finished) * 1000.0)
        frame_id = frame_payload.get("frame_id")
        if frame_id is not None:
            frame_id = int(frame_id)
            with lock:
                frame_lag.append(float(abs(latest_reference[0] - frame_id)))
            if frame_id != last_frame:
                freeze_ms.append((frame_finished - last_change) * 1000.0)
                last_change = frame_finished
                last_frame = frame_id
        remaining = period - (time.perf_counter() - started)
        if remaining > 0.0:
            time.sleep(remaining)

    if last_frame is not None:
        freeze_ms.append((time.perf_counter() - last_change) * 1000.0)
    stop.set()
    thread.join()
    reference.close()
    print("mode=live")
    frame = _summary("frame_http", frame_http_ms, "ms")
    scene = _summary("scene_http", scene_http_ms, "ms")
    lag = _summary("frame_lag", frame_lag, " frames")
    freeze = _summary("frame_update_interval", freeze_ms, "ms")
    lag_ms = lag["p95"] * 1000.0 / args.source_hz
    print(f"estimated_state_lag_p95={lag_ms:.2f}ms")
    return (
        frame["p95"] <= args.max_latency_ms
        and scene["p95"] <= args.max_latency_ms
        and lag_ms <= args.max_latency_ms
        and freeze["max"] <= args.max_freeze_ms
    )


def main() -> None:
    """Run one latency benchmark and exit nonzero when its thresholds are exceeded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "recording", "live"), default="synthetic")
    parser.add_argument("--robot", default="agilex_piper")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--source-hz", type=float, default=60.0)
    parser.add_argument("--poll-hz", type=float, default=12.5)
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--console-url", default="http://127.0.0.1:8080")
    parser.add_argument("--max-latency-ms", type=float, default=250.0)
    parser.add_argument("--max-freeze-ms", type=float, default=500.0)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--capture-stall", type=float, default=3.0)
    parser.add_argument("--catchup-timeout", type=float, default=5.0)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.mode == "synthetic":
        passed = _run_synthetic(args)
    elif args.mode == "recording":
        passed = _run_recording(args)
    else:
        passed = _run_live(args)
    print(f"result={'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
