#!/usr/bin/env python3
"""Evaluate one recorded episode through a configured policy client.

Use this tool for offline policy inspection: it loads the robot, LeRobot
dataset, policy backend, optional task-local model server, and output settings
from one EVA config. Predictions are aligned with recorded actions by chunk and
written as trajectory.jsonl, summary.json, and open_loop.png. The configured
server is terminated on success or failure, and an existing output directory is
never overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import policy_client  # noqa: F401
import robots  # noqa: F401
from core.app.handlers.imaging import prepare_image
from core.config import ConfigDict, load_config
from core.registry import POLICY_REGISTRY, ROBOT_REGISTRY
from policy_client.base import PolicyBuildContext, PolicyConnectionError
from transport.dataset import DatasetTransport

logger = logging.getLogger(__name__)


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _stop(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _start_server(config: ConfigDict, prompt: str, output: Path) -> subprocess.Popen[Any] | None:
    server = config.openloop.get("server")
    if not server:
        return None
    port = _free_port()
    config.policy.port = port
    command = [str(part).format(prompt=prompt, port=port) for part in server.command]
    environment = dict(os.environ)
    environment.update({str(key): str(value) for key, value in server.get("env", {}).items()})
    with (output / "server.log").open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            command,
            cwd=_path(server.get("cwd", ".")),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _connect_policy(config: ConfigDict, process: subprocess.Popen[Any] | None):
    timeout = float(config.openloop.get("startup_timeout_s", 120.0))
    deadline = time.monotonic() + timeout
    context = PolicyBuildContext(retry_until_connected=False)
    while True:
        if process is not None and process.poll() is not None:
            log = Path(config.openloop.output_dir) / "server.log"
            detail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"policy server exited with {process.returncode}:\n{detail}")
        try:
            return POLICY_REGISTRY.build(config.policy.type, config.policy, context)
        except PolicyConnectionError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"policy server was not ready within {timeout:.0f}s") from None
            time.sleep(1.0)


def _action_names(config: ConfigDict, dimension: int) -> list[str]:
    info = json.loads((_path(config.transport.dataset_dir) / "meta/info.json").read_text())
    feature = info.get("features", {}).get(config.transport.dataset_keys.action_key, {})
    names = feature.get("names")
    if isinstance(names, list) and len(names) == dimension:
        return [str(name) for name in names]
    return [f"action.{index}" for index in range(dimension)]


def _plot(target: np.ndarray, prediction: np.ndarray, labels: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    columns = 2
    rows = math.ceil(target.shape[1] / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, rows * 2.8), sharex=True)
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        if index >= target.shape[1]:
            axis.set_visible(False)
            continue
        axis.plot(target[:, index], label="GT", linewidth=1.5)
        axis.plot(prediction[:, index], label="prediction", linewidth=1.2)
        axis.set_title(labels[index])
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _write_results(
    config: ConfigDict,
    output: Path,
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    chunks: list[dict[str, Any]],
    metadata: dict,
) -> None:
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    error = prediction - target
    labels = _action_names(config, target.shape[1])
    with (output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for step, (predicted, expected) in enumerate(zip(prediction, target, strict=True)):
            stream.write(
                json.dumps(
                    {
                        "step": step,
                        "prediction": predicted.tolist(),
                        "ground_truth": expected.tolist(),
                    }
                )
                + "\n"
            )
    summary = {
        "dataset": str(_path(config.transport.dataset_dir)),
        "episode": int(config.transport.episode_id),
        "policy_type": str(config.policy.type),
        "client": "policy_client.xpolicylab.XPolicyLabPolicyClient",
        "server_metadata": metadata,
        "steps": int(target.shape[0]),
        "action_dim": int(target.shape[1]),
        "chunks": chunks,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "per_dim_mae": np.abs(error).mean(axis=0).tolist(),
        "finite": bool(np.isfinite(prediction).all()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _plot(target, prediction, labels, output / "open_loop.png")


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    config.transport.dataset_dir = str(_path(config.transport.dataset_dir))
    config.openloop.output_dir = str(_path(config.openloop.output_dir))
    output = Path(config.openloop.output_dir)
    output.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(output / "client.log"), logging.StreamHandler()],
    )

    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = DatasetTransport(
        config,
        robot,
        config.transport.dataset_dir,
        config.transport.episode_id,
    )
    process = _start_server(config, transport.current_task, output)
    client = None
    try:
        client = _connect_policy(config, process)
        client.reset()
        targets = transport.get_action_trajectory()
        limit = min(transport.n_steps, int(config.openloop.get("max_steps") or transport.n_steps))
        execute_horizon = config.openloop.get("execute_horizon")
        predictions: list[np.ndarray] = []
        expected: list[np.ndarray] = []
        chunks: list[dict[str, Any]] = []
        step = 0
        while step < limit:
            transport.seek(step)
            frame = transport.get_frame()
            if frame is None or frame.state_qpos is None:
                raise RuntimeError(f"dataset has no observation at step {step}")
            images = {
                name: prepare_image(
                    image,
                    config.transport.convert_bgr_to_rgb,
                    config.transport.image_height,
                    config.transport.image_width,
                    resize_pad=config.transport.resize_pad,
                    image_layout=config.transport.image_layout,
                )
                for name, image in frame.images.items()
            }
            observation = robot.build_observation(images, frame.state_qpos, transport.current_task)
            started = time.perf_counter()
            actions = np.asarray(client.infer(observation)["actions"], dtype=np.float32)
            elapsed = time.perf_counter() - started
            count = min(
                actions.shape[0],
                actions.shape[0] if execute_horizon is None else int(execute_horizon),
                limit - step,
            )
            if count <= 0:
                raise RuntimeError("policy returned an empty action chunk")
            predictions.append(actions[:count])
            expected.append(targets[step : step + count])
            chunks.append({"start": step, "count": count, "latency_s": elapsed})
            step += count
            logger.info("open-loop progress %d/%d, last chunk %.2fs", step, limit, elapsed)
        _write_results(config, output, predictions, expected, chunks, client.metadata)
    finally:
        transport.close()
        if client is not None and hasattr(client, "close"):
            client.close()
        _stop(process)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    output = run(args.config)
    print(output / "open_loop.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
