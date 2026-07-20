#!/usr/bin/env python3
"""Run R1 Lite fake-node end-to-end checks against EVA's ROS2 path."""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_cdp import (  # noqa: E402
    Cdp,
    human_click,
    human_scrub,
    human_select_first,
    start_chrome,
    wait_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "pack up a smart phone"


class E2EPaths(NamedTuple):
    base_dir: Path
    config_dir: Path
    deploy_config: Path
    rollout_hil_config: Path
    hil_config: Path
    collection_config: Path
    collection_no_hil_config: Path
    eval_config: Path
    rl_config: Path
    rollout_no_hil_dir: Path
    rollout_hil_dir: Path
    hil_dir: Path
    collection_hil_dir: Path
    collection_no_hil_dir: Path
    eval_output_dir: Path
    rl_dir: Path

    @property
    def eval_dir(self) -> Path:
        return self.eval_output_dir


class ManagedProcess:
    """Small subprocess wrapper that writes logs and stops the whole process group."""

    def __init__(self, name: str, command: Sequence[str], env: dict[str, str], log_path: Path):
        self.name = name
        self.command = list(command)
        self.log_path = log_path
        self._log_file = log_path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    @property
    def returncode(self) -> int | None:
        return self._proc.poll()

    def assert_running(self) -> None:
        code = self.returncode
        if code is not None:
            raise RuntimeError(f"{self.name} exited with code {code}; log={self.log_path}")

    def stop(self) -> None:
        if self.returncode is None:
            os.killpg(self._proc.pid, signal.SIGINT)
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, signal.SIGTERM)
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                    self._proc.wait(timeout=3.0)
        self._log_file.close()


def e2e_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    env["PYTHONPATH"] = f"{root}:{root}/src:{env.get('PYTHONPATH', '')}"
    env["ROS_DOMAIN_ID"] = "77"
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "ROS_DISCOVERY_SERVER",
        "RMW_IMPLEMENTATION",
    ):
        env.pop(key, None)
    env["ROS_LOCALHOST_ONLY"] = "1"
    return env


def write_e2e_configs(
    base_dir: Path,
    policy_port: int,
    critic_port: int = 19100,
) -> E2EPaths:
    base_dir = base_dir.resolve()
    config_dir = base_dir / "configs"
    rollout_no_hil_dir = base_dir / "rollout_no_hil"
    rollout_hil_dir = base_dir / "rollout_hil"
    hil_dir = base_dir / "hil"
    collection_hil_dir = base_dir / "collection_hil"
    collection_no_hil_dir = base_dir / "collection_no_hil"
    eval_output_dir = base_dir / "eval"
    rl_dir = base_dir / "rl"
    config_dir.mkdir(parents=True, exist_ok=True)
    rollout_no_hil_dir.mkdir(parents=True, exist_ok=True)
    rollout_hil_dir.mkdir(parents=True, exist_ok=True)
    hil_dir.mkdir(parents=True, exist_ok=True)
    collection_hil_dir.mkdir(parents=True, exist_ok=True)
    collection_no_hil_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)

    deploy_base = REPO_ROOT / "configs/01_deploy/r1lite/openpi_qpos.py"
    collection_base = REPO_ROOT / "configs/02_collection/r1lite.py"
    eval_base = REPO_ROOT / "configs/03_evaluation/r1lite_eval.py"

    deploy_config = config_dir / "r1lite_fake_deploy.py"
    deploy_config.write_text(
        f"""_base_ = ['{deploy_base}']

policy = dict(port={policy_port})

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir='{rollout_no_hil_dir}',
        fps=15,
        save_queue_max=15,
        async_save=False,
        image_height=360,
        image_width=640,
    ),
    intervention=dict(control_mode='relative'),
)

operator_control = dict(enabled=True, button_topic='/eva/operator_button')

inference_cfg = dict(publish_rate=15, setup_warmup_chunks=1)
""",
        encoding="utf-8",
    )

    rollout_hil_config = config_dir / "r1lite_fake_rollout_hil.py"
    rollout_hil_config.write_text(
        f"""_base_ = ['{deploy_base}']

policy = dict(port={policy_port})

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir='{rollout_hil_dir}',
        fps=15,
        save_queue_max=15,
        async_save=False,
        image_height=360,
        image_width=640,
    ),
    intervention=dict(control_mode='relative'),
)

operator_control = dict(enabled=True, button_topic='/eva/operator_button')

inference_cfg = dict(publish_rate=15, setup_warmup_chunks=1)
""",
        encoding="utf-8",
    )

    hil_config = config_dir / "r1lite_fake_hil.py"
    hil_config.write_text(
        f"""_base_ = ['{rollout_hil_config}']

rollout = dict(
    storage=dict(
        log_dir='{hil_dir}',
    ),
    intervention=dict(control_mode='relative'),
)
""",
        encoding="utf-8",
    )

    collection_config = config_dir / "r1lite_fake_collection.py"
    collection_config.write_text(
        f"""_base_ = ['{collection_base}']

policy = dict(port={policy_port})

collection = dict(
    storage=dict(
        log_dir='{collection_hil_dir}',
        fps=15,
        save_queue_max=15,
        image_height=360,
        image_width=640,
    ),
)

rollout = dict(
    intervention=dict(control_mode='relative'),
)

inference_cfg = dict(publish_rate=15, setup_warmup_chunks=1)
""",
        encoding="utf-8",
    )

    collection_no_hil_config = config_dir / "r1lite_fake_collection_no_hil.py"
    collection_no_hil_config.write_text(
        f"""_base_ = ['{collection_config}']

collection = dict(
    storage=dict(
        log_dir='{collection_no_hil_dir}',
    ),
)

rollout = dict(
    intervention=dict(control_mode='relative'),
)
""",
        encoding="utf-8",
    )

    eval_config = config_dir / "r1lite_fake_eval.py"
    eval_config.write_text(
        f"""_base_ = ['{eval_base}']

eval_cfg = dict(
    output_dir='{eval_output_dir}',
    storage=dict(
        fps=15,
        save_queue_max=15,
        image_height=360,
        image_width=640,
    ),
    trials_per_prompt=1,
    cli_mode='real',
    inference_strategy='sync',
    reset_after_each_trial=False,
    skip_warmup_after_first=True,
    checkpoints=[
        dict(
            name='fake_policy',
            config='{deploy_base}',
            port={policy_port},
        ),
    ],
    shuffle_ckpts=False,
    shuffle_seed=42,
    enable_ssh_forward=False,
    tasks=[
        dict(
            prompt_en='{PROMPT}',
            milestones=(('done', 'fake episode completed'),),
        ),
    ],
)
""",
        encoding="utf-8",
    )

    rl_config = config_dir / "r1lite_fake_rl.py"
    rl_config.write_text(
        f"""_base_ = ['{deploy_base}']

rl_cfg = dict(
    cli_mode='real',
    inference_strategy='sync',
    tasks=['{PROMPT}'],
    policies=[
        dict(
            name='fake_policy',
            config='{deploy_config}',
            host='127.0.0.1',
            port={policy_port},
        ),
    ],
    critics=[
        dict(
            name='fake_critic',
            type='websocket',
            host='127.0.0.1',
            port={critic_port},
            backend_options={{}},
        ),
    ],
    data=dict(
        format='lerobot',
        storage=dict(
            log_dir='{rl_dir}',
            fps=15,
            save_queue_max=15,
            async_save=True,
            image_height=360,
            image_width=640,
        ),
    ),
    intervention=dict(control_mode='relative'),
)
""",
        encoding="utf-8",
    )

    return E2EPaths(
        base_dir=base_dir,
        config_dir=config_dir,
        deploy_config=deploy_config,
        rollout_hil_config=rollout_hil_config,
        hil_config=hil_config,
        collection_config=collection_config,
        collection_no_hil_config=collection_no_hil_config,
        eval_config=eval_config,
        rl_config=rl_config,
        rollout_no_hil_dir=rollout_no_hil_dir,
        rollout_hil_dir=rollout_hil_dir,
        hil_dir=hil_dir,
        collection_hil_dir=collection_hil_dir,
        collection_no_hil_dir=collection_no_hil_dir,
        eval_output_dir=eval_output_dir,
        rl_dir=rl_dir,
    )


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def get_json(base_url: str, path: str) -> dict[str, Any]:
    return http_json("GET", f"{base_url}{path}")


def post_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return http_json("POST", f"{base_url}{path}", payload or {})


def wait_for_http(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            get_json(base_url, "/api/status")
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.RemoteDisconnected,
        ):
            time.sleep(0.2)
    raise TimeoutError(f"EVA did not answer at {base_url}")


def wait_for_fake_ui(fake_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            get_json(fake_url, "/api/state")
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.RemoteDisconnected,
        ):
            time.sleep(0.2)
    raise TimeoutError(f"fake node UI did not answer at {fake_url}")


def wait_status_value(
    base_url: str,
    key: str,
    expected: Any,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_json(base_url, "/api/status")
        if last.get(key) == expected:
            return last
        time.sleep(0.2)
    raise TimeoutError(f"status[{key!r}] did not become {expected!r}; last={last}")


def poll_status_for(base_url: str, duration_s: float, interval_s: float = 0.5) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        get_json(base_url, "/api/status")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(interval_s, remaining))


def wait_for_file(path: Path, timeout_s: float) -> Path:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return path
        time.sleep(0.2)
    raise TimeoutError(f"file not written: {path}")


def wait_for_latest_parquet(root: Path, timeout_s: float) -> Path:
    deadline = time.monotonic() + timeout_s
    last: list[Path] = []
    while time.monotonic() < deadline:
        last = sorted(root.glob("**/data/chunk-000/episode_*.parquet"))
        if last:
            return last[-1]
        time.sleep(0.3)
    raise TimeoutError(f"no episode parquet under {root}; last={last}")


def latest_episode_row(dataset_root: Path) -> dict[str, Any]:
    candidates = sorted(dataset_root.glob("**/meta/episodes.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no episodes.jsonl under {dataset_root}")
    path = candidates[-1]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise AssertionError(f"empty episodes.jsonl: {path}")
    return rows[-1]


def wait_latest_episode_row(dataset_root: Path, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return latest_episode_row(dataset_root)
        except (FileNotFoundError, AssertionError) as exc:
            last_error = exc
            time.sleep(0.2)
    if last_error is not None:
        raise last_error
    raise TimeoutError(f"no episode metadata under {dataset_root}")


def assert_fixed_clock(table: pa.Table, fps: int) -> None:
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if timestamps.size < 2:
        raise AssertionError("need at least two timestamps to validate fixed-clock output")
    expected = 1.0 / float(fps)
    diffs = np.diff(timestamps)
    if not np.allclose(diffs, expected, atol=1e-6):
        raise AssertionError(
            f"timestamp is not fixed-clock: expected dt={expected}, "
            f"min={float(diffs.min())}, max={float(diffs.max())}"
        )


def table_vector(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def assert_vector_varies(
    values: np.ndarray,
    *,
    columns: Sequence[int],
    label: str,
    min_std: float = 1e-4,
) -> None:
    if values.ndim != 2:
        raise AssertionError(f"{label} must be 2-D, got shape={values.shape}")
    selected = values[:, list(columns)]
    if float(selected.std()) <= min_std:
        raise AssertionError(f"{label} did not vary in columns {tuple(columns)}")


def drive_hil(fake_url: str, repeats: int, delay_s: float) -> None:
    for _ in range(repeats):
        post_json(fake_url, "/api/joint_delta", {"group": "left_arm", "index": 2, "delta": 0.03})
        post_json(
            fake_url,
            "/api/joint_delta",
            {"group": "right_arm", "index": 3, "delta": -0.02},
        )
        time.sleep(delay_s)


def drive_motion_target(fake_url: str, repeats: int, delay_s: float) -> None:
    for _ in range(repeats):
        post_json(
            fake_url,
            "/api/command_delta",
            {"group": "left_arm", "index": 2, "delta": 0.03},
        )
        post_json(
            fake_url,
            "/api/command_delta",
            {"group": "right_arm", "index": 3, "delta": -0.02},
        )
        time.sleep(delay_s)


def validate_rollout(dataset_root: Path) -> dict[str, Any]:
    parquet = wait_for_latest_parquet(dataset_root, 30.0)
    table = pq.read_table(parquet)
    assert_fixed_clock(table, 15)
    state = table_vector(table, "observations.state.qpos")
    action_key = "action" if "action" in table.column_names else "action.qpos"
    action = table_vector(table, action_key)
    if state.shape[1] != 14 or action.shape[1] != 14:
        raise AssertionError(
            f"rollout qpos shape mismatch: state={state.shape} action={action.shape}"
        )
    assert_vector_varies(action, columns=(2, 10), label="rollout action")
    return {"path": str(parquet), "rows": table.num_rows, "action_std": float(action.std())}


def validate_collection(dataset_root: Path) -> dict[str, Any]:
    parquet = wait_for_latest_parquet(dataset_root, 45.0)
    table = pq.read_table(parquet)
    row = wait_latest_episode_row(dataset_root, 30.0)
    assert_fixed_clock(table, 15)
    state = table_vector(table, "observations.state.qpos")
    action = table_vector(table, "action.qpos")
    if state.shape[1] != 14 or action.shape[1] != 14:
        raise AssertionError(
            f"collection qpos shape mismatch: state={state.shape} action={action.shape}"
        )
    if row.get("quality") != "green":
        raise AssertionError(
            f"collection quality is {row.get('quality')}: {row.get('quality_issues')}"
        )
    assert_vector_varies(action, columns=(2, 10), label="collection action.qpos")
    return {
        "path": str(parquet),
        "rows": table.num_rows,
        "quality": row.get("quality"),
        "image_skew": row.get("alignment_image_max_skew_sec"),
        "action_std": float(action.std()),
    }


def validate_eval(paths: E2EPaths, clip_id: str) -> dict[str, Any]:
    dataset_root = paths.eval_output_dir / "fake_policy" / "episodes"
    parquet = wait_for_latest_parquet(dataset_root, 45.0)
    table = pq.read_table(parquet)
    row = wait_latest_episode_row(dataset_root, 30.0)
    assert_fixed_clock(table, 15)
    if row.get("clip_id") != clip_id:
        raise AssertionError(f"eval clip_id mismatch: {row.get('clip_id')} != {clip_id}")
    return {"path": str(parquet), "rows": table.num_rows, "clip_id": row.get("clip_id")}


def start_fake_node(paths: E2EPaths, env: dict[str, str], ui_port: int) -> ManagedProcess:
    env = dict(env, PUBLISH_RATE="30", UI_PORT=str(ui_port), IMAGE_HEIGHT="360", IMAGE_WIDTH="640")
    return ManagedProcess(
        "fake-node",
        ["bash", "examples/hardware/r1_lite/run_fake_node.sh", "--no-open-ui"],
        env,
        paths.base_dir / "fake-node.log",
    )


def start_fake_policy(paths: E2EPaths, env: dict[str, str], policy_port: int) -> ManagedProcess:
    return ManagedProcess(
        "fake-policy",
        [
            sys.executable,
            "examples/fake_policy/fake_policy_server.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(policy_port),
            "--chunk-size",
            "50",
            "--action-dim",
            "14",
            "--action-mode",
            "qpos",
        ],
        env,
        paths.base_dir / "fake-policy.log",
    )


def start_fake_critic(paths: E2EPaths, env: dict[str, str], critic_port: int) -> ManagedProcess:
    return ManagedProcess(
        "fake-critic",
        [
            sys.executable,
            "examples/fake_critic/fake_critic_server.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(critic_port),
            "--delay-ms",
            "80",
        ],
        env,
        paths.base_dir / "fake-critic.log",
    )


def start_eva(
    paths: E2EPaths,
    env: dict[str, str],
    config_path: Path,
    web_port: int,
    name: str,
) -> ManagedProcess:
    return ManagedProcess(
        name,
        [sys.executable, "src/main.py", "--config", str(config_path), "--web-port", str(web_port)],
        env,
        paths.base_dir / f"{name}.log",
    )


def stop_process(proc: ManagedProcess | None) -> None:
    if proc is not None:
        proc.stop()


def run_rollout(
    paths: E2EPaths,
    env: dict[str, str],
    web_port: int,
    fake_url: str,
    *,
    config_path: Path,
    dataset_root: Path,
    name: str,
) -> dict[str, Any]:
    proc = start_eva(paths, env, config_path, web_port, name)
    eva_url = f"http://127.0.0.1:{web_port}"
    try:
        wait_for_http(eva_url, 30.0)
        post_json(eva_url, "/api/connect")
        post_json(eva_url, "/api/select_mode", {"mode": "real"})
        post_json(eva_url, "/api/select_strategy", {"strategy": "sync"})
        post_json(eva_url, "/api/select_task", {"task": PROMPT})
        post_json(eva_url, "/api/setup")
        wait_status_value(eva_url, "session_status", "ready", 45.0)
        post_json(eva_url, "/api/run")
        wait_status_value(eva_url, "session_status", "running", 15.0)
        poll_status_for(eva_url, 4.0)
        post_json(eva_url, "/api/rollout_stop")
        wait_status_value(eva_url, "session_status", "ready", 15.0)
        status = get_json(eva_url, "/api/status")
        if not status.get("rollout", {}).get("save_ready"):
            raise AssertionError(f"rollout is not save_ready after stop: {status.get('rollout')}")
        post_json(eva_url, "/api/rollout_save")
        return validate_rollout(dataset_root)
    finally:
        stop_process(proc)
        _ = fake_url


def run_collection(
    paths: E2EPaths,
    env: dict[str, str],
    web_port: int,
    fake_url: str,
    *,
    config_path: Path,
    dataset_root: Path,
    name: str,
    command_source: str,
) -> dict[str, Any]:
    proc = start_eva(paths, env, config_path, web_port, name)
    eva_url = f"http://127.0.0.1:{web_port}"
    try:
        wait_for_http(eva_url, 30.0)
        post_json(eva_url, "/api/tab_switch", {"tab": "collect", "collect_teleop_armed": True})
        time.sleep(1.0)
        post_json(fake_url, "/api/operator", {"button": "x"})
        wait_status_value(eva_url, "session_status", "running", 15.0)
        if command_source == "hil":
            drive_hil(fake_url, repeats=30, delay_s=0.1)
        elif command_source == "motion_target":
            drive_motion_target(fake_url, repeats=30, delay_s=0.1)
        else:
            raise ValueError(f"unsupported collection command_source {command_source!r}")
        post_json(fake_url, "/api/gripper", {"side": "right", "command": "close"})
        time.sleep(0.5)
        post_json(fake_url, "/api/gripper", {"side": "right", "command": "open"})
        time.sleep(0.5)
        post_json(fake_url, "/api/operator", {"button": "y"})
        wait_status_value(eva_url, "session_status", "ready", 15.0)
        return validate_collection(dataset_root)
    finally:
        stop_process(proc)


def run_hil_intervention(
    paths: E2EPaths,
    env: dict[str, str],
    web_port: int,
    fake_url: str,
) -> dict[str, Any]:
    proc = start_eva(paths, env, paths.hil_config, web_port, "eva-hil")
    eva_url = f"http://127.0.0.1:{web_port}"
    try:
        wait_for_http(eva_url, 30.0)
        post_json(eva_url, "/api/connect")
        post_json(eva_url, "/api/select_mode", {"mode": "real"})
        post_json(eva_url, "/api/select_strategy", {"strategy": "sync"})
        post_json(eva_url, "/api/select_task", {"task": PROMPT})
        post_json(eva_url, "/api/setup")
        wait_status_value(eva_url, "session_status", "ready", 45.0)
        post_json(eva_url, "/api/run")
        wait_status_value(eva_url, "session_status", "running", 15.0)
        poll_status_for(eva_url, 3.0)
        post_json(fake_url, "/api/operator", {"button": "x"})
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = get_json(eva_url, "/api/status")
            if status.get("rollout_intervention_active"):
                break
            time.sleep(0.2)
        else:
            raise AssertionError("rollout intervention did not become active after fake X")
        post_json(fake_url, "/api/sync_hil_to_feedback")
        drive_hil(fake_url, repeats=20, delay_s=0.1)
        post_json(fake_url, "/api/operator", {"button": "y"})
        wait_status_value(eva_url, "session_status", "running", 30.0)
        poll_status_for(eva_url, 2.0)
        post_json(eva_url, "/api/rollout_stop")
        wait_status_value(eva_url, "session_status", "ready", 15.0)
        status = get_json(eva_url, "/api/status")
        segments = int(status.get("rollout_intervention_segments") or 0)
        if segments < 1:
            raise AssertionError(f"no accepted intervention segment: {status}")
        post_json(eva_url, "/api/rollout_save")
        summary = validate_rollout(paths.hil_dir)
        summary["segments"] = segments
        return summary
    finally:
        stop_process(proc)


def run_eval(paths: E2EPaths, env: dict[str, str], web_port: int) -> dict[str, Any]:
    proc = start_eva(paths, env, paths.eval_config, web_port, "eva-eval")
    eva_url = f"http://127.0.0.1:{web_port}"
    clip_id = "fake-clip-001"
    try:
        wait_for_http(eva_url, 45.0)
        post_json(eva_url, "/api/tab_switch", {"tab": "eval"})
        post_json(eva_url, "/api/eval_start", {"clip_id": clip_id, "prompt": PROMPT, "trial": 1})
        wait_status_value(eva_url, "session_status", "running", 45.0)
        poll_status_for(eva_url, 5.0)
        post_json(eva_url, "/api/eval_stop")
        wait_status_value(eva_url, "session_status", "ready", 30.0)
        return validate_eval(paths, clip_id)
    finally:
        stop_process(proc)


def wait_status(
    base_url: str,
    predicate,
    description: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_json(base_url, "/api/status")
        if predicate(last):
            return last
        time.sleep(0.1)
    raise TimeoutError(f"status did not reach {description}; last={last}")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise AssertionError("performance sample is empty")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_rl_browser(
    paths: E2EPaths,
    env: dict[str, str],
    web_port: int,
    fake_url: str,
    headless: bool,
) -> dict[str, Any]:
    """Drive the complete RL flow through hit-tested Chrome mouse/keyboard input."""
    proc = start_eva(paths, env, paths.rl_config, web_port, "eva-rl-browser")
    eva_url = f"http://127.0.0.1:{web_port}"
    chrome: subprocess.Popen[Any] | None = None
    cdp: Cdp | None = None
    profile: Path | None = None
    try:
        wait_for_http(eva_url, 30.0)
        chrome, cdp, profile = start_chrome(headless=headless)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                  window.__rlE2EErrors = [];
                  window.addEventListener("error", (event) => {
                    window.__rlE2EErrors.push(String(event.message || event.error));
                  });
                  window.addEventListener("unhandledrejection", (event) => {
                    window.__rlE2EErrors.push(String(event.reason));
                  });
                """
            },
        )
        cdp.call("Page.navigate", {"url": eva_url})
        wait_for(cdp, 'document.readyState === "complete"', 30.0)
        wait_for(cdp, '!!document.querySelector(\'.tab[data-tab="rl"]\')', 10.0)

        human_click(cdp, '.tab[data-tab="rl"]')
        wait_status(
            eva_url,
            lambda status: bool(status.get("rl", {}).get("active")),
            "RL active",
            10.0,
        )
        wait_for(cdp, 'document.querySelector("#view-rl").classList.contains("active")')

        human_select_first(cdp, "#rl-task-list")
        wait_status(
            eva_url,
            lambda status: status.get("selected_task") == PROMPT,
            "task selected",
            10.0,
        )
        human_select_first(cdp, "#rl-policy-list")
        wait_status(
            eva_url,
            lambda status: status.get("rl", {}).get("selected_policy_slot") == 0,
            "Policy selected",
            10.0,
        )
        human_select_first(cdp, "#rl-critic-list")
        wait_status(
            eva_url,
            lambda status: status.get("rl", {}).get("selected_critic_slot") == 0,
            "Critic selected",
            10.0,
        )
        wait_for(cdp, '!document.querySelector("#rl-b-setup").disabled', 10.0)

        setup_started = time.monotonic()
        human_click(cdp, "#rl-b-setup")
        setup_status = wait_status(
            eva_url,
            lambda status: bool(status.get("is_setup_done"))
            and bool(status.get("policy_connected"))
            and bool(status.get("rl", {}).get("critic_connected")),
            "RL setup complete",
            60.0,
        )
        setup_ms = (time.monotonic() - setup_started) * 1000.0
        wait_for(cdp, '!document.querySelector("#rl-b-run").disabled', 10.0)
        wait_for(cdp, 'document.querySelectorAll("#rl-gripper-buttons .grip-sw").length > 0')
        human_click(cdp, '#rl-gripper-buttons .grip-sw[data-grip="l"]')

        human_click(cdp, "#rl-hil-gate")
        wait_status(
            eva_url,
            lambda status: bool(status.get("rollout_intervention_enabled")),
            "HIL enabled",
            10.0,
        )

        critic_before_run = int(setup_status.get("rl", {}).get("critic_samples") or 0)
        run_started = time.monotonic()
        human_click(cdp, "#rl-b-run")
        wait_status_value(eva_url, "session_status", "running", 20.0)
        run_click_ms = (time.monotonic() - run_started) * 1000.0
        curve_started = time.monotonic()
        wait_status(
            eva_url,
            lambda status: int(status.get("rl", {}).get("critic_samples") or 0)
            > critic_before_run
            and int(status.get("rl", {}).get("policy_samples") or 0) >= 4,
            "live Policy and Critic samples",
            20.0,
        )
        curve_refresh_ms = (time.monotonic() - curve_started) * 1000.0
        wait_for(cdp, 'document.querySelector("#critic-latest").textContent !== "—"', 10.0)

        intervention_started = time.monotonic()
        wait_for(cdp, '!document.querySelector("#rl-b-intervene").disabled', 10.0)
        human_click(cdp, "#rl-b-intervene")
        wait_status(
            eva_url,
            lambda status: bool(status.get("rollout_intervention_active")),
            "first intervention active",
            15.0,
        )
        intervene_click_ms = (time.monotonic() - intervention_started) * 1000.0
        post_json(fake_url, "/api/sync_hil_to_feedback")
        drive_hil(fake_url, repeats=18, delay_s=0.08)
        first_intervention = wait_status(
            eva_url,
            lambda status: int(status.get("rl", {}).get("intervention_samples") or 0) >= 6
            and int(status.get("rollout_intervention_active_frames") or 0) >= 6,
            "first intervention samples",
            15.0,
        )
        human_click(cdp, "#rl-b-accept")
        wait_status_value(eva_url, "session_status", "running", 30.0)
        wait_status(
            eva_url,
            lambda status: not status.get("rollout_intervention_active")
            and int(status.get("rollout_intervention_segments") or 0) >= 1,
            "accepted intervention",
            10.0,
        )

        wait_for(cdp, '!document.querySelector("#rl-b-intervene").disabled', 10.0)
        human_click(cdp, "#rl-b-intervene")
        wait_status(
            eva_url,
            lambda status: bool(status.get("rollout_intervention_active")),
            "second intervention active",
            15.0,
        )
        post_json(fake_url, "/api/sync_hil_to_feedback")
        drive_hil(fake_url, repeats=8, delay_s=0.08)
        wait_for(cdp, '!document.querySelector("#rl-b-abandon").disabled', 10.0)
        human_click(cdp, "#rl-b-abandon")
        abandoned = wait_status(
            eva_url,
            lambda status: status.get("session_status") == "running"
            and not status.get("rollout_intervention_active"),
            "abandoned intervention resumed",
            30.0,
        )
        if int(abandoned.get("rollout_intervention_segments") or 0) != 1:
            raise AssertionError("ABANDON unexpectedly kept the second intervention segment")

        wait_for(cdp, '!document.querySelector("#rl-hil-enable").disabled', 10.0)
        human_click(cdp, "#rl-hil-gate")
        wait_status(
            eva_url,
            lambda status: not status.get("rollout_intervention_enabled"),
            "HIL disabled",
            10.0,
        )
        wait_for(cdp, '!document.querySelector("#rl-b-intervene").disabled', 10.0)
        human_click(cdp, "#rl-b-intervene")
        stopped = wait_status(
            eva_url,
            lambda status: status.get("session_status") == "ready"
            and bool(status.get("rollout", {}).get("save_ready")),
            "rollout stopped and save-ready",
            20.0,
        )
        if int(stopped.get("rollout_intervention_segments") or 0) < 1:
            raise AssertionError("accepted HIL segment was lost before save")

        wait_for(cdp, '!document.querySelector("#rl-b-save").disabled', 10.0)
        cdp.eval("window.__rlSaveFrameStart = performance.now();")
        save_started = time.monotonic()
        human_click(cdp, "#rl-b-save")
        save_frame_ms = float(
            cdp.eval(
                "new Promise((resolve) => requestAnimationFrame(() => "
                "resolve(performance.now() - window.__rlSaveFrameStart)))"
            )
        )
        saved_status = wait_status(
            eva_url,
            lambda status: len(status.get("rollout", {}).get("episodes") or []) >= 1
            and not (status.get("rollout", {}).get("queue") or []),
            "saved RL episode",
            60.0,
        )
        save_complete_ms = (time.monotonic() - save_started) * 1000.0
        wait_for(cdp, '!!document.querySelector("#rl-save-tiles .collect-tile.replayable")', 15.0)

        # Create a second saved episode so a real browser click can switch tracks
        # while the first one is already playing.
        human_click(cdp, "#rl-b-run")
        wait_status_value(eva_url, "session_status", "running", 20.0)
        poll_status_for(eva_url, 2.0)
        human_click(cdp, "#rl-b-intervene")
        wait_status(
            eva_url,
            lambda status: status.get("session_status") == "ready"
            and bool(status.get("rollout", {}).get("save_ready")),
            "second rollout stopped and save-ready",
            20.0,
        )
        wait_for(cdp, '!document.querySelector("#rl-b-save").disabled', 10.0)
        human_click(cdp, "#rl-b-save")
        second_saved_status = wait_status(
            eva_url,
            lambda status: len(status.get("rollout", {}).get("episodes") or []) >= 2
            and not (status.get("rollout", {}).get("queue") or []),
            "second RL episode",
            60.0,
        )
        wait_for(
            cdp,
            'document.querySelectorAll("#rl-save-tiles .collect-tile.replayable").length >= 2',
            15.0,
        )

        first_episode = int(saved_status["rollout"]["episodes"][-1]["episode_index"])
        second_episode = int(second_saved_status["rollout"]["episodes"][-1]["episode_index"])
        human_click(cdp, "#rl-save-tiles .collect-tile.replayable:nth-of-type(1)")
        wait_status(
            eva_url,
            lambda status: status.get("rl", {}).get("replay_episode") is not None
            and int(status["rl"]["replay_episode"]) == first_episode,
            "first RL replay selected",
            20.0,
        )

        replay_started = time.monotonic()
        wait_for(cdp, '!!document.querySelector("#cam-strip .cam-cell.video-ready")', 20.0)
        replay_first_frame_ms = (time.monotonic() - replay_started) * 1000.0
        replay_status = wait_status(
            eva_url,
            lambda status: int(status.get("rl", {}).get("critic_samples") or 0) >= 1,
            "replay Critic sample",
            20.0,
        )
        wait_for(cdp, 'document.querySelector("#critic-latest").textContent !== "—"', 10.0)
        replay_critic_before_scrub = int(replay_status["rl"]["critic_samples"])
        human_scrub(cdp, "#scrub-range", 0.72)
        scrub_status = wait_status(
            eva_url,
            lambda status: int(status.get("rl", {}).get("critic_samples") or 0)
            > replay_critic_before_scrub,
            "scrub Critic refresh",
            20.0,
        )
        wait_for(
            cdp,
            'document.querySelector("#scrub-range").classList.contains("has-intervention")',
        )

        switch_started = time.monotonic()
        human_click(cdp, "#rl-save-tiles .collect-tile.replayable:nth-of-type(2)")
        switched_status = wait_status(
            eva_url,
            lambda status: status.get("rl", {}).get("replay_episode") is not None
            and int(status["rl"]["replay_episode"]) == second_episode,
            "second RL replay selected",
            20.0,
        )
        wait_for(cdp, '!!document.querySelector("#cam-strip .cam-cell.video-ready")', 20.0)
        replay_switch_ms = (time.monotonic() - switch_started) * 1000.0
        replay_sync = cdp.eval("window.__evaReplaySync")
        if replay_sync["expectedVideos"] < 3 or replay_sync["maxReadyVideos"] < 3:
            raise AssertionError(f"not all three replay videos became ready: {replay_sync}")
        if replay_sync["maxVideoSkewSec"] > 0.12:
            raise AssertionError(f"replay video streams lost sync: {replay_sync}")
        if replay_sync["maxUrdfFrameSkew"] > 1.0:
            raise AssertionError(f"replay URDF lost frame sync: {replay_sync}")
        if replay_sync["maxFrameGapMs"] > 250.0:
            raise AssertionError(f"replay playback stalled: {replay_sync}")

        validation = validate_rollout(paths.rl_dir)
        table = pq.read_table(validation["path"])
        control_sources = [str(value) for value in table["control_source"].to_pylist()]
        if "policy" not in control_sources or "intervention" not in control_sources:
            raise AssertionError(f"saved RL control sources are incomplete: {set(control_sources)}")

        resource_metrics = cdp.eval(
            """
            (() => {
              const rows = performance.getEntriesByType("resource");
              const durations = (needle) => rows
                .filter((row) => new URL(row.name).pathname.includes(needle))
                .map((row) => row.duration);
              return {
                status: durations("/api/status"),
                rl: durations("/api/rl/"),
              };
            })()
            """
        )
        status_p95_ms = percentile(resource_metrics["status"], 95)
        rl_api_p95_ms = percentile(resource_metrics["rl"], 95)
        critic_request_ms = float(replay_status["rl"]["critic_last_request_ms"])
        critic_coalesced = int(scrub_status["rl"]["coalesced_requests"])
        if critic_coalesced < 1:
            raise AssertionError("slow fake Critic did not exercise pending-request coalescing")
        browser_errors = cdp.eval("window.__rlE2EErrors")
        if browser_errors:
            raise AssertionError(f"browser errors: {browser_errors}")

        thresholds = {
            "run_click_ms": (run_click_ms, 1000.0),
            "intervene_click_ms": (intervene_click_ms, 1000.0),
            "curve_refresh_ms": (curve_refresh_ms, 3000.0),
            "save_frame_ms": (save_frame_ms, 250.0),
            "replay_first_frame_ms": (replay_first_frame_ms, 5000.0),
            "replay_switch_ms": (replay_switch_ms, 5000.0),
            "status_p95_ms": (status_p95_ms, 250.0),
            "rl_api_p95_ms": (rl_api_p95_ms, 250.0),
            "critic_request_ms": (critic_request_ms, 500.0),
        }
        failures = {
            name: {"actual_ms": actual, "limit_ms": limit}
            for name, (actual, limit) in thresholds.items()
            if actual > limit
        }
        if failures:
            raise AssertionError(f"RL browser performance thresholds exceeded: {failures}")

        screenshot_path = paths.base_dir / "rl-browser-e2e.png"
        screenshot = cdp.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        screenshot_path.write_bytes(base64.b64decode(screenshot["data"]))
        return {
            "setup_ms": setup_ms,
            "run_click_ms": run_click_ms,
            "intervene_click_ms": intervene_click_ms,
            "curve_refresh_ms": curve_refresh_ms,
            "save_frame_ms": save_frame_ms,
            "save_complete_ms": save_complete_ms,
            "replay_first_frame_ms": replay_first_frame_ms,
            "replay_switch_ms": replay_switch_ms,
            "replay_sync": replay_sync,
            "status_p95_ms": status_p95_ms,
            "rl_api_p95_ms": rl_api_p95_ms,
            "critic_request_ms": critic_request_ms,
            "critic_coalesced": critic_coalesced,
            "policy_samples": int(first_intervention["rl"]["policy_samples"]),
            "intervention_samples": int(first_intervention["rl"]["intervention_samples"]),
            "saved_rows": validation["rows"],
            "saved_episode": saved_status["rollout"]["episodes"][-1]["episode_index"],
            "replay_episode": switched_status["rl"]["replay_episode"],
            "accepted_segments": 1,
            "abandoned_segments": 1,
            "browser_errors": browser_errors,
            "screenshot": str(screenshot_path),
        }
    finally:
        if cdp is not None:
            cdp.close()
        if chrome is not None:
            chrome.terminate()
            chrome.wait(timeout=10.0)
        if profile is not None:
            shutil.rmtree(profile)
        stop_process(proc)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    paths = write_e2e_configs(Path(args.base_dir), args.policy_port, args.critic_port)
    env = e2e_env()
    fake_url = f"http://127.0.0.1:{args.fake_ui_port}"
    fake_node = start_fake_node(paths, env, args.fake_ui_port)
    fake_policy = start_fake_policy(paths, env, args.policy_port)
    fake_critic = start_fake_critic(paths, env, args.critic_port)
    summary: dict[str, Any] = {"artifacts": str(paths.base_dir)}
    try:
        wait_for_fake_ui(fake_url, 30.0)
        fake_node.assert_running()
        fake_policy.assert_running()
        fake_critic.assert_running()
        if not args.only or args.only == "rl":
            summary["rl"] = run_rl_browser(
                paths,
                env,
                args.web_port,
                fake_url,
                args.headless,
            )
        if not args.only or args.only == "rollout":
            summary["rollout_no_hil"] = run_rollout(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.deploy_config,
                dataset_root=paths.rollout_no_hil_dir,
                name="eva-rollout-no-hil",
            )
            summary["rollout_hil"] = run_rollout(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.rollout_hil_config,
                dataset_root=paths.rollout_hil_dir,
                name="eva-rollout-hil",
            )
        if args.only == "rollout_no_hil":
            summary["rollout_no_hil"] = run_rollout(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.deploy_config,
                dataset_root=paths.rollout_no_hil_dir,
                name="eva-rollout-no-hil",
            )
        if args.only == "rollout_hil":
            summary["rollout_hil"] = run_rollout(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.rollout_hil_config,
                dataset_root=paths.rollout_hil_dir,
                name="eva-rollout-hil",
            )
        if not args.only or args.only == "collect":
            summary["collect_hil"] = run_collection(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.collection_config,
                dataset_root=paths.collection_hil_dir,
                name="eva-collection-hil",
                command_source="hil",
            )
            summary["collect_no_hil"] = run_collection(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.collection_no_hil_config,
                dataset_root=paths.collection_no_hil_dir,
                name="eva-collection-no-hil",
                command_source="motion_target",
            )
        if args.only == "collect_hil":
            summary["collect_hil"] = run_collection(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.collection_config,
                dataset_root=paths.collection_hil_dir,
                name="eva-collection-hil",
                command_source="hil",
            )
        if args.only == "collect_no_hil":
            summary["collect_no_hil"] = run_collection(
                paths,
                env,
                args.web_port,
                fake_url,
                config_path=paths.collection_no_hil_config,
                dataset_root=paths.collection_no_hil_dir,
                name="eva-collection-no-hil",
                command_source="motion_target",
            )
        if not args.only or args.only == "hil":
            summary["hil"] = run_hil_intervention(paths, env, args.web_port, fake_url)
        if not args.only or args.only == "eval":
            summary["eval"] = run_eval(paths, env, args.web_port)
        return summary
    finally:
        stop_process(fake_critic)
        stop_process(fake_policy)
        stop_process(fake_node)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="/tmp/eva-r1lite-fake-e2e")
    parser.add_argument("--web-port", type=int, default=18080)
    parser.add_argument("--policy-port", type=int, default=19000)
    parser.add_argument("--critic-port", type=int, default=19100)
    parser.add_argument("--fake-ui-port", type=int, default=18765)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--only",
        choices=(
            "rollout",
            "rollout_no_hil",
            "rollout_hil",
            "collect",
            "collect_hil",
            "collect_no_hil",
            "hil",
            "eval",
            "rl",
        ),
        default="",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all(args)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    (Path(args.base_dir) / "summary.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
