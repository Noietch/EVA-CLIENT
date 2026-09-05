#!/usr/bin/env python3
"""Prepare R1 Lite fake-stack configs and probe an already running local stack.

Run with --prepare-only to write isolated deployment, collection, evaluation,
and RL configs. Otherwise poll the EVA watchdog and fake-node HTTP endpoints;
--drive-motion sends bounded fake joint deltas, --dataset-dir checks recorded
Parquet timestamps and motion, and --browser checks two saved RL replays.
The caller starts the ROS2 fake node, EVA, policy, and critic with these configs.
Outputs are configs and summary.json below --output; existing datasets are read
only and no processes are stopped or launched by this tool.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from pprint import pformat
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "pack up a smart phone"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class E2EPaths(NamedTuple):
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
    eval_dir: Path
    rl_dir: Path


def write_e2e_configs(base_dir: Path, policy_port: int) -> E2EPaths:
    root = base_dir.resolve()
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    modes = (
        "rollout_no_hil",
        "rollout_hil",
        "hil",
        "collection_hil",
        "collection_no_hil",
        "eval",
        "rl",
    )
    paths = E2EPaths(
        *(config_dir / f"{mode}.py" for mode in modes),
        *(root / mode for mode in modes),
    )
    deploy = str(REPO_ROOT / "configs/01_deploy/r1lite/openpi_qpos.py")
    collection = str(REPO_ROOT / "configs/02_collection/r1lite.py")
    documents = {}
    for mode, path, output in zip(modes, paths[:7], paths[7:], strict=True):
        storage = dict(log_dir=str(output), fps=15, image_height=360, image_width=640)
        document = dict(
            _base_=[collection if mode.startswith("collection") else deploy],
            policy=dict(port=policy_port),
            inference_cfg=dict(publish_rate=15, setup_warmup_chunks=1),
            rollout=dict(intervention=dict(control_mode="relative")),
        )
        if mode.startswith("collection"):
            document["collection"] = dict(storage=storage, tasks={"fake": [(PROMPT, -1)]})
            if mode == "collection_no_hil":
                document["_base_"] = [str(paths.collection_config)]
        elif mode == "eval":
            document["_base_"] = [str(REPO_ROOT / "configs/03_evaluation/r1lite_eval.py")]
            document["eval_cfg"] = dict(
                output_dir=str(output),
                storage=storage,
                trials_per_prompt=1,
                cli_mode="real",
                inference_strategy="sync",
                checkpoints=[dict(name="fake_policy", config=deploy, port=policy_port)],
                tasks=[dict(prompt_en=PROMPT, milestones=(("done", "fake episode completed"),))],
            )
        elif mode == "rl":
            document["_base_"] = [str(REPO_ROOT / "configs/04_rl/r1lite_rl.py")]
            document["rl_cfg"] = dict(
                tasks=[PROMPT],
                inference_strategy="sync",
                policies=[
                    dict(name="fake_policy", config=deploy, host="127.0.0.1", port=policy_port)
                ],
                critics=[
                    dict(
                        name="fake_critic",
                        type="websocket",
                        host="127.0.0.1",
                        port=policy_port + 100,
                    )
                ],
                data=dict(storage=storage),
            )
        else:
            document["rollout"]["storage"] = dict(enabled=True, **storage)
        documents[path] = document

    # Write each complete preset
    for path, document in documents.items():
        path.write_text(
            "\n\n".join(
                f"{key} = {pformat(value, sort_dicts=False)}" for key, value in document.items()
            )
            + "\n"
        )
    return paths


def get_json(base_url: str, path: str) -> dict:
    with HTTP.open(base_url.rstrip("/") + path, timeout=5) as response:
        return json.load(response)


def post_json(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with HTTP.open(request, timeout=5) as response:
        return json.load(response)


def poll_status_for(base_url: str, duration_s: float, interval_s: float = 0.5) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        get_json(base_url, "/api/status")
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval_s, remaining))


def drive_motion_target(fake_url: str, repeats: int, delay_s: float) -> None:
    for _ in range(repeats):
        for group, index, delta in (("left_arm", 2, 0.03), ("right_arm", 3, -0.02)):
            post_json(fake_url, "/api/command_delta", dict(group=group, index=index, delta=delta))
        time.sleep(delay_s)


def assert_fixed_clock(table, fps: int) -> None:
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if timestamps.size < 2 or not np.isfinite(timestamps).all():
        raise AssertionError("need at least two finite timestamps")
    if not np.allclose(np.diff(timestamps), 1.0 / fps, rtol=0, atol=1e-6):
        raise AssertionError("timestamp is not fixed-clock")


def assert_vector_varies(values, *, columns, label: str, min_std: float = 1e-4) -> None:
    selected = np.asarray(values)[:, list(columns)]
    if selected.shape[0] < 2 or not np.isfinite(selected).all():
        raise AssertionError(f"{label} needs at least two finite samples")
    if not np.any(selected.std(axis=0) > min_std):
        raise AssertionError(f"{label} did not vary in columns {tuple(columns)}")


def probe_rl_replay(page, eva_url: str, duration_s: float = 2.0) -> list[dict]:
    page.goto(eva_url)
    tiles = page.locator("#rl-save-tiles .collect-tile.replayable")
    page.wait_for_function(
        'document.querySelectorAll("#rl-save-tiles .collect-tile.replayable").length >= 2'
    )
    results = []
    for index in range(2):
        with page.expect_response(
            lambda response: response.url.endswith("/api/rl/review_episode")
        ) as pending:
            tiles.nth(index).click()
        if not pending.value.ok:
            raise AssertionError("RL replay episode could not be loaded")
        page.wait_for_function("""async () => {
            const { LIVE } = await import('/js/core.js');
            return LIVE.replayMode && !LIVE.replayLoading && LIVE.playing
                && window.__evaReplaySync.samples > 0;
        }""")
        page.wait_for_timeout(duration_s * 1000)
        metrics = page.evaluate("({...window.__evaReplaySync})")
        if metrics["maxReadyVideos"] < 3:
            raise AssertionError("RL replay did not render all three cameras")
        for name, limit in (
            ("maxCameraSkewSec", 0.15),
            ("maxUrdfFrameSkew", 2),
            ("maxFrameGapMs", 500),
        ):
            if not np.isfinite(metrics[name]) or metrics[name] > limit:
                raise AssertionError(f"RL replay {name} exceeded {limit}: {metrics[name]}")
        results.append(metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-port", type=int, default=19000)
    parser.add_argument("--eva-url", default="http://127.0.0.1:18080")
    parser.add_argument("--fake-url", default="http://127.0.0.1:18765")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--drive-motion", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    paths = write_e2e_configs(args.output, args.policy_port)
    summary = {"configs": [str(path) for path in paths[:7]]}
    if not args.prepare_only:
        summary["fake"] = get_json(args.fake_url, "/api/state")
        if args.drive_motion:
            drive_motion_target(args.fake_url, repeats=2, delay_s=0.1)
        poll_status_for(args.eva_url, duration_s=1)
        summary["eva"] = get_json(args.eva_url, "/api/status")
        if args.dataset_dir:
            files = sorted(args.dataset_dir.glob("data/chunk-*/episode_*.parquet"))
            if not files:
                raise FileNotFoundError("no recorded episodes under dataset-dir")
            for path in files:
                table = pq.read_table(path)
                assert_fixed_clock(table, fps=15)
                key = "action.qpos" if "action.qpos" in table.column_names else "action"
                assert_vector_varies(table[key].to_pylist(), columns=(2, 10), label=key)
            summary["episodes"] = len(files)
        if args.browser:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    summary["replay"] = probe_rl_replay(browser.new_page(), args.eva_url)
                finally:
                    browser.close()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
