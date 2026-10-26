"""EVA Client entry point — parses CLI arguments and launches the web application."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SRC_ROOT.parent
for _path in (_REPOSITORY_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.app.run import run  # noqa: E402
from core.config import ConfigDict, load_config  # noqa: E402
from core.devices import REPOSITORY_ROOT, DeviceWorkspace  # noqa: E402


def _configure_eval_service(
    config: ConfigDict, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Turn an existing deploy preset into one externally controlled eval service."""
    if not args.eval_service:
        return
    required = {
        "--obs-endpoint": args.obs_endpoint,
        "--action-endpoint": args.action_endpoint,
        "--control-port": args.control_port,
        "--eval-output-dir": args.eval_output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"--eval-service requires {', '.join(missing)}")
    if not 1 <= args.policy_port <= 65535 or not 1 <= args.control_port <= 65535:
        parser.error("policy and control ports must be within 1..65535")

    policy = ConfigDict(
        type=args.policy_type,
        host=args.policy_host,
        port=args.policy_port,
        backend_options={},
    )
    if args.policy_type == "xpolicylab":
        is_eef = config.inference_cfg.action_space.is_eef()
        policy.backend_options = ConfigDict(
            action_type="ee" if is_eef else "joint",
            arm_dims=[7, 7] if is_eef else [6, 6],
            gripper_dims=[1, 1],
            request_timeout_s=120.0,
            connect_timeout_s=10.0,
            handshake_timeout_s=30.0,
        )

    config.policy = copy.deepcopy(policy)
    config.transport.update(
        type="zmq",
        sub_endpoint=args.obs_endpoint,
        pub_endpoint=args.action_endpoint,
        image_mode="on_demand",
        convert_bgr_to_rgb=False,
        resize=False,
        image_layout="hwc",
        wait_action_applied=True,
        action_ack_timeout_s=30.0,
    )
    config.control_channel = ConfigDict(enabled=True, port=args.control_port)
    config.rollout.storage.enabled = False
    # The evaluator owns termination by task success, step budget, and trial timeout.
    config.inference_cfg.max_debug_time_s = float("inf")
    active = copy.deepcopy(config)
    active.eval_cfg = None
    active.eval = None
    eval_cfg = ConfigDict(
        task_set_dir="",
        result_models=[],
        storage=ConfigDict(enabled=False),
        output_dir=args.eval_output_dir,
        trials_per_prompt=5,
        cli_mode="real",
        reset_after_each_trial=False,
        inference_strategy="sync",
        skip_warmup_after_first=False,
        checkpoints=[
            ConfigDict(
                name=args.eval_name,
                config=active,
                host=args.policy_host,
                port=args.policy_port,
            )
        ],
        tasks=[],
        shuffle_ckpts=False,
        shuffle_seed=42,
        enable_ssh_forward=False,
    )
    config.eval_cfg = eval_cfg
    config.eval = eval_cfg


def parse_args() -> tuple[ConfigDict, int, str | None, bool]:
    """Parse CLI arguments into an ConfigDict plus runtime options.

    Loads the ConfigDict from --config.

    Returns:
        Tuple of (config, web_port, config_path, headless), where config_path is the
        --config value or None.
    """
    parser = argparse.ArgumentParser(
        description="EVA - Unified robot VLA inference debugging client"
    )
    parser.add_argument("--config", type=str, help="Path to business workflow configuration")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port (default 8080)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Low-power mode: no web server; control over the ZMQ channel, status via logs",
    )
    parser.add_argument(
        "--language",
        "--locale",
        choices=("zh", "en"),
        default="en",
        help="Console language: zh or en (default: en)",
    )
    parser.add_argument(
        "--eval-service",
        action="store_true",
        help="Run a deploy preset as an externally controlled evaluation Client",
    )
    parser.add_argument("--policy-type", choices=("openpi", "xpolicylab"), default="openpi")
    parser.add_argument("--policy-host", default="127.0.0.1", help="Policy service host")
    parser.add_argument("--policy-port", type=int, default=9000, help="Policy service port")
    parser.add_argument("--obs-endpoint", help="SIM observation ZMQ endpoint")
    parser.add_argument("--action-endpoint", help="SIM action ZMQ endpoint")
    parser.add_argument("--control-port", type=int, help="Evaluation control-channel port")
    parser.add_argument("--eval-output-dir", help="Client evaluation output directory")
    parser.add_argument("--eval-name", default="eva_sim", help="Evaluation model label")
    args = parser.parse_args()

    import robots  # noqa: F401

    config = load_config(
        args.config or REPOSITORY_ROOT / "configs/00_base/defaults.py",
        workspace=DeviceWorkspace(),
    )
    if not args.config:
        config.console.initial_tab = "manual"
    _configure_eval_service(config, args, parser)
    config.console.language = args.language

    return config, args.web_port, args.config, args.headless


def main() -> None:
    """Configure logging, parse CLI arguments, and launch the web application."""
    if sys.argv[1:2] == ["device"]:
        from core.devices.cli import main as device_main

        device_main(sys.argv[2:])
        return
    logging.addLevelName(logging.WARNING, "WARN")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    config, web_port, config_path, headless = parse_args()
    language = str((config.get("console") or {}).get("language", "en"))
    if run(
        config, web_port=web_port, config_path=config_path, headless=headless, language=language
    ):
        os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
