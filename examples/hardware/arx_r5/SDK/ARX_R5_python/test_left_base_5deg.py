#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np


BASE_JOINT_INDEX = 0
DEFAULT_DEGREES = 5.0
DEFAULT_HOLD_SECONDS = 2.0
DEFAULT_SETTLE_SECONDS = 3.0


def prepare_runtime_paths() -> None:
    project_dir = Path(__file__).resolve().parent
    api_dir = project_dir / "bimanual" / "api"
    arx_src_dir = api_dir / "arx_r5_src"
    arx_python_dir = api_dir / "arx_r5_python"
    runtime_paths = [str(arx_src_dir), str(api_dir), "/opt/ros/jazzy/lib", "/usr/local/lib"]

    current_ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    missing_ld_paths = [path for path in runtime_paths if path and path not in current_ld_paths]
    if (
        missing_ld_paths
        and os.environ.get("ARX_R5_RUNTIME_REEXEC") != "1"
        and Path(sys.argv[0]).resolve() == Path(__file__).resolve()
    ):
        env = os.environ.copy()
        env["ARX_R5_RUNTIME_REEXEC"] = "1"
        env["LD_LIBRARY_PATH"] = ":".join(runtime_paths + current_ld_paths)
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

    for path in (arx_src_dir, api_dir, arx_python_dir):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(runtime_paths + ([current_ld_path] if current_ld_path else []))


def check_import() -> None:
    prepare_runtime_paths()
    from bimanual import SingleArm
    from bimanual.config import LEFT_ARM_CONFIG

    print("Import ok:", SingleArm.__name__)
    print("Left arm config:", LEFT_ARM_CONFIG)


def fmt(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def read_joint_positions(arm) -> List[float]:
    positions = list(arm.get_joint_positions())
    if len(positions) < 6:
        raise RuntimeError(f"Expected at least 6 joint positions, got {len(positions)}")
    return positions


def command_joint_positions(arm, positions: List[float]) -> None:
    arm.set_joint_positions(np.array(positions[:6], dtype=float))


def run_left_base_test(
    *,
    degrees: float,
    hold_seconds: float,
    settle_seconds: float,
    require_confirm: bool,
) -> None:
    prepare_runtime_paths()
    from bimanual import SingleArm
    from bimanual.config import LEFT_ARM_CONFIG

    arm_config = dict(LEFT_ARM_CONFIG)
    left_arm = SingleArm(arm_config)

    start_positions = read_joint_positions(left_arm)
    delta = math.radians(degrees)
    target_positions = start_positions.copy()
    target_positions[BASE_JOINT_INDEX] += delta

    print("Left arm CAN:", arm_config["can_port"])
    print("Start joints:", fmt(start_positions))
    print(
        f"Plan: joint1/base moves by {degrees:.2f} deg "
        f"({delta:.4f} rad), then returns to start."
    )
    print("Target joints:", fmt(target_positions))

    if require_confirm:
        confirmation = input("Type MOVE_LEFT_BASE_5DEG to execute: ").strip()
        if confirmation != "MOVE_LEFT_BASE_5DEG":
            print("Cancelled.")
            return

    try:
        print("Moving joint1/base to target...")
        command_joint_positions(left_arm, target_positions)
        time.sleep(settle_seconds)

        print("Holding target...")
        time.sleep(hold_seconds)

        print("Returning joint1/base to start...")
        command_joint_positions(left_arm, start_positions)
        time.sleep(settle_seconds)

        final_positions = read_joint_positions(left_arm)
        print("Final joints:", fmt(final_positions))
        print(
            "Final joint1 error: "
            f"{math.degrees(final_positions[BASE_JOINT_INDEX] - start_positions[BASE_JOINT_INDEX]):.3f} deg"
        )
    finally:
        left_arm.protect_mode()
        print("Protect mode requested.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move the left arm base joint by 5 degrees and return."
    )
    parser.add_argument("--execute", action="store_true", help="Send motion commands.")
    parser.add_argument("--degrees", type=float, default=DEFAULT_DEGREES)
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument(
        "--check-import",
        action="store_true",
        help="Verify Python/C++ imports without creating an arm interface.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the typed confirmation when used with --execute.",
    )
    args = parser.parse_args()

    if args.check_import:
        check_import()
        return

    if not args.execute:
        delta = math.radians(args.degrees)
        print(
            "Dry run. Re-run with --execute after the left arm area is clear. "
            f"Requested move: {args.degrees:.2f} deg ({delta:.4f} rad)."
        )
        return

    if abs(args.degrees) > 10.0:
        raise ValueError("--degrees is limited to +/-10 for this test")
    if args.hold < 0 or args.settle < 0:
        raise ValueError("--hold and --settle must be non-negative")

    run_left_base_test(
        degrees=args.degrees,
        hold_seconds=args.hold,
        settle_seconds=args.settle,
        require_confirm=not args.yes,
    )


if __name__ == "__main__":
    main()
