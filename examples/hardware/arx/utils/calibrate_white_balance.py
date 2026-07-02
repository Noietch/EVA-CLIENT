#!/usr/bin/env python3
"""Calibrate ARX Orbbec camera white balance one camera at a time."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.hardware.arx.camera import (  # noqa: E402
    OrbbecCameraSpec,
    default_arx_orbbec_camera_specs,
    frame_to_bgr_image,
    get_orbbec_sdk,
    get_white_balance,
    merge_orbbec_camera_specs,
    parse_orbbec_camera_specs,
    parse_resolution,
    select_color_profile,
    select_orbbec_device,
    set_auto_white_balance,
)

DEFAULT_OUTPUT = Path(__file__).with_name("white_balance.yaml")
DEFAULT_IMAGE_ROOT = REPO_ROOT / "work_dirs" / "arx_orbbec_white_balance"


def parse_name_list(values: list[str]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        for item in value.split(","):
            name = item.strip()
            if name and name not in names:
                names.append(name)
    return tuple(names)


def load_disabled_cameras(config_path: str) -> tuple[str, ...]:
    if not config_path:
        return ()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"EVA config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    transport = raw.get("transport", {})
    if not isinstance(transport, dict):
        return ()
    disabled = transport.get("disabled_cameras", ())
    if disabled is None:
        return ()
    return tuple(str(name) for name in disabled)


def print_devices(sdk: Any) -> None:
    ctx = sdk.Context()
    devices = ctx.query_devices()
    for index in range(len(devices)):
        name = devices.get_device_name_by_index(index)
        serial = devices.get_device_serial_number_by_index(index)
        print(f"[{index}] {name} serial={serial} pid={devices.get_device_pid_by_index(index)}")


def build_camera_specs(args: argparse.Namespace) -> tuple[OrbbecCameraSpec, ...]:
    configured_disabled = load_disabled_cameras(args.eva_config)
    cli_disabled = parse_name_list(args.disabled_camera)
    disabled_cameras = tuple(dict.fromkeys(configured_disabled + cli_disabled))
    resolution = parse_resolution(args.orbbec_resolution)
    default_cameras = default_arx_orbbec_camera_specs(
        resolution=resolution,
        fps=int(args.orbbec_fps),
        color_format=str(args.orbbec_color_format),
        timeout_ms=int(args.orbbec_timeout_ms),
    )
    override_cameras = parse_orbbec_camera_specs(
        args.orbbec_camera,
        resolution=resolution,
        fps=int(args.orbbec_fps),
        color_format=str(args.orbbec_color_format),
        timeout_ms=int(args.orbbec_timeout_ms),
    )
    return merge_orbbec_camera_specs(
        default_cameras,
        override_cameras,
        disabled_cameras=disabled_cameras,
    )


def image_filename(image_key: str, serial: str) -> str:
    safe_serial = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in serial)
    return f"{image_key}_{safe_serial}.jpg"


def calibrate_camera(
    sdk: Any,
    spec: OrbbecCameraSpec,
    warmup_s: float,
    min_frames: int,
    image_dir: Path,
) -> dict:
    ctx = sdk.Context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No Orbbec camera is connected")

    device = select_orbbec_device(devices, spec)
    info = device.get_device_info()
    serial = info.get_serial_number()
    auto_supported = set_auto_white_balance(device, sdk, True)

    pipeline = sdk.Pipeline(device)
    stream_config = sdk.Config()
    try:
        profile = select_color_profile(sdk, pipeline, spec)
    except Exception:
        fallback = dataclasses.replace(spec, resolution=None)
        profile = select_color_profile(sdk, pipeline, fallback)
    stream_config.enable_stream(profile)

    print(f"{spec.image_key}: warming auto white balance serial={serial}")
    frames_seen = 0
    last_image = None
    center_rgb_mean: list[float] | None = None
    deadline = time.monotonic() + max(0.0, warmup_s)
    pipeline.start(stream_config)
    try:
        while frames_seen < min_frames or time.monotonic() < deadline:
            frameset = pipeline.wait_for_frames(spec.timeout_ms)
            if frameset is None:
                continue
            color_frame = frameset.get_color_frame()
            if color_frame is None:
                continue
            image = frame_to_bgr_image(color_frame, sdk)
            last_image = image
            height, width = image.shape[:2]
            center = image[
                height // 2 - height // 10 : height // 2 + height // 10,
                width // 2 - width // 10 : width // 2 + width // 10,
            ]
            bgr_mean = center.reshape(-1, 3).mean(axis=0)
            center_rgb_mean = [float(bgr_mean[2]), float(bgr_mean[1]), float(bgr_mean[0])]
            frames_seen += 1
    finally:
        pipeline.stop()

    white_balance = get_white_balance(device, sdk)
    image_path = None
    if last_image is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / image_filename(spec.image_key, serial)
        cv2.imwrite(str(image_path), last_image)
    print(
        f"{spec.image_key}: white_balance={white_balance} "
        f"frames={frames_seen} auto_supported={auto_supported}"
    )
    if image_path is not None:
        print(f"{spec.image_key}: saved image {image_path}")
    return {
        "image_key": spec.image_key,
        "serial": serial,
        "white_balance": white_balance,
        "frames": frames_seen,
        "auto_supported": auto_supported,
        "center_rgb_mean": center_rgb_mean,
        "image_path": None if image_path is None else str(image_path),
    }


def wait_for_camera_ready(spec: OrbbecCameraSpec, index: int, total: int) -> None:
    selector = spec.serial if spec.serial is not None else f"index:{spec.device_index}"
    input(
        f"[{index}/{total}] Aim {spec.image_key} ({selector}) at the white/gray reference, "
        "then press Enter..."
    )


def write_white_balance_file(path: Path, records: list[dict]) -> None:
    payload = {
        "white_balance": {
            str(record["image_key"]): int(record["white_balance"]) for record in records
        },
        "cameras": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    print(f"Wrote {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="list connected Orbbec cameras and exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="YAML file written with per-camera white-balance values.",
    )
    parser.add_argument(
        "--eva-config",
        default="",
        help="Optional EVA config; transport.disabled_cameras disables missing camera keys.",
    )
    parser.add_argument(
        "--disabled-camera",
        action="append",
        default=[],
        metavar="KEY",
        help="Disable an EVA image key; repeatable or comma-separated.",
    )
    parser.add_argument(
        "--orbbec-camera",
        action="append",
        default=[],
        metavar="KEY=SERIAL_OR_INDEX",
        help="Camera mapping, e.g. cam_high=CP053... or cam_high=index:0; repeatable.",
    )
    parser.add_argument(
        "--orbbec-resolution",
        default="default",
        help="Requested Orbbec color resolution, e.g. 640x480; default uses SDK profile.",
    )
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-color-format", default="MJPG")
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="directory for saved calibration frames; default is under work_dirs/.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="run through all cameras without waiting for Enter before each one.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        sdk = get_orbbec_sdk()
        if args.list:
            print_devices(sdk)
            return

        specs = build_camera_specs(args)
        if not specs:
            raise RuntimeError("No ARX Orbbec cameras are enabled")

        image_dir = args.image_dir
        if image_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            image_dir = DEFAULT_IMAGE_ROOT / stamp
        records = []
        for index, spec in enumerate(specs, start=1):
            if not args.no_prompt:
                wait_for_camera_ready(spec, index, len(specs))
            records.append(
                calibrate_camera(
                    sdk,
                    spec,
                    float(args.warmup_s),
                    int(args.min_frames),
                    image_dir,
                )
            )
        write_white_balance_file(args.output, records)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
