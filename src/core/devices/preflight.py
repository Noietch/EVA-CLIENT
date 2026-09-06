"""Validate and prepare follower CAN before a local launcher opens motor hardware."""

import subprocess
from pathlib import Path


def check_yam_can(
    command: list[str],
    *,
    net_root: Path = Path("/sys/class/net"),
    prepare: bool = False,
) -> None:
    if command[1:3] != ["-m", "examples.hardware.yam.node"] or "--camera-only" in command:
        return
    channels = []
    for index, argument in enumerate(command):
        if argument == "--follower-can" and index + 1 < len(command):
            channels.append(command[index + 1].rsplit("=", 1)[-1])
    if not channels:
        raise ValueError("YAM startup requires explicit follower CAN channels")
    down = []
    for channel in channels:
        if not channel or Path(channel).name != channel or channel in {".", ".."}:
            raise ValueError("Invalid follower CAN interface name")
        root = net_root / channel
        try:
            interface_type = int((root / "type").read_text().strip())
            flags = int((root / "flags").read_text().strip(), 16)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot read CAN interface {channel}; check the adapter connection"
            ) from error
        if interface_type != 280:
            raise ValueError(f"{channel} is not a CAN interface")
        if not flags & 1:
            down.append(channel)
    if not down:
        return
    if not prepare:
        raise ValueError(
            f"CAN interface {down[0]} is DOWN; configure the robot CAN bus before Start"
        )
    script = Path(__file__).resolve().parents[3] / "examples/hardware/yam/prepare_can.sh"
    try:
        result = subprocess.run(
            ["bash", str(script), *dict.fromkeys(down)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("CAN initialization timed out; robot was not started") from error
    if result.returncode:
        raise ValueError(result.stderr.strip()[-1000:] or "CAN initialization failed")
    check_yam_can(command, net_root=net_root)
