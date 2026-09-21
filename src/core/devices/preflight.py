"""Validate and prepare robot CAN before a local launcher opens motor hardware."""

import subprocess
from pathlib import Path

CAN_INTERFACE_TYPE = 280
CAN_PREPARE_TIMEOUT_S = 120
ARX_X5_ARM_OPTIONS = (("left_arm", "--left-can-port"), ("right_arm", "--right-can-port"))
ARX_X5_ARM_ALIASES = {"left": "left_arm", "right": "right_arm"}


def _read_can_interface(channel: str, net_root: Path) -> tuple[int, int]:
    """Return the (type, flags) of a CAN interface, or raise when it is unreadable."""
    root = net_root / channel
    try:
        interface_type = int((root / "type").read_text().strip())
        flags = int((root / "flags").read_text().strip(), 16)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read CAN interface {channel}; check the adapter connection"
        ) from error
    return interface_type, flags


def _option_values(command: list[str], option: str) -> list[str]:
    """Collect every value passed for a repeatable ``--option`` flag."""
    values = []
    for index, argument in enumerate(command):
        if argument == option and index + 1 < len(command):
            values.append(command[index + 1])
        elif argument.startswith(f"{option}="):
            values.append(argument.split("=", 1)[1])
    return values


def _invalid_interface_name(channel: str) -> bool:
    return not channel or Path(channel).name != channel or channel in {".", ".."}


def _prepare_can(script: Path, channels: list[str]) -> None:
    """Run a robot's CAN preparation script, which must never prompt for a password."""
    try:
        result = subprocess.run(
            ["bash", str(script), *dict.fromkeys(channels)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=CAN_PREPARE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("CAN initialization timed out; robot was not started") from error
    if result.returncode:
        raise ValueError(result.stderr.strip()[-1000:] or "CAN initialization failed")


def check_yam_can(
    command: list[str],
    *,
    net_root: Path = Path("/sys/class/net"),
    prepare: bool = False,
) -> None:
    if command[1:3] != ["-m", "examples.hardware.yam.node"] or "--camera-only" in command:
        return
    channels = []
    for value in _option_values(command, "--follower-can"):
        channels.append(value.rsplit("=", 1)[-1])
    if not channels:
        raise ValueError("YAM startup requires explicit follower CAN channels")
    down = []
    for channel in channels:
        if _invalid_interface_name(channel):
            raise ValueError("Invalid follower CAN interface name")
        interface_type, flags = _read_can_interface(channel, net_root)
        if interface_type != CAN_INTERFACE_TYPE:
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
    _prepare_can(script, down)
    check_yam_can(command, net_root=net_root)


def check_arx_x5_can(
    command: list[str],
    *,
    net_root: Path = Path("/sys/class/net"),
    prepare: bool = False,
) -> None:
    """Validate and prepare the SLCAN interfaces the X5 arms open.

    Unlike the YAM adapters, a CANable interface does not exist until the preparation
    script binds it with slcand, so an absent interface is a state to prepare rather
    than a wiring error to report.
    """
    if command[1:3] != ["-m", "examples.hardware.arx_x5.node"] or "--camera-only" in command:
        return
    disabled = set()
    for value in _option_values(command, "--disabled-arm"):
        for item in value.split(","):
            name = item.strip()
            if name:
                disabled.add(ARX_X5_ARM_ALIASES.get(name, name))
    channels = []
    for arm, option in ARX_X5_ARM_OPTIONS:
        if arm not in disabled:
            channels.extend(_option_values(command, option))
    if not channels:
        return
    silent = []
    for channel in channels:
        if _invalid_interface_name(channel):
            raise ValueError("Invalid X5 CAN interface name")
        if not (net_root / channel).exists():
            silent.append(channel)
            continue
        interface_type, flags = _read_can_interface(channel, net_root)
        if interface_type != CAN_INTERFACE_TYPE:
            raise ValueError(f"{channel} is not a CAN interface")
        if not flags & 1:
            silent.append(channel)
    if not silent:
        return
    if not prepare:
        raise ValueError(
            f"CAN interface {silent[0]} is DOWN; configure the robot CAN bus before Start"
        )
    script = Path(__file__).resolve().parents[3] / "examples/hardware/arx_x5/prepare_can.sh"
    _prepare_can(script, silent)
    check_arx_x5_can(command, net_root=net_root)
