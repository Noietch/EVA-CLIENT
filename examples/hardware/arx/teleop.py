#!/usr/bin/env python3
"""Teleoperate ARX R5 arms from Alicia-D leader arms."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

DEFAULT_ARX_SDK_DIR = Path(__file__).resolve().parent / "SDK" / "ARX_R5_python"
DEFAULT_ALICIA_PORT = ""
DEFAULT_LEFT_ALICIA_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C192742-if00"
DEFAULT_RIGHT_ALICIA_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C192642-if00"
DEFAULT_LEFT_CAN_PORT = "can0"
DEFAULT_RIGHT_CAN_PORT = "can1"
DEFAULT_ARM_TYPE = 0
DEFAULT_RATE_HZ = 30.0
DEFAULT_GRIPPER_OPEN_POS = 5.0
DEFAULT_MAX_ALICIA_READ_FAILURES = 5
DEFAULT_ALICIA_READER = "sdk"
DEFAULT_ALICIA_STATE_TIMEOUT = 2.0

LOCAL_ALICIA_SDK_DIR = Path(__file__).resolve().parent / "SDK" / "alicia_d"
ROS_JAZZY_SETUP = Path("/opt/ros/jazzy/setup.sh")

ALICIA_MOTOR_DIR = np.asarray([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
ARX_R5_JOINT_LIMITS = np.asarray([[-10.0, 10.0]] * 6)
ALICIA_PORT_INCLUDE_KEYWORDS = ("usb single serial", "wch", "ch340", "ch341", "ch343")
ALICIA_PORT_EXCLUDE_KEYWORDS = ("canable",)
GROUP_NAMES = ("left_arm", "right_arm")
WRIST_JOINT_INDEX = 5
LEFT_ALICIA_WRIST_COMPENSATION_RAD = math.pi


class AliciaJointState(NamedTuple):
    angles: list[float]
    gripper: float
    timestamp: float


class AliciaDuoSerialReader:
    """Read Alicia Duo leader state using the v5.4 9-servo serial protocol."""

    FRAME_HEADER = 0xAA
    FRAME_FOOTER = 0xFF
    FRAME_BASE_SIZE = 5
    CMD_GRIPPER = 0x02
    CMD_JOINT = 0x04
    JOINT_DATA_SIZE = 18
    BAUDRATE = 921600
    GRIPPER_MIN = 2048
    GRIPPER_MAX = 3290
    SERVO_TO_JOINT = (
        (0, 1.0),
        None,
        (1, 1.0),
        None,
        (2, 1.0),
        None,
        (3, 1.0),
        (4, 1.0),
        (5, 1.0),
    )

    def __init__(self, port: str, debug: bool) -> None:
        self._port = port
        self._debug = debug
        self._serial: Any | None = None
        self._rx_buffer = bytearray()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = AliciaJointState([0.0] * 6, 0.0, 0.0)

    def connect(self) -> None:
        import serial

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self.BAUDRATE,
            timeout=0.02,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def wait_for_state(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.read_joint_state().timestamp > 0.0:
                return True
            time.sleep(0.01)
        return False

    def read_joint_state(self) -> AliciaJointState:
        with self._lock:
            state = self._state
            return AliciaJointState(list(state.angles), state.gripper, state.timestamp)

    def _update_loop(self) -> None:
        while not self._stop_event.is_set():
            serial_port = self._serial
            if serial_port is None:
                return
            waiting = serial_port.in_waiting
            if waiting <= 0:
                time.sleep(0.001)
                continue
            self._rx_buffer.extend(serial_port.read(waiting))
            self._drain_frames()

    def _drain_frames(self) -> None:
        while len(self._rx_buffer) >= self.FRAME_BASE_SIZE:
            if self._rx_buffer[0] != self.FRAME_HEADER:
                self._rx_buffer.pop(0)
                continue

            frame_length = self._rx_buffer[2] + self.FRAME_BASE_SIZE
            if len(self._rx_buffer) < frame_length:
                break

            frame = list(self._rx_buffer[:frame_length])
            del self._rx_buffer[:frame_length]
            if len(self._rx_buffer) > 1000:
                self._rx_buffer.clear()
            if frame[-1] != self.FRAME_FOOTER or not self._verify_checksum(frame):
                continue
            self._parse_frame(frame)

    def _parse_frame(self, frame: list[int]) -> None:
        cmd_id = frame[1]
        if cmd_id == self.CMD_JOINT:
            self._parse_joint_frame(frame)
        elif cmd_id == self.CMD_GRIPPER:
            self._parse_gripper_frame(frame)

    def _parse_joint_frame(self, frame: list[int]) -> None:
        if frame[2] != self.JOINT_DATA_SIZE:
            return

        joint_values = [0.0] * 6
        for servo_idx, mapping in enumerate(self.SERVO_TO_JOINT):
            if mapping is None:
                continue
            joint_idx, direction = mapping
            byte_idx = 3 + servo_idx * 2
            raw_value = (frame[byte_idx] & 0xFF) | ((frame[byte_idx + 1] & 0xFF) << 8)
            joint_values[joint_idx] = self._raw_to_radians(raw_value) * direction

        self._update_state(angles=joint_values, gripper=None)

    def _parse_gripper_frame(self, frame: list[int]) -> None:
        if len(frame) < 8:
            return

        button1 = len(frame) >= 10 and (frame[8] & 0x01) != 0
        if button1:
            raw_value = (frame[6] & 0xFF) | ((frame[7] & 0xFF) << 8)
        else:
            raw_value = (frame[4] & 0xFF) | ((frame[5] & 0xFF) << 8)

        clipped = max(self.GRIPPER_MIN, min(raw_value, self.GRIPPER_MAX))
        gripper = (clipped - self.GRIPPER_MIN) / (self.GRIPPER_MAX - self.GRIPPER_MIN)
        self._update_state(angles=None, gripper=float(gripper * 1000.0))

    def _update_state(
        self,
        angles: list[float] | None,
        gripper: float | None,
    ) -> None:
        with self._lock:
            previous = self._state
            self._state = AliciaJointState(
                angles=angles if angles is not None else previous.angles,
                gripper=gripper if gripper is not None else previous.gripper,
                timestamp=time.time(),
            )

    @staticmethod
    def _raw_to_radians(value: int) -> float:
        clipped = max(0, min(value, 4095))
        return (-180.0 + (clipped / 2048.0) * 180.0) * math.pi / 180.0

    @staticmethod
    def _verify_checksum(frame: list[int]) -> bool:
        data_len = frame[2]
        if len(frame) != data_len + AliciaDuoSerialReader.FRAME_BASE_SIZE:
            return False
        payload = frame[3 : 3 + data_len]
        return frame[3 + data_len] == sum(payload) % 2


def load_ros_runtime_environment() -> None:
    if not ROS_JAZZY_SETUP.exists() or os.environ.get("AMENT_PREFIX_PATH"):
        return

    result = subprocess.run(
        ["bash", "-c", f"source {ROS_JAZZY_SETUP} && env -0"],
        check=True,
        capture_output=True,
    )
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        key, value = item.split(b"=", 1)
        os.environ[key.decode()] = value.decode()


def prepare_arx_runtime_paths(sdk_dir: Path) -> None:
    load_ros_runtime_environment()
    sdk_dir = sdk_dir.expanduser().resolve()
    if not sdk_dir.exists():
        raise FileNotFoundError(f"ARX R5 SDK directory not found: {sdk_dir}")

    api_dir = sdk_dir / "bimanual" / "api"
    runtime_paths = [
        sdk_dir / "bimanual" / "lib",
        sdk_dir / "bimanual" / "lib" / "arx_r5_src",
        api_dir,
        api_dir / "arx_r5_src",
        api_dir / "arx_r5_python",
        Path("/opt/ros/jazzy/lib"),
        Path("/usr/local/lib"),
    ]
    runtime_path_strings = [str(path) for path in runtime_paths if path.exists()]

    current_ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    missing_ld_paths = [
        path for path in runtime_path_strings if path and path not in current_ld_paths
    ]
    if (
        missing_ld_paths
        and os.environ.get("ARX_R5_RUNTIME_REEXEC") != "1"
        and Path(sys.argv[0]).resolve() == Path(__file__).resolve()
    ):
        env = os.environ.copy()
        env["ARX_R5_RUNTIME_REEXEC"] = "1"
        env["LD_LIBRARY_PATH"] = ":".join(runtime_path_strings + current_ld_paths)
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

    for path in (sdk_dir, api_dir, api_dir / "arx_r5_src", api_dir / "arx_r5_python"):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        runtime_path_strings + ([current_ld_path] if current_ld_path else [])
    )


def prepare_alicia_runtime_path() -> None:
    if LOCAL_ALICIA_SDK_DIR.exists():
        path = str(LOCAL_ALICIA_SDK_DIR)
        if path not in sys.path:
            sys.path.insert(0, path)


def map_alicia_to_arx_joints(alicia_joints: np.ndarray) -> np.ndarray:
    """Map Alicia-D leader joints to ARX R5 follower joint targets.

    Args:
        alicia_joints: [6] Alicia-D joint angles in radians.

    Returns:
        arx_joints: [6] ARX R5 joint targets in radians.
    """
    joints = np.asarray(alicia_joints, dtype=float)
    if joints.shape[0] < 6:
        raise ValueError(f"Expected at least 6 Alicia joints, got {joints.shape}")
    arx_joints = joints[:6] * ALICIA_MOTOR_DIR
    return np.clip(arx_joints, ARX_R5_JOINT_LIMITS[:, 0], ARX_R5_JOINT_LIMITS[:, 1])


def read_arx_joints(arm: Any) -> np.ndarray:
    positions = np.asarray(arm.get_joint_positions(), dtype=float)
    if positions.shape[0] < 6:
        raise RuntimeError(f"Expected at least 6 ARX joints, got {positions.shape}")
    return positions[:6]


def calibrate_arx_joint_offset(alicia_joints: np.ndarray, arx_joints: np.ndarray) -> np.ndarray:
    """Compute the startup offset from Alicia leader joints to ARX follower joints.

    Args:
        alicia_joints: [6] Alicia-D joint angles in radians.
        arx_joints: [6] current ARX R5 joint angles in radians.

    Returns:
        offset: [6] offset added after applying ALICIA_MOTOR_DIR.
    """
    return np.asarray(arx_joints, dtype=float)[:6] - map_alicia_to_arx_joints(alicia_joints)


def wrap_angle_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def compensate_alicia_joints(
    joints: np.ndarray,
    group_name: str,
) -> np.ndarray:
    """Apply per-arm compensation at the Alicia joint-read boundary.

    Args:
        joints: [6] Alicia-D joint angles in radians.
        group_name: Alicia group name, "left_arm" or "right_arm".

    Returns:
        compensated: [6] Alicia-D joint angles with per-arm compensation applied.
    """
    compensated = np.asarray(joints, dtype=float)[:6].copy()
    if group_name == "left_arm":
        compensated[WRIST_JOINT_INDEX] = wrap_angle_pi(
            compensated[WRIST_JOINT_INDEX] + LEFT_ALICIA_WRIST_COMPENSATION_RAD
        )
    return compensated


def read_alicia_state(robot: Any, group_name: str) -> tuple[np.ndarray, float | None]:
    if hasattr(robot, "read_joint_state"):
        state = robot.read_joint_state()
    else:
        state = robot.get_robot_state("joint_gripper")
    if state is None or state.angles is None:
        raise RuntimeError("Alicia-D joint state unavailable")
    if getattr(state, "timestamp", 0.0) <= 0.0:
        raise RuntimeError("Alicia-D joint state has not been updated")
    return compensate_alicia_joints(state.angles, group_name), float(state.gripper)


def map_alicia_gripper_to_arx(raw_gripper: float, gripper_open_pos: float) -> float:
    scalar = np.clip(float(raw_gripper) / 1000.0, 0.0, 1.0)
    return float(scalar * gripper_open_pos)


def format_joints(joints: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in joints) + "]"


def list_alicia_port_candidates() -> list[str]:
    import serial.tools.list_ports

    candidates: list[tuple[str, str, bool]] = []
    for item in serial.tools.list_ports.comports():
        text = " ".join(
            str(value)
            for value in (
                item.device,
                item.description,
                item.hwid,
                item.manufacturer,
                item.product,
            )
            if value
        ).lower()
        if any(keyword in text for keyword in ALICIA_PORT_EXCLUDE_KEYWORDS):
            continue
        if "ttyacm" not in text and "ttyusb" not in text and "com" not in text:
            continue
        preferred = any(keyword in text for keyword in ALICIA_PORT_INCLUDE_KEYWORDS)
        candidates.append((str(item.device), text, preferred))

    return [
        device
        for device, _text, _preferred in sorted(
            candidates,
            key=lambda candidate: (not candidate[2], candidate[0]),
        )
    ]


def resolve_alicia_port(port: str) -> str:
    if port:
        return port

    candidates = list_alicia_port_candidates()
    if len(candidates) == 1:
        return candidates[0]

    devices = ", ".join(candidates) or "none"
    raise RuntimeError(
        "Could not determine Alicia-D serial port. "
        f"Candidates after filtering CAN adapters: {devices}. "
        "Pass --alicia-port explicitly."
    )


def resolve_alicia_group_ports(
    left_port: str,
    right_port: str,
    legacy_port: str,
    candidates: list[str] | None = None,
) -> dict[str, str]:
    right = right_port or legacy_port
    left = left_port
    if left and right:
        if left == right:
            raise RuntimeError("Left and right Alicia ports must be different.")
        return {"left_arm": left, "right_arm": right}

    available = candidates if candidates is not None else list_alicia_port_candidates()
    used = {port for port in (left, right) if port}
    remaining = [port for port in available if port not in used]

    if not left and not right:
        if len(available) < 2:
            devices = ", ".join(available) or "none"
            raise RuntimeError(
                "Expected two Alicia-D serial ports for dual-arm teleop. "
                f"Candidates after filtering CAN adapters: {devices}. "
                "Pass --left-alicia-port and --right-alicia-port explicitly."
            )
        right = available[0]
        left = available[1]
    elif not left:
        if len(remaining) != 1:
            devices = ", ".join(remaining) or "none"
            raise RuntimeError(
                "Could not determine left Alicia-D serial port. "
                f"Remaining candidates: {devices}. "
                "Pass --left-alicia-port explicitly."
            )
        left = remaining[0]
    elif not right:
        if len(remaining) != 1:
            devices = ", ".join(remaining) or "none"
            raise RuntimeError(
                "Could not determine right Alicia-D serial port. "
                f"Remaining candidates: {devices}. "
                "Pass --right-alicia-port explicitly."
            )
        right = remaining[0]

    if left == right:
        raise RuntimeError("Left and right Alicia ports must be different.")
    return {"left_arm": left, "right_arm": right}


def create_alicia_robot(port: str, debug: bool, reader: str) -> Any:
    if reader == "duo":
        robot = AliciaDuoSerialReader(resolve_alicia_port(port), debug)
        robot.connect()
        if not robot.wait_for_state(DEFAULT_ALICIA_STATE_TIMEOUT):
            robot.disconnect()
            raise RuntimeError("Timed out waiting for Alicia Duo joint state")
        return robot

    prepare_alicia_runtime_path()
    import alicia_d_sdk

    return alicia_d_sdk.create_robot(port=resolve_alicia_port(port), debug_mode=debug)


def create_arx_arm(sdk_dir: Path, can_port: str, arm_type: int) -> Any:
    prepare_arx_runtime_paths(sdk_dir)
    from bimanual import SingleArm
    from bimanual.config import LEFT_ARM_CONFIG

    arm_config = dict(LEFT_ARM_CONFIG)
    arm_config["can_port"] = can_port
    arm_config["type"] = arm_type
    arm_config["num_joints"] = 7
    return SingleArm(arm_config)


def resolve_arx_can_ports(
    left_can_port: str,
    right_can_port: str,
    legacy_can_port: str,
) -> dict[str, str]:
    return {
        "left_arm": left_can_port,
        "right_arm": legacy_can_port or right_can_port,
    }


def run_teleop(args: argparse.Namespace) -> None:
    if not args.dry_run:
        prepare_arx_runtime_paths(args.arx_sdk_dir)
    alicia_ports = resolve_alicia_group_ports(
        args.left_alicia_port,
        args.right_alicia_port,
        args.alicia_port,
    )
    can_ports = resolve_arx_can_ports(
        args.left_can_port,
        args.right_can_port,
        args.can_port,
    )
    alicia_arms: dict[str, Any] = {}
    arx_arms: dict[str, Any] = {}
    period = 1.0 / max(args.rate, 1e-6)
    next_tick = time.monotonic()
    next_print = next_tick

    try:
        alicia_joints: dict[str, np.ndarray] = {}
        alicia_grippers: dict[str, float | None] = {}
        arx_joints: dict[str, np.ndarray] = {}
        joint_offsets: dict[str, np.ndarray] = {}
        consecutive_alicia_read_failures = dict.fromkeys(GROUP_NAMES, 0)

        for group_name in GROUP_NAMES:
            alicia_arms[group_name] = create_alicia_robot(
                alicia_ports[group_name],
                args.debug_alicia,
                args.alicia_reader,
            )
            alicia_joints[group_name], alicia_grippers[group_name] = read_alicia_state(
                alicia_arms[group_name],
                group_name,
            )
            if not args.dry_run:
                arx_arms[group_name] = create_arx_arm(
                    args.arx_sdk_dir,
                    can_ports[group_name],
                    args.arm_type,
                )
                joint_offsets[group_name] = calibrate_arx_joint_offset(
                    alicia_joints[group_name],
                    read_arx_joints(arx_arms[group_name]),
                )
            else:
                joint_offsets[group_name] = np.zeros(6, dtype=float)

        while True:
            for group_name in GROUP_NAMES:
                try:
                    alicia_joints[group_name], alicia_grippers[group_name] = read_alicia_state(
                        alicia_arms[group_name],
                        group_name,
                    )
                    consecutive_alicia_read_failures[group_name] = 0
                except Exception as exc:
                    consecutive_alicia_read_failures[group_name] += 1
                    print(
                        f"{group_name} Alicia-D read failed; holding last command "
                        f"({consecutive_alicia_read_failures[group_name]}/"
                        f"{args.max_alicia_read_failures}): {exc}"
                    )
                    if (
                        consecutive_alicia_read_failures[group_name]
                        >= args.max_alicia_read_failures
                    ):
                        raise

                arx_joints[group_name] = np.clip(
                    map_alicia_to_arx_joints(alicia_joints[group_name]) + joint_offsets[group_name],
                    ARX_R5_JOINT_LIMITS[:, 0],
                    ARX_R5_JOINT_LIMITS[:, 1],
                )

                arx_arm = arx_arms[group_name] if group_name in arx_arms else None
                if arx_arm is not None:
                    arx_arm.set_joint_positions(np.array(arx_joints[group_name], dtype=float))
                    if not args.no_gripper:
                        alicia_gripper = alicia_grippers[group_name]
                        if alicia_gripper is not None:
                            arx_arm.set_catch_pos(
                                map_alicia_gripper_to_arx(
                                    alicia_gripper,
                                    args.gripper_open_pos,
                                )
                            )

            now = time.monotonic()
            if args.print_interval > 0 and now >= next_print:
                print(
                    "left_alicia="
                    f"{format_joints(alicia_joints['left_arm'])} "
                    f"left_arx={format_joints(arx_joints['left_arm'])} "
                    "right_alicia="
                    f"{format_joints(alicia_joints['right_arm'])} "
                    f"right_arx={format_joints(arx_joints['right_arm'])}"
                )
                next_print = now + args.print_interval

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    finally:
        for arx_arm in arx_arms.values():
            arx_arm.protect_mode()
        for alicia in alicia_arms.values():
            alicia.disconnect()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alicia-port",
        default=DEFAULT_ALICIA_PORT,
        help="Legacy alias for --right-alicia-port.",
    )
    parser.add_argument("--left-alicia-port", default=DEFAULT_LEFT_ALICIA_PORT)
    parser.add_argument("--right-alicia-port", default=DEFAULT_RIGHT_ALICIA_PORT)
    parser.add_argument(
        "--can-port",
        default="",
        help="Legacy alias for --right-can-port.",
    )
    parser.add_argument("--left-can-port", default=DEFAULT_LEFT_CAN_PORT)
    parser.add_argument("--right-can-port", default=DEFAULT_RIGHT_CAN_PORT)
    parser.add_argument("--arm-type", type=int, default=DEFAULT_ARM_TYPE)
    parser.add_argument("--arx-sdk-dir", type=Path, default=DEFAULT_ARX_SDK_DIR)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--gripper-open-pos", type=float, default=DEFAULT_GRIPPER_OPEN_POS)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-alicia", action="store_true")
    parser.add_argument(
        "--alicia-reader",
        choices=("duo", "sdk"),
        default=DEFAULT_ALICIA_READER,
        help=(
            "Alicia state reader: 'duo' matches the v5.4 Alicia Duo teleop "
            "example; 'sdk' uses local alicia_d_sdk."
        ),
    )
    parser.add_argument(
        "--max-alicia-read-failures",
        type=int,
        default=DEFAULT_MAX_ALICIA_READ_FAILURES,
    )
    parser.add_argument("--print-interval", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_teleop(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
