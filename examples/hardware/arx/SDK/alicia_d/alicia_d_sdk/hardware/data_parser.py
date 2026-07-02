# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import math
import time
from typing import List, Dict, Optional, NamedTuple, Union
import threading
import copy
from alicia_d_sdk.utils.logger import logger


class JointState(NamedTuple):
    """Joint state data structure."""
    angles: List[float]  # Six joint angles (radians)
    gripper: float       # Gripper value
    timestamp: float     # Timestamp (seconds)
    run_status_text: str  # Run status text


class DataParser:
    """Robot arm data parsing module."""
    # Constants
    DEG_TO_RAD = math.pi / 180.0  # Degrees to radians
    RAD_TO_DEG = 180.0 / math.pi  # Radians to degrees

    # Command IDs
    CMD_GRIPPER = 0x04     # Gripper control and travel feedback
    CMD_ZERO_POS = 0x03    # Set current position as zero
    CMD_JOINT = 0x06       # Joint angle feedback and control
    CMD_VERSION = 0x01     # Firmware version feedback
    CMD_TORQUE = 0x05      # Torque control
    CMD_ERROR = 0xEE       # Error feedback
    CMD_SELF_CHECK = 0xFE  # Machine self-check (servo health)
    GRI_MAX_50MM = 3290
    GRI_MAX_100MM = 3600
    # Data sizes
    JOINT_DATA_SIZE = 18

    def __init__(self, lock: threading.Lock, debug_mode: bool = False):
        """
        Initialize data parser.

        :param lock: Shared threading lock for concurrent access
        :param debug_mode: Whether to enable debug logging
        """
        self.debug_mode = debug_mode

        # Store latest joint state
        self._joint_states = JointState([0.0]*6, 0.0, 0.0, "idle")

        self._firmware_version: Optional[str] = None
        # Full version information dict: serial, hardware, firmware
        self._version_info: Optional[Dict[str, str]] = None
        self._lock = lock

        # Store run status from joint data
        self._run_status: Optional[int] = None
        self._run_status_text: Optional[str] = None

        # Store temperature data (in Celsius)
        self._temperature_data: Optional[List[float]] = None
        self._temperature_timestamp: Optional[float] = None

        # Store velocity data (in degrees per second)
        self._velocity_data: Optional[List[float]] = None
        self._velocity_timestamp: Optional[float] = None

        # Store gripper type data
        self._gripper_type: Optional[int] = None
        self._gripper_type_name_map = {
            0x00: "50mm",
            0x02: "100mm",
        }

        # Event-based synchronization for async data acquisition
        # Events are set when corresponding data is received and parsed
        self._version_event = threading.Event()
        self._joint_event = threading.Event()
        self._temperature_event = threading.Event()
        self._velocity_event = threading.Event()
        self._self_check_event = threading.Event()
        self._gripper_type_event = threading.Event()

        # Mapping from info type to corresponding event
        self._info_event_map = {
            "version": self._version_event,
            "joint_gripper": self._joint_event,
            "temperature": self._temperature_event,
            "velocity": self._velocity_event,
            "self_check": self._self_check_event,
            "gripper_type": self._gripper_type_event,
        }

        # Store self-check (servo health) data
        self._self_check_raw_mask: Optional[int] = None
        self._self_check_bits: Optional[List[bool]] = None
        self._self_check_timestamp: Optional[float] = None

    def parse_frame(self, frame: List[int]) -> Optional[Dict]:
        """
        Parse a full data frame.

        :param frame: Complete data frame (byte list)
        """
        cmd_id = frame[1]
        if cmd_id == self.CMD_VERSION:
            return self._parse_version_data(frame)
        elif cmd_id == self.CMD_JOINT:
            # Check function code to determine which parser to use
            func_code = frame[2]
            if func_code == 0x00:
                return self._parse_joint_data(frame)
            elif func_code == 0x01:
                return self._parse_temperature_data(frame)
            elif func_code == 0x02:
                return self._parse_velocity_data(frame)
            else:
                if self.debug_mode:
                    logger.debug(f"Unhandled function code in CMD_JOINT: 0x{func_code:02X}")
                return None
        elif cmd_id == self.CMD_GRIPPER:
            # Check function code to determine which parser to use
            func_code = frame[2]
            if func_code == 0x0E:
                return self._parse_gripper_type_data(frame)
            else:
                if self.debug_mode:
                    logger.debug(f"Unhandled function code in CMD_GRIPPER: 0x{func_code:02X}")
                return None
        elif cmd_id == self.CMD_ERROR:
            return self._parse_error_data(frame)
        elif cmd_id == self.CMD_SELF_CHECK:
            return self._parse_self_check_data(frame)
        else:
            if self.debug_mode:
                logger.debug(f"Unhandled command ID: 0x{cmd_id:02X}")
            return None

    def get_joint_state(self) -> Optional[JointState]:
        """
        Get current joint state.
        """
        with self._lock:
            js = self._joint_states
            # print("self joint states:", js)
            if js.angles is None or js.timestamp is None:
                logger.warning("Robot state has not been updated yet")
                return None
            return copy.deepcopy(self._joint_states)

    def get_info(self, info_type: str):
        """
        Unified getter for parsed information, for cooperation with high-level APIs.

        :param info_type: 'joint_gripper' | 'joint' | 'gripper' | 'version' | 'temperature' | 'velocity' | 'self_check' | 'gripper_type'
        :return: Parsed data for the given type, or None if unavailable
        """
        with self._lock:
            if info_type == "joint_gripper":
                js = self._joint_states
                if js.angles is None or js.timestamp is None:
                    return None
                return copy.deepcopy(js)
            elif info_type == "joint":
                js = self._joint_states
                if js.angles is None or js.timestamp is None:
                    return None
                return list(js.angles)
            elif info_type == "gripper":
                js = self._joint_states
                if js.angles is None or js.timestamp is None:
                    return None
                return js.gripper
            elif info_type == "version":
                return dict(self._version_info) if self._version_info else None
            elif info_type == "temperature":
                return list(self._temperature_data) if self._temperature_data else None
            elif info_type == "velocity":
                return list(self._velocity_data) if self._velocity_data else None
            elif info_type == "self_check":
                if self._self_check_raw_mask is None or self._self_check_bits is None:
                    return None
                return {
                    "raw_mask": int(self._self_check_raw_mask),
                    "bits": list(self._self_check_bits),
                    "timestamp": self._self_check_timestamp,
                }
            elif info_type == "gripper_type":
                if self._gripper_type is None:
                    return None
                return self._gripper_type_name_map.get(
                    self._gripper_type,
                    f"unknown(0x{self._gripper_type:02X})",
                )
            else:
                raise ValueError(f"Unsupported info type: {info_type}")

    def wait_for_info(self, info_type: str, timeout: float = 2.0) -> bool:
        """
        Wait for specified info type to be received and parsed.

        :param info_type: Type of information to wait for (version, joint, gripper, etc.)
        :param timeout: Maximum time to wait in seconds
        """
        if info_type not in self._info_event_map:
            raise ValueError(f"Unsupported info type: {info_type}. Supported types: {list(self._info_event_map.keys())}")

        event = self._info_event_map[info_type]
        return event.wait(timeout)

    def _update_joint_state(self,
                            angles: Optional[List[float]] = None,
                            gripper: Optional[float] = None,
                            run_status_text: Optional[str] = None):
        with self._lock:
            prev = self._joint_states
            self._joint_states = JointState(
                angles=angles if angles is not None else prev.angles,
                gripper=gripper if gripper is not None else prev.gripper,
                timestamp=time.time(),
                run_status_text=run_status_text if run_status_text is not None else prev.run_status_text
            )

    def _parse_joint_data(self, frame: List[int]) -> Dict:
        """
        Parse joint data frame.

        :param frame: Complete data frame
        """

        # 灵动系列 DATA layout:
        #   - 6 joint angles (each 2 bytes, low byte first)
        #   - 1 gripper position (2 bytes)
        #   - 1 run status (1 byte)
        #
        # Total DATA length = 6*2 + 2 + 1 = 15 bytes (LEN should be >= 0x0F, example shows 0x10).

        # Basic length check: header(1)+CMD(1)+func(1)+LEN(1)+DATA(LEN)+checksum(1)+footer(1)

        data_len = frame[3]
        expected_min_len = 4 + data_len + 2  # header+cmd+func+LEN + DATA + checksum+footer
        if len(frame) < expected_min_len:
            logger.warning(f"Joint frame length mismatch: LEN={data_len}, frame_len={len(frame)}")
            return None

        data_start = 4
        data_end = data_start + data_len
        data_bytes = frame[data_start:data_end]

        # print frame, data bytes in hex
        # print(f"frame: {' '.join(f'{b:02X}' for b in frame)}")
        # print(f"data bytes: {' '.join(f'{b:02X}' for b in data_bytes)}")

        if len(data_bytes) < 15:
            logger.warning(f"Joint DATA too short: expect ≥15 bytes, got {len(data_bytes)}")
            return None

        # Parse 6 joint angles
        joint_values: List[float] = [0.0] * 6
        joint_bytes = data_bytes
        # print joint_bytes in hex
        for i in range(6):
            start = i * 2
            joint_bytes = data_bytes[start:start + 2]
            angle_rad = self._bytes_to_radians(joint_bytes)
            joint_values[i] = angle_rad

        gripper_low = data_bytes[12]
        gripper_high = data_bytes[13]
        gripper_raw = (gripper_low & 0xFF) | ((gripper_high & 0xFF) << 8)

        # Map raw gripper value (0-1000) to percentage (0-100)
        # Hardware uses 0-1000 range, convert to 0-100 percentage
        gripper_value = int(max(0.0, min(1000, gripper_raw)))
        # Parse run status (last mandatory byte)
        run_status = data_bytes[14]
        run_status_map = {
            0x00: "idle",
            0x01: "locked",
            0x10: "sync",
            0x11: "sync_locked",
            0xE1: "overheat",
            0xE2: "overheat_protect",
        }
        run_status_text = run_status_map.get(run_status, "unknown")
        # print("run_status: 0x{:02X}".format(run_status))
        # Store run status
        with self._lock:
            self._run_status = run_status
            self._run_status_text = run_status_text

        # Update stored joint & gripper state
        self._update_joint_state(angles=joint_values, gripper=gripper_value, run_status_text=run_status_text)

        # Signal that joint state has been updated
        self._joint_event.set()

        if self.debug_mode:
            degrees = [round(rad * self.RAD_TO_DEG, 2) for rad in joint_values]
            logger.debug(
                f"Joint angles (deg): {degrees}, "
                f"gripper={gripper_value}, "
                f"run_status=0x{run_status:02X}({run_status_text})"
            )

        return {
            "type": "joint_data",
            "angles": self._joint_states.angles,
            "gripper": self._joint_states.gripper,
            "run_status": run_status,
            "run_status_text": run_status_text,
            "timestamp": self._joint_states.timestamp,
        }

    def _parse_temperature_data(self, frame: List[int]) -> Dict:
        """
        Parse temperature data frame (CMD=0x06, FUNC=0x01).

        :param frame: Complete data frame
        """
        data_len = frame[3]
        expected_min_len = 4 + data_len + 2
        if len(frame) < expected_min_len:
            logger.warning(f"Temperature frame length mismatch: LEN={data_len}, frame_len={len(frame)}")
            return None

        data_start = 4
        data_end = data_start + data_len
        data_bytes = frame[data_start:data_end]

        # Parse temperature values (each byte represents temperature in Celsius)
        temperatures = [float(byte) for byte in data_bytes]

        # Store temperature data
        with self._lock:
            self._temperature_data = temperatures
            self._temperature_timestamp = time.time()

        # Signal that temperature data has been updated
        self._temperature_event.set()

        if self.debug_mode:
            logger.debug(f"Temperature data: {temperatures}°C")

        return {
            "type": "temperature_data",
            "temperatures": temperatures,
            "timestamp": self._temperature_timestamp,
        }

    def _parse_velocity_data(self, frame: List[int]) -> Dict:
        """
        Parse velocity data frame (CMD=0x06, FUNC=0x02).

        :param frame: Complete data frame
        """
        # print frame in hex
        # print("Velocity frame:", " ".join(f"{b:02X}" for b in frame))
        data_len = frame[3]
        expected_min_len = 4 + data_len + 2
        if len(frame) < expected_min_len:
            logger.warning(f"Velocity frame length mismatch: LEN={data_len}, frame_len={len(frame)}")
            return None

        data_start = 4
        data_end = data_start + data_len
        data_bytes = frame[data_start:data_end]

        # Parse velocity values (2 bytes per servo, low byte first)
        num_servos = data_len // 2
        velocities = []
        for i in range(num_servos):
            low_byte = data_bytes[i * 2]
            high_byte = data_bytes[i * 2 + 1]
            velocity_raw = (low_byte & 0xFF) | ((high_byte & 0xFF) << 8)
            # Convert raw velocity to degrees per second
            # Note: velocity_raw can exceed the expected limit of 5000
            velocity_deg_s = self._raw_velocity_to_deg_per_sec(velocity_raw)
            velocities.append(velocity_deg_s)

        # Store velocity data
        with self._lock:
            self._velocity_data = velocities
            self._velocity_timestamp = time.time()

        # Signal that velocity data has been updated
        self._velocity_event.set()

        if self.debug_mode:
            logger.debug(f"Velocity data (deg/s): {velocities}")

        return {
            "type": "velocity_data",
            "velocities": velocities,
            "timestamp": self._velocity_timestamp,
        }

    def _parse_self_check_data(self, frame: List[int]) -> Dict:
        """
        Parse machine self-check frame (CMD=0xFE, FUNC=0x00).

        :param frame: Complete data frame
        """
        data_len = frame[3]
        expected_min_len = 4 + data_len + 2
        if len(frame) < expected_min_len:
            logger.warning(f"Self-check frame length mismatch: LEN={data_len}, frame_len={len(frame)}")
            return None

        data_start = 4
        data_end = data_start + data_len
        data_bytes = frame[data_start:data_end]

        if data_len < 2:
            logger.warning(f"Self-check DATA too short: expect ≥2 bytes, got {data_len}")
            return None

        low = data_bytes[0]
        high = data_bytes[1]
        raw_mask = (low & 0xFF) | ((high & 0xFF) << 8)
        # Decode to boolean list (LSB first), up to 16 bits to be safe
        bits: List[bool] = [(raw_mask >> i) & 0x1 == 1 for i in range(10)]

        with self._lock:
            self._self_check_raw_mask = raw_mask
            self._self_check_bits = bits
            self._self_check_timestamp = time.time()

        # Signal that self-check data has been updated
        self._self_check_event.set()

        if self.debug_mode:
            logger.debug(
                f"Self-check result: raw_mask=0x{raw_mask:04X}, "
                f"bits={bits}"
            )

        return {
            "type": "self_check_data",
            "raw_mask": raw_mask,
            "bits": bits,
            "timestamp": self._self_check_timestamp,
        }

    def _parse_gripper_type_data(self, frame: List[int]) -> Dict:
        """
        Parse gripper type data frame (CMD=0x04, FUNC=0x0E).

        :param frame: Complete data frame
        """
        data_len = frame[3]
        expected_min_len = 4 + data_len + 2
        if len(frame) < expected_min_len:
            logger.warning(f"Gripper type frame length mismatch: LEN={data_len}, frame_len={len(frame)}")
            return None

        data_start = 4
        data_end = data_start + data_len
        data_bytes = frame[data_start:data_end]

        if data_len < 1:
            logger.warning(f"Gripper type DATA too short: expect ≥1 byte, got {data_len}")
            return None

        gripper_type_raw = data_bytes[0] & 0xFF

        # Map gripper type values
        type_name = self._gripper_type_name_map.get(
            gripper_type_raw,
            f"unknown(0x{gripper_type_raw:02X})",
        )

        # Store gripper type data
        with self._lock:
            self._gripper_type = gripper_type_raw
            self._gripper_type_timestamp = time.time()

        # Signal that gripper type data has been updated
        self._gripper_type_event.set()

        if self.debug_mode:
            logger.debug(f"Gripper type: {type_name} (0x{gripper_type_raw:02X})")

        return {
            "type": "gripper_type_data",
            "gripper_type": gripper_type_raw,
            "type_name": type_name,
            "timestamp": self._gripper_type_timestamp,
        }

    def _parse_error_data(self, frame: List[int]) -> Dict:
        """
        Parse error data frame (0xEE).

        :param frame: Complete data frame
        """
        # Minimal length check
        if len(frame) < 7:
            logger.warning("Error frame too short")
            return None

        # Extract error code and parameter
        error_code = frame[3]
        error_param = frame[4]

        error_types = {
            0x00: "Header/footer or length error",
            0x01: "Checksum error",
            0x02: "Mode error",
            0x03: "Invalid ID",
        }

        error_message = error_types.get(error_code, f"Unknown error (0x{error_code:02X})")

        logger.warning(f"Device error: {error_message}, param: 0x{error_param:02X}")

        return {
            "type": "error_data",
            "error_code": error_code,
            "error_param": error_param,
            "error_message": error_message,
            "timestamp": time.time()
        }

    def _parse_version_data(self, frame: List[int]) -> Dict:
        """
        Parse version data frame (CMD=0x01).

        :param frame: Complete data frame
        """
        # Basic length check: header(1)+CMD(1)+func(1)+LEN(1)+DATA(LEN)+checksum(1)+footer(1)
        if len(frame) < 4 + frame[3] + 2:
            logger.warning(f"Version frame too short: expect ≥{4 + frame[3] + 2}, got {len(frame)}")
            return None

        data_len = frame[3]
        data_start = 4
        data_end = data_start + data_len
        data_bytes = frame[data_start:data_end]

        if data_len < 24:
            logger.warning(f"Version data length too short: expect 24, got {data_len}")
            return None

        # Split fields according to protocol
        serial_bytes = data_bytes[0:16]
        hardware_bytes = data_bytes[16:20]
        firmware_bytes = data_bytes[20:24]

        def _bytes_to_ascii(b: List[int]) -> str:
            try:
                return "".join(chr(x) for x in b).strip()
            except Exception as e:
                logger.error(f"Version ASCII parse exception: {e}")
                return ""

        def _bytes_to_decimal(b: List[int]) -> int:
            """Convert little-endian byte array to decimal integer."""
            result = 0
            for i, byte in enumerate(b):
                result |= (byte & 0xFF) << (i * 8)
            return result

        # Parse serial number as ASCII string
        serial_number = _bytes_to_ascii(serial_bytes)

        # Parse hardware and firmware versions as little-endian decimal values
        hardware_decimal = _bytes_to_decimal(hardware_bytes)
        firmware_decimal = _bytes_to_decimal(firmware_bytes)

        # Convert decimal values to version strings
        hardware_str = self._decimal_to_version_string(hardware_decimal)
        firmware_str = self._decimal_to_version_string(firmware_decimal)

        # Store firmware version (for upper-level API)
        with self._lock:
            self._firmware_version = firmware_str
            self._version_info = {
                "serial_number": serial_number,
                "hardware_version": hardware_str,
                "firmware_version": firmware_str,
            }

        # Signal that version info has been received and parsed
        self._version_event.set()

        if self.debug_mode:
            logger.debug(
                f"Version parsed: SN='{serial_number}', HW={hardware_decimal}('{hardware_str}'), FW={firmware_decimal}('{firmware_str}')"
            )

        return {
            "type": "version_data",
            "serial_number": serial_number,
            "hardware_version": hardware_str,
            "firmware_version_raw": firmware_decimal,
            "version": firmware_str,
            "timestamp": time.time(),
        }

    def _bytes_to_radians(self, byte_array: List[int]) -> float:
        """
        Convert 2-byte array (little endian) to radians directly.

        :param byte_array: 2-byte array (little endian)
        """
        if len(byte_array) != 2:
            logger.warning(f"Data length error: need 2 bytes, got {len(byte_array)}")
            return 0.0

        # Build 16-bit integer
        hex_value = (byte_array[0] & 0xFF) | ((byte_array[1] & 0xFF) << 8)
        # print in hex
        # print(f"hex_value: {hex_value}")
        # Range check
        if hex_value < 0 or hex_value > 4095:
            logger.warning(f"Servo value out of range: {hex_value} (valid 0–4095)")
            hex_value = max(0, min(hex_value, 4095))

        # Directly map raw value to radians: 0–4095 -> [-π, π]
        return (hex_value / 4096.0) * (2 * math.pi) - math.pi

    def _value_to_radians(self, value: int) -> float:
        """
        Convert servo raw value to radians directly.

        :param value: Servo raw value (0-4095)
        """
        if value < 0 or value > 4095:
            logger.warning(f"Servo value out of range: {value} (valid 0–4095)")
            value = max(0, min(value, 4095))

        # Directly map raw value to radians: 0–4095 -> [-π, π]
        return (value / 4096.0) * (2 * math.pi) - math.pi

    def _raw_velocity_to_deg_per_sec(self, velocity_raw: int) -> float:
        """
        Convert raw velocity value to degrees per second.
        :param velocity_raw: Raw velocity value from hardware
        :return: Velocity in degrees per second
        """
        # Known mapping: 360 deg/s = 4096 ticks/s
        # Ratio: 360 / 4096 = 0.087890625 deg/(tick/s)
        DEG_PER_TICK_PER_SEC = 360.0 / 4096.0
        
        # Expected hardware range: 50-5000 ticks/s
        MAX_HARDWARE_VALUE = 5000
        
        # Handle abnormal values that exceed the expected limit
        if velocity_raw > MAX_HARDWARE_VALUE:
            # Clamp abnormal values to maximum expected value
            velocity_raw = MAX_HARDWARE_VALUE
            if self.debug_mode:
                logger.debug(
                    f"Velocity raw value exceeds expected limit (5000), "
                    f"clamped to {MAX_HARDWARE_VALUE} before conversion"
                )
        
        # Convert raw value to degrees per second
        velocity_deg_s = velocity_raw * DEG_PER_TICK_PER_SEC
        
        return velocity_deg_s

    def _decimal_to_version_string(self, decimal_value: int) -> str:
        """
        Convert decimal value to version string.
        :param decimal_value: Decimal version value
        :return: Version string in format "X.Y.Z" (MAJOR.MINOR.PATCH)
        """
        if decimal_value < 0:
            return "unknown"
        
        decimal_str = str(decimal_value)
        
        if len(decimal_str) == 1:
            version_str = f"0.0.{decimal_str}"
        elif len(decimal_str) == 2:
            version_str = f"{decimal_str[0]}.{decimal_str[1]}.0"
        elif len(decimal_str) == 3:
            version_str = f"{decimal_str[0]}.{decimal_str[1]}.{decimal_str[2]}"
        else:
            version_str = f"{decimal_str[0]}.{decimal_str[1]}.{decimal_str[2:]}"
        
        return version_str
