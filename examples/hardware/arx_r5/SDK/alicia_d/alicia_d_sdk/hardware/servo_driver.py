# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import math
import time
import threading
from typing import List, Optional, Dict, Union
import numpy as np

from alicia_d_sdk.hardware.serial_comm import SerialComm
from alicia_d_sdk.hardware.data_parser import DataParser
from alicia_d_sdk.utils.logger import logger, hex_print


class ServoDriver:
    """Robot arm control module"""

    # Constant definitions

    RAD_TO_DEG = 180.0 / math.pi  # 弧度转角度系数
    DEG_TO_RAD = math.pi / 180.0  # 角度转弧度系数
    # Frame constants
    FRAME_HEADER = 0xAA
    FRAME_FOOTER = 0xFF

    # Command IDs
    CMD_JOINT = 0x06       # Arm joint angle feedback and control

    # Gripper type configuration
    GRI_MAX_50MM = 3290
    GRI_MAX_100MM = 3590

    # Default gripper speed: 5500 ticks/s ≈ 483.4 deg/s
    DEFAULT_GRIPPER_SPEED_TICKS = 5500

    # Raw instruction mapping table for information retrieval and control
    INFO_COMMAND_MAP: Dict[str, List[int]] = {
        # Get firmware version
        "version": [0xAA, 0x01, 0x00, 0x01, 0xFE, 0x23, 0xFF],
        # Set current position as zero
        "zero_cali": [0xAA, 0x03, 0x00, 0x01, 0xFE, 0xA8, 0xFF],
        # Torque on/off
        "torque_on": [0xAA, 0x05, 0x00, 0x01, 0x01, 0xF9, 0xFF],
        "torque_off": [0xAA, 0x05, 0x00, 0x01, 0x00, 0x6F, 0xFF],
        # Joint information acquisition (position and status)
        "joint_gripper": [0xAA, 0x06, 0x00, 0x01, 0xFE, 0x9A, 0xFF],
        # Temperature information acquisition
        "temperature": [0xAA, 0x06, 0x01, 0x01, 0xFE, 0xAD, 0xFF],
        # Velocity information acquisition
        "velocity": [0xAA, 0x06, 0x02, 0x01, 0xFE, 0xF4, 0xFF],
        "self_check": [0xAA, 0xFE, 0x00, 0x00, 0xFE, 0x93, 0xFF],
        # Gripper type acquisition
        "gripper_type": [0xAA, 0x04, 0x0E, 0x01, 0xFE, 0x1B, 0xFF],
    }

    def __init__(self, port: str = "", debug_mode: bool = False):
        """
        Initialize robot arm controller

        :param port: Serial port name, leave empty to auto search
        :param debug_mode: Whether to enable debug mode
        """
        self.debug_mode = debug_mode
        self._lock = threading.Lock()

        # Create serial communication module and data parser
        self.serial_comm = SerialComm(lock=self._lock, port=port, debug_mode=debug_mode)
        self.data_parser = DataParser(lock=self._lock, debug_mode=debug_mode)


        self.joint_count = 6
        # State update thread related
        self._update_thread = None
        self.thread_update_interval = 0.001  # Update interval in seconds
        self._stop_thread = threading.Event()
        self._thread_running = False

        self.disconnect()

    def wait_for_valid_state(self, timeout: float = 1.5) -> bool:
        """
        Wait for robot arm state to become valid

        :param timeout: Maximum waiting time (seconds)
        :return: Whether a valid state was received within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            js = self.data_parser.get_joint_state()
            if js and max(abs(a) for a in js.angles) > 1e-3:
                return True
            time.sleep(0.05)
        # print(f"[Timeout] No valid joint state received")
        return False

    def __del__(self):
        """Destructor to ensure threads and connections are properly cleaned up"""
        try:
            # Stop state update thread
            self.stop_update_thread()
            # Disconnect
            self.disconnect()
        except Exception as e:
            if hasattr(logger, 'error'):  # logger may be destroyed in some cases
                logger.error(f"Exception in destructor: {str(e)}")

    def connect(self) -> bool:
        """
        Connect to the robot arm

        :return: Whether connection is successful
        """
        result = self.serial_comm.connect()
        if result:
            # Start state update thread after successful connection
            self.start_update_thread()
            # self.wait_for_valid_state()
        return result

    def disconnect(self):
        """Disconnect from the robot arm"""
        # Stop state update thread first
        self.stop_update_thread()
        self.serial_comm.disconnect()

    def start_update_thread(self):
        """Start state update thread"""
        if self._update_thread is not None and self._thread_running:
            logger.info("State update thread is already running")
            return

        # Reset stop flag
        self._stop_thread.clear()
        self._thread_running = True

        # Create and start thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop_update_thread(self):
        """Stop state update thread"""
        if self._update_thread is None or not self._thread_running:
            return

        # Set stop flag
        self._stop_thread.set()
        self._thread_running = False

        # Wait for thread to finish
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)

        self._update_thread = None

    def is_update_thread_running(self) -> bool:
        """
        Check whether state update thread is running
        """
        return self._thread_running and self._update_thread is not None and self._update_thread.is_alive()

    def get_update_thread_status(self) -> Dict:
        """
        Get detailed status of the state update thread

        :return: Dictionary containing thread status
        """
        return {
            "running": self.is_update_thread_running(),
            "enabled": self._thread_running,
            "thread_exists": self._update_thread is not None,
            "thread_alive": self._update_thread.is_alive() if self._update_thread else False,
            "stop_flag_set": self._stop_thread.is_set()
        }

    def _update_loop(self):
        """Main loop of state update thread (supports responsive polling)"""
        while not self._stop_thread.is_set():
            try:
                # Read and process all available frames in one iteration
                frames_read = 0
                max_frames_per_iteration = 10  # Limit to prevent blocking too long

                while frames_read < max_frames_per_iteration and not self._stop_thread.is_set():
                    with self._lock:
                        frame = self.serial_comm.read_frame()
                    
                    if frame is None:
                        # No more frames available, break inner loop
                        break
                    
                    # Handle severe communication error
                    if frame == 9999999:
                        try:
                            logger.error("Severe serial communication error detected, robot arm may be disconnected")
                        except Exception:
                            pass  # Logger raises exception, but we want to continue to break
                        break
                    
                    if frame:
                        self.data_parser.parse_frame(frame)
                        frames_read += 1
                
                # If we read frames, continue immediately (no sleep) for better responsiveness
                # Otherwise, sleep to avoid busy waiting
                if frames_read == 0:
                    time.sleep(self.thread_update_interval)
            except Exception as e:
                # Log error but continue loop for resilience (improved from breaking)
                try:
                    logger.error(f"State update thread exception: {str(e)}")
                except Exception:
                    # Logger raises exception, but we want to continue the loop
                    pass
                # Sleep briefly before retrying to avoid rapid error loops
                time.sleep(self.thread_update_interval)
        self._thread_running = False

    def acquire_info(self, info_type: str, wait: bool = False, timeout: float = 2.0, retry_interval: float = 0.2) -> bool:
        """
        General information acquisition interface, selecting different commands by type.

        :param info_type: Type of information to acquire (version, zero_cali, torque_on, torque_off, joint, etc.)
        :param wait: If True, wait for the response to be received and parsed
        :param timeout: Maximum time to wait in seconds (only used if wait=True)
        :param retry_interval: Time interval between retry attempts in seconds (default 0.2s)
        :return: True if successful
        """
        if info_type not in self.INFO_COMMAND_MAP:
            raise ValueError(f"Unsupported info type: {info_type}")

        # Clear the corresponding event before sending request (if applicable)
        if info_type in self.data_parser._info_event_map:
            event = self.data_parser._info_event_map[info_type]
            event.clear()

        command = self.INFO_COMMAND_MAP[info_type]
        # If not waiting, just send once
        if not wait:
            success = self.serial_comm.send_data(command)
            return success

        # If waiting and has an event, implement retry logic
        if info_type in self.data_parser._info_event_map:
            event = self.data_parser._info_event_map[info_type]
            start_time = time.time()

            while time.time() - start_time < timeout:
                # Send command
                success = self.serial_comm.send_data(command)
                if not success:
                    logger.warning(f"Failed to send {info_type} command, retrying...")
                    time.sleep(retry_interval)
                    continue

                # Wait for response with a short timeout (retry_interval)
                remaining_time = timeout - (time.time() - start_time)
                wait_time = min(retry_interval, remaining_time)

                if event.wait(wait_time):
                    # Successfully received response
                    return True

            # Timeout exceeded
            logger.warning(f"Failed to get {info_type} within timeout period after multiple retries")
            return False
        else:
            # For commands without events, just send once
            success = self.serial_comm.send_data(command)
            return success

    def set_joint_and_gripper(self,
                              joint_angles: Optional[List[float]] = None,
                              gripper_value: Optional[float] = None,
                              speed_deg_s: Union[int, float, List[float], np.ndarray] = 10,
                              gripper_speed_deg_s: Optional[float] = None) -> bool:
        """
        Unified method to set joints, gripper, or both in a single combined frame.

        :param joint_angles: Optional angle list (radians) for 6 joints. If None, keeps current joints (or zeros if state unavailable)
        :param gripper_value: Optional gripper value (0-1000). If None, keeps current gripper
        :param speed_deg_s: Speed in degrees per second. Can be int/float (same for all joints) or list/array (per-joint speeds, 4.39-439.45 deg/s), default 10
        :param gripper_speed_deg_s: Gripper speed in degrees per second. If None, uses default 5500 ticks/s (≈483.4 deg/s)
        :return: True if successful
        """

        # Speed validation
        if isinstance(speed_deg_s, (int, float)):
            if speed_deg_s <= 0:
                logger.error(f"Speed must be positive: {speed_deg_s} deg/s")
                return False
        elif isinstance(speed_deg_s, (list, np.ndarray)):
            speed_list = list(speed_deg_s) if isinstance(speed_deg_s, np.ndarray) else speed_deg_s
            if any(s <= 0 for s in speed_list):
                logger.error(f"All speeds must be positive: {speed_list} deg/s")
                return False
            if len(speed_list) != 6:
                logger.error(f"Incorrect speed list length: need 6, got {len(speed_list)}")
                return False
        else:
            logger.error(f"Invalid speed_deg_s type: {type(speed_deg_s)}")
            return False

        if gripper_speed_deg_s is not None and gripper_speed_deg_s <= 0:
            logger.error(f"Gripper speed must be positive: {gripper_speed_deg_s} deg/s")
            return False

        frame = self._build_joint_frame(
            joint_angles=joint_angles,
            gripper_value=gripper_value,
            speed_deg_s=speed_deg_s,
            gripper_speed_deg_s=gripper_speed_deg_s
        )


        # self.serial_comm._hex_print("Send combined control", frame)
        result = self.serial_comm.send_data(frame)

        if self.debug_mode:
            self.serial_comm._hex_print("Send combined control", frame)

        return result

    def _build_joint_frame(self,
                           joint_angles: Optional[List[float]] = None,
                           gripper_value: Optional[float] = None,
                           speed_deg_s: Union[int, float, List[float], np.ndarray] = 10,
                           gripper_speed_deg_s: Optional[float] = None) -> List[int]:
        """
        Build combined joint + gripper + speed control frame (CMD=0x06, FUNC=0x03)

        :param joint_angles: Optional angle list (radians) for 6 joints. If None, keeps current joints (or zeros if state unavailable)
        :param gripper_value: Optional gripper value (0-1000). If None, keeps current gripper
        :param speed_deg_s: Speed in degrees per second. Can be int/float (same for all joints) or list/array (per-joint speeds, 4.39-439.45 deg/s)
        :param gripper_speed_deg_s: Gripper speed in degrees per second. If None, uses default 5500 ticks/s (≈483.4 deg/s)
        :return: Frame byte list
        """
        # 6 joints * 4 bytes (value + speed) + 1 gripper * 4 bytes (value + speed) = 28 bytes
        DATA_LENGTH = 0x1C
        FRAME_SIZE = 1 + 1 + 1 + 1 + DATA_LENGTH + 1 + 1  # header + cmd + func + len + data + checksum + footer

        # Create frame
        frame = [0] * FRAME_SIZE
        frame[0] = self.FRAME_HEADER  # 0xAA
        frame[1] = self.CMD_JOINT     # 0x06
        frame[2] = 0x03               # Function code
        frame[3] = DATA_LENGTH        # Data length: 0x1C (28 bytes)
        frame[-1] = self.FRAME_FOOTER  # 0xFF

        data_start = 4

        # Get current state for optional values
        current_state = self.data_parser.get_joint_state()
        if joint_angles is None:
            if current_state and current_state.angles:
                effective_joints = current_state.angles
            else:
                # Default to zero if no current state available
                effective_joints = [0.0] * self.joint_count
        else:
            if len(joint_angles) != self.joint_count:
                logger.error(f"Incorrect joint count: need {self.joint_count}, got {len(joint_angles)}")
                # Fall back to current or zeros to avoid crashing
                if current_state and current_state.angles:
                    effective_joints = current_state.angles
                else:
                    effective_joints = [0.0] * self.joint_count
            else:
                effective_joints = joint_angles

        # Handle speed_deg_s: can be int/float (uniform) or list/array (per-joint)
        if isinstance(speed_deg_s, (int, float)):
            # Uniform speed for all joints
            speed_hw_values = [self._value_to_hardware_value_speed(speed_deg_s)] * 6
        elif isinstance(speed_deg_s, (list, np.ndarray)):
            # Per-joint speeds
            speed_list = list(speed_deg_s) if isinstance(speed_deg_s, np.ndarray) else speed_deg_s
            if len(speed_list) != 6:
                logger.error(f"Incorrect speed list length: need 6, got {len(speed_list)}, using first value for all joints")
                speed_hw_values = [self._value_to_hardware_value_speed(speed_list[0])] * 6
            else:
                speed_hw_values = [self._value_to_hardware_value_speed(s) for s in speed_list]
        else:
            logger.error(f"Invalid speed_deg_s type: {type(speed_deg_s)}, using default 10 deg/s")
            speed_hw_values = [self._value_to_hardware_value_speed(10)] * 6

        for joint_idx in range(6):
            angle_rad = effective_joints[joint_idx]
            hardware_value = self._rad_to_hardware_value(angle_rad)
            # print(f"hardware_value_target: {hardware_value}")
            offset = data_start + joint_idx * 4
            frame[offset] = hardware_value & 0xFF              # low byte
            frame[offset + 1] = (hardware_value >> 8) & 0xFF   # high byte
            frame[offset + 2] = speed_hw_values[joint_idx] & 0xFF
            frame[offset + 3] = (speed_hw_values[joint_idx] >> 8) & 0xFF

        # Gripper value and speed (4 bytes: 2 bytes value, 2 bytes speed)
        gripper_offset = data_start + 6 * 4
        if gripper_value is not None:
            gripper_hw_value = int(max(0, min(1000, gripper_value)))
        else:
            if current_state and current_state.gripper is not None:
                gripper_hw_value = int(max(0, min(1000, current_state.gripper)))
            else:
                gripper_hw_value = 1000  # Default middle position

        # Handle gripper speed: convert from deg/s to ticks/s, default is 5500 ticks/s
        if gripper_speed_deg_s is not None:
            gripper_speed_hw_value = self._gripper_speed_deg_s_to_ticks(gripper_speed_deg_s)
        else:
            # Default: 5500 ticks/s (≈483.4 deg/s)
            gripper_speed_hw_value = self.DEFAULT_GRIPPER_SPEED_TICKS

        frame[gripper_offset] = gripper_hw_value & 0xFF
        frame[gripper_offset + 1] = (gripper_hw_value >> 8) & 0xFF
        frame[gripper_offset + 2] = gripper_speed_hw_value & 0xFF
        frame[gripper_offset + 3] = (gripper_speed_hw_value >> 8) & 0xFF
        frame[-2] = self.serial_comm.calculate_checksum(frame[1:-2])

        if self.debug_mode:
            angle_deg = [round(angle * self.RAD_TO_DEG, 2) for angle in effective_joints]
            if isinstance(speed_deg_s, (int, float)):
                speed_str = f"{speed_deg_s} deg/s (uniform)"
            else:
                speed_str = f"{speed_deg_s} deg/s (per-joint)"
            default_gripper_speed_deg_s = self._gripper_speed_ticks_to_deg_s(self.DEFAULT_GRIPPER_SPEED_TICKS)
            gripper_speed_str = f"{gripper_speed_deg_s} deg/s" if gripper_speed_deg_s is not None else f"{self.DEFAULT_GRIPPER_SPEED_TICKS} ticks/s (≈{default_gripper_speed_deg_s:.1f} deg/s, default)"
            logger.debug(f"Send combined frame - joints (deg): {angle_deg}, gripper: {gripper_value}, joint speeds: {speed_str}, gripper speed: {gripper_speed_str}")

        return frame

    def _rad_to_hardware_value(self, angle_rad: float) -> int:
        """
        Convert radians to hardware value (0-4095)

        :param angle_rad: Angle (radians)
        :return: Hardware value (0-4095)
        """
        # Range check in radians
        if angle_rad < -math.pi or angle_rad > math.pi:
            logger.warning(f"Angle out of range: {angle_rad:.2f} rad, will be clipped")
            angle_rad = max(-math.pi, min(math.pi, angle_rad))

        value = int((angle_rad + math.pi) / (2 * math.pi) * 4096)

        return max(0, min(4095, value))

    def _value_to_hardware_value_speed(self, speed_deg_s: Union[int, float]) -> int:
        """
        Converts speed from degrees per second to hardware value (50-5000, step 50).
        Mapping: 360 deg/s = 4096 ticks/s, so 50 ticks/s ≈ 4.39 deg/s, 5000 ticks/s ≈ 439.45 deg/s.

        :param speed_deg_s: The desired speed in degrees per second (4.39-439.45, required range)
        :return: A corresponding raw integer speed value (50-5000, multiple of 50)
        """
        # Hardware speed range: 50-5000 ticks/s (step 50)
        MIN_HARDWARE_VALUE = 50
        MAX_HARDWARE_VALUE = 5000
        STEP_SIZE = 50

        # Known mapping: 360 deg/s = 4096 ticks/s
        # Calculate speed range based on hardware range
        # Ratio: 360 / 4096 = 0.087890625 deg/(tick/s)
        DEG_PER_TICK_PER_SEC = 360.0 / 4096.0
        MIN_SPEED_DEG_S = MIN_HARDWARE_VALUE * DEG_PER_TICK_PER_SEC  # ≈ 4.39 deg/s
        MAX_SPEED_DEG_S = MAX_HARDWARE_VALUE * DEG_PER_TICK_PER_SEC  # ≈ 439.45 deg/s

        # Validate and clip speed to required range
        if speed_deg_s < MIN_SPEED_DEG_S:
            logger.warning(f"Speed below range: {speed_deg_s} deg/s (min {MIN_SPEED_DEG_S:.2f}), will be clipped to {MIN_SPEED_DEG_S:.2f}")
            speed_deg_s = MIN_SPEED_DEG_S
        elif speed_deg_s > MAX_SPEED_DEG_S:
            logger.warning(f"Speed above range: {speed_deg_s} deg/s (max {MAX_SPEED_DEG_S:.2f}), will be clipped to {MAX_SPEED_DEG_S:.2f}")
            speed_deg_s = MAX_SPEED_DEG_S

        # Convert deg/s to ticks/s using the known ratio
        hardware_value = speed_deg_s / DEG_PER_TICK_PER_SEC

        # Round to nearest multiple of 50
        hardware_value = round(hardware_value / STEP_SIZE) * STEP_SIZE
        logger.debug(f"Speed: {speed_deg_s} deg/s, Hardware value: {hardware_value}")

        return max(MIN_HARDWARE_VALUE, min(MAX_HARDWARE_VALUE, int(hardware_value)))

    def _gripper_speed_deg_s_to_ticks(self, speed_deg_s: float) -> int:
        """
        Converts gripper speed from degrees per second to hardware ticks/s.
        Mapping: 360 deg/s = 4096 ticks/s.
        No step size restriction for gripper speed.

        :param speed_deg_s: Gripper speed in degrees per second
        :return: Hardware speed value in ticks/s
        """
        # Known mapping: 360 deg/s = 4096 ticks/s
        DEG_PER_TICK_PER_SEC = 360.0 / 4096.0
        hardware_value = speed_deg_s / DEG_PER_TICK_PER_SEC
        return int(round(hardware_value))

    def _gripper_speed_ticks_to_deg_s(self, speed_ticks: int) -> float:
        """
        Converts gripper speed from hardware ticks/s to degrees per second.
        Mapping: 360 deg/s = 4096 ticks/s.

        :param speed_ticks: Gripper speed in ticks/s
        :return: Speed in degrees per second
        """
        # Known mapping: 360 deg/s = 4096 ticks/s
        DEG_PER_TICK_PER_SEC = 360.0 / 4096.0
        return speed_ticks * DEG_PER_TICK_PER_SEC
