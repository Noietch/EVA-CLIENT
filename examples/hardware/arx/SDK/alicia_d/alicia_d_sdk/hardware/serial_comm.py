# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import binascii
import serial
import platform
import serial.tools.list_ports
import time
import os
from typing import List, Optional, Tuple
import threading
from datetime import datetime
# from PyCRC.CRC32 import CRC32
import getpass
from alicia_d_sdk.utils.logger import logger
READ_LENGTH = 50
DEFAULT_LENGTH = 6


class CRC32:
    @staticmethod
    def calculate(data):
        if isinstance(data, str):
            data = data.encode()
        return binascii.crc32(data) & 0xffffffff

class SerialComm:
    """Robot arm serial communication module"""

    PLATFORM_PRIORITIES = {
        "Darwin": ["cu.wchusbserial", "cu.SLAB_USBtoUART", "cu.usbserial", "cu.usbmodem", "ttyUSB", "COM"],
        "Linux": ["ttyUSB", "ttyACM", "ttyCH343USB", "ttyCH341USB", "cu.wchusbserial",
                  "cu.SLAB_USBtoUART", "cu.usbserial", "cu.usbmodem", "COM"],
        "Windows": ["COM", "ttyUSB", "cu.usbserial", "cu.usbmodem"]
    }

    def __init__(self, port: str = "", timeout: float = 1.0, debug_mode: bool = False, lock: Optional[threading.Lock] = None):
        """
        :param port: Serial port name, leave empty to auto-search
        :param timeout: Timeout in seconds
        :param debug_mode: Whether to enable debug mode
        :param lock: Optional thread lock, auto-created if not provided
        """
        self.port_name = port
        self.timeout = timeout
        self.baudrate = 1000000
        self.debug_mode = debug_mode
        self.serial_port = None
        self.last_log_time = 0
        self._last_print_time = 0
        self._lock = lock if lock is not None else threading.Lock()
        self._rx_buffer = bytearray()
        self._frames_processed = 0
        self._frames_dropped = 0

    def __del__(self):
        """Destructor, ensure serial port is closed"""
        self.disconnect()

    def connect(self) -> bool:
        """Connect to serial port device"""
        try:
            port = self.find_serial_port()
            if not port:
                logger.warning("No available serial port found")
                return False

            has_permission, error_msg = self._check_serial_permissions(port)
            if not has_permission:
                logger.error(error_msg)
                return False

            logger.info(f"Connecting to port: {port}")

            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()

            port = self._prefer_cu_port(port)

            if 'cu.usbserial' in port:
                logger.info(f"Current baudrate is {self.baudrate}, if communication is abnormal, try 1000000/1000000/921600")

            self.serial_port = serial.Serial(
                port=port, baudrate=self.baudrate, timeout=self.timeout,
                write_timeout=self.timeout, xonxoff=False, rtscts=False, dsrdtr=False
            )
            self._initialize_serial_port()

            if self.serial_port.is_open:
                return True
            return False
        except Exception as e:
            logger.error(f"Serial port connection exception: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect serial port connection"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("Serial port closed")

    def is_connected(self) -> bool:
        """Check if serial port is connected and open"""
        return self.serial_port is not None and self.serial_port.is_open

    def find_serial_port(self) -> str:
        """Find available serial port device"""
        current_time = time.time()
        should_log = (current_time - self.last_log_time) >= 2.0

        # Handle user-specified port
        if self.port_name:
            device_name = self._normalize_device_name(self.port_name, should_log)
            if self._is_device_accessible(device_name):
                logger.info(f"Using specified port: {device_name}")
                return device_name
            logger.warning(f"Specified port {device_name} is not available, will search for other devices")

        # Get serial port list
        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as e:
            if should_log:
                logger.error(f"Exception when listing ports: {str(e)}")
                self.last_log_time = current_time
            return ""

        if not ports:
            return ""

        if should_log:
            self.last_log_time = current_time

        # Find device by priority
        for key in self.PLATFORM_PRIORITIES.get(platform.system(), self.PLATFORM_PRIORITIES["Windows"]):
            for p in ports:
                if key in p.device:
                    device_name = self._normalize_device_name(p.device, should_log)
                    if self._is_device_accessible(device_name):
                        return device_name

        # macOS: try to map tty.* to cu.*
        if platform.system() == "Darwin":
            for p in ports:
                if p.device.startswith('/dev/tty.'):
                    cu_candidate = p.device.replace('/dev/tty.', '/dev/cu.')
                    if os.path.exists(cu_candidate) and os.access(cu_candidate, os.R_OK | os.W_OK):
                        if should_log:
                            logger.info(f"Map {p.device} to {cu_candidate}")
                        return cu_candidate

        if should_log:
            logger.warning("No available serial port device found (supports ttyUSB/ttyACM/ttyCH343USB/cu.usbserial/cu.usbmodem/COM)")
        return ""

    def send_data(self, data: List[int]) -> bool:
        """
        Send data to serial port

        :param data: Byte data list to send
        :return: Whether send is successful
        """
        with self._lock:
            try:
                if not self.serial_port or not self.serial_port.is_open:
                    logger.warning("Serial port not open, trying to reconnect")
                    if not self.connect():
                        logger.error("Cannot connect to serial port")
                        return False

                # Convert to byte array
                data_bytes = bytes(data)
                # self._hex_print("data_bytes", data_bytes)
                # Write data
                bytes_written = self.serial_port.write(data_bytes)
                time.sleep(0.001)  # 必须 0.001， Mac上158hz
                try:
                    self.serial_port.flush()
                except Exception:
                    print("flush error")
                    pass

                if bytes_written != len(data):
                    logger.warning(f"Only wrote {bytes_written} bytes, should be {len(data)} bytes")
                    return False

                if self.debug_mode:
                    self._print_hex_frame(data, 0)

                return True

            except Exception as e:
                logger.error(f"Exception sending data: {str(e)}")
                return False

    def read_frame(self) -> Optional[List[int]]:
        """
        Read one frame of data (non-blocking, returns None if no complete frame)

        :return: Complete data frame, returns None if not available
        """
        try:
            # Check serial port status before any operation
            if not self.serial_port or not self.serial_port.is_open:
                return None

            # Check if there is data to read (may raise OSError if port is closed)
            try:
                if self.serial_port.in_waiting == 0:
                    return None
                available_bytes = self.serial_port.in_waiting
            except (OSError, AttributeError) as e:
                # Port was closed between check and read
                if hasattr(e, 'errno') and e.errno == 9:  # Bad file descriptor
                    return None
                raise

            max_read_size = 80
            read_size = min(available_bytes, max_read_size)

            # Read data (may raise OSError if port is closed)
            try:
                self._rx_buffer += self.serial_port.read(read_size)
            except (OSError, AttributeError) as e:
                # Port was closed during read
                if hasattr(e, 'errno') and e.errno == 9:  # Bad file descriptor
                    return None
                raise

            while len(self._rx_buffer) >= 6:
                if len(self._rx_buffer) > 200:
                    self._rx_buffer.clear()
                    continue

                # self._hex_print("self._rx_buffer", self._rx_buffer)
                # Step 2: Sync to frame header 0xAA
                if self._rx_buffer[0] != 0xAA:
                    self._rx_buffer.pop(0)
                    continue

                if self.debug_mode:
                    print(f" Buffer size: {len(self._rx_buffer)} bytes, first bytes: {self._rx_buffer[:min(12, len(self._rx_buffer))]}")

                data_len = self._rx_buffer[3]

                frame_length = data_len + DEFAULT_LENGTH
                if len(self._rx_buffer) < frame_length:
                    # Wait for more data
                    break

                # Step 4: Check if buffer contains complete frame
                if len(self._rx_buffer) < frame_length:
                    if self.debug_mode:
                        logger.debug(f" Incomplete frame: need {frame_length}, have {len(self._rx_buffer)}")
                    break

                candidate = self._rx_buffer[:frame_length]

                # Step 5: Verify frame tail and checksum
                valid_tail = candidate[-1] == 0xFF

                if not valid_tail:
                    # Tail mismatch, this 0xAA was not a start or data is corrupted
                    self._rx_buffer.pop(0)
                    continue

                if self._serial_data_check(candidate):
                    self._rx_buffer = self._rx_buffer[frame_length:]
                    # frames_processed += 1
                    # self._hex_print("candidate", list(candidate))
                    return list(candidate)
                else:
                    # Checksum failed
                    logger.warning(f"CRC Error. Raw: {' '.join(f'{b:02X}' for b in candidate)}")
                    self._rx_buffer.pop(0)

            return None

        except (OSError, AttributeError) as e:
            # Handle Bad file descriptor error gracefully (port closed)
            if hasattr(e, 'errno') and e.errno == 9:  # Bad file descriptor
                # Port was closed, this is expected during shutdown
                return None
            logger.error(f"Exception reading data: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Exception reading data: {str(e)}")
            return None

    def _serial_data_check(self, frame: bytearray) -> bool:
        """
        Verify CRC8 checksum using specific robot algorithm.
        Frame: [AA] [Cmd] [Func] [Len] [Data...] [CRC] [FF]
        """
        received_checksum = frame[-2]
        # Payload includes Cmd, Func, Len, and Data (everything between Header and CRC)
        payload_to_check = frame[1:-2]

        calculated_checksum = self.calculate_checksum(payload_to_check)
        return received_checksum == calculated_checksum

    def calculate_checksum(self, data) -> int:
        """
        Use CRC-32 and only use the last 8 bits by pycrc
        """
        crc_calculator = CRC32()
        crc = crc_calculator.calculate(bytes(data))
        return crc & 0xFF

    def get_processing_stats(self) -> dict:
        """
        Get frame processing statistics

        :return: Contains statistics of processed and dropped frames
        """
        return {
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "buffer_size": len(self._rx_buffer)
        }

    def _prefer_cu_port(self, port: str) -> str:
        """Convert macOS tty.* to cu.* if available"""
        if '/dev/tty.' in port:
            cu_candidate = port.replace('/dev/tty.', '/dev/cu.')
            if os.path.exists(cu_candidate) and os.access(cu_candidate, os.R_OK | os.W_OK):
                logger.info(f"Detected macOS port {port}, switching to {cu_candidate} for writing")
                return cu_candidate
        return port

    def _initialize_serial_port(self):
        """Initialize serial port buffers and handshake lines"""
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        self.serial_port.setDTR(True)  # Some controllers ignore TX when DTR is low
        self.serial_port.setRTS(False)

    def _normalize_device_name(self, device_name: str, should_log: bool = False) -> str:
        """Normalize device name for Windows COM port and Linux path"""
        # Windows: add prefix for COM ports > 9
        if platform.system() == "Windows" and device_name.startswith("COM"):
            try:
                port_num = int(device_name[3:])
                if port_num > 9 and not device_name.startswith("\\\\.\\"):
                    device_name = f"\\\\.\\{device_name}"
                    if should_log:
                        logger.info(f"Windows COM port number greater than 9, add prefix: {device_name}")
            except ValueError:
                pass

        # Linux: ensure /dev/ prefix
        if platform.system() == "Linux" and not device_name.startswith("/dev/"):
            if device_name.startswith(("tty", "cu")):
                device_name = f"/dev/{device_name}"

        return device_name

    def _check_serial_permissions(self, device_name: str) -> Tuple[bool, Optional[str]]:
        """Check serial port device permissions

        :param device_name: Device name to check
        :return: Tuple of (has_permission, error_message)
        """
        if platform.system() == "Windows":
            return True, None

        if not os.path.exists(device_name):
            return False, f"Device {device_name} does not exist"

        if not os.access(device_name, os.R_OK | os.W_OK):
            current_user = getpass.getuser()
            system = platform.system()
            solutions = {
                "Linux": (
                    f"  1. Add user '{current_user}' to dialout group:\n"
                    f"     sudo usermod -a -G dialout {current_user}\n"
                    f"  2. Log out and log back in, or run: newgrp dialout\n"
                    f"  3. Or temporarily use: sudo chmod 666 {device_name}\n"
                ),
                "Darwin": (
                    f"  1. Add user '{current_user}' to dialout or uucp group\n"
                    f"  2. Or temporarily use: sudo chmod 666 {device_name}\n"
                )
            }
            solution = solutions.get(system, f"  Temporarily use: sudo chmod 666 {device_name}\n")
            return False, f"Insufficient permissions: Cannot access serial port device {device_name}\nSolution:\n{solution}"

        return True, None

    def _is_device_accessible(self, device_name: str) -> bool:
        """Check if device exists and is accessible"""
        if platform.system() == "Windows" and device_name.startswith(("COM", "\\\\.\\COM")):
            return True
        if not os.path.exists(device_name):
            return False
        # Permission check is done in connect() for detailed error messages
        has_permission, error_msg = self._check_serial_permissions(device_name)
        if not has_permission and error_msg and self.debug_mode:
            logger.warning(error_msg)
        return True

    def _hex_print(self, title: str, data: List[int]):
        hex_buf = ' '.join(f"{b:02X}" for b in data)
        logger.info(f"{title}: {hex_buf}")
