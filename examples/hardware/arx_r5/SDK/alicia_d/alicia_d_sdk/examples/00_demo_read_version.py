# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Read robot firmware version
"""

import alicia_d_sdk
from alicia_d_sdk.utils.logger import logger

def main(args):
    """Read and print robot firmware version.

    :param args: Command line arguments containing port
    """
    # Initialize robot instance
    robot = alicia_d_sdk.create_robot(port=args.port)

    try:
        robot_version = robot.get_robot_state("version")
        if robot_version:
            logger.info(
                "Version info: "
                f"Unique ID = {robot_version.get('serial_number')}, "
                f"Hardware Version = {robot_version.get('hardware_version')}, "
                f"Firmware Version = {robot_version.get('firmware_version')}"
            )
        
        gripper_type = robot.get_robot_state("gripper_type")
        if gripper_type:
            logger.info(f"Gripper type: {gripper_type}")

    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    finally:
        robot.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Read robot firmware version")

    # Robot configuration
    parser.add_argument('--port', type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    args = parser.parse_args()

    main(args)