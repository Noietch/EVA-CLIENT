# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Gripper control

Features:
- Open/close gripper
- Control gripper to specific angle
- Wait for gripper motion completion
"""

import alicia_d_sdk
import time
from alicia_d_sdk.utils.logger import logger

def main(args):
    """Demonstrate gripper control.
    
    :param args: Command line arguments
    """
    # Initialize robot instance
    robot = alicia_d_sdk.create_robot(port=args.port)
    
    try:
        # Read and print initial gripper value (0-1000)
        gripper_value = robot.get_robot_state("gripper")
        if gripper_value is not None:
            logger.info(f"Initial gripper value: {gripper_value:.1f}")
        else:
            logger.warning("Failed to read initial gripper value")

        # Demo sequence: open -> close -> half-open -> open
        sequence = [
            ("Step 1", 1000, "Open gripper"),
            ("Step 2", 0, "Close gripper"),
            ("Step 3", 500, "Half-open gripper"),
            ("Step 4", 1000, "Open gripper again"),
        ]

        for step, target, description in sequence:
            logger.info(f"{step}: {description}, target={target}")
            ok = robot.set_robot_state(
                gripper_value=target,
                wait_for_completion=True,
                gripper_speed_deg_s=100,
            )
            if not ok:
                logger.warning(f"{step}: command failed")
                continue

            current = robot.get_robot_state("gripper")
            if current is not None:
                logger.info(f"{step}: reached value={current:.1f}")
            else:
                logger.warning(f"{step}: command sent but failed to read current gripper value")

            time.sleep(1)

        
    except KeyboardInterrupt:
        print("\n✗ Processing interrupted")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    finally:
        robot.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Gripper Control Demo")
    
    # Serial port settings
    parser.add_argument('--port', type=str, default="", help="串口端口 (例如: /dev/ttyUSB0 或 COM3)")
    args = parser.parse_args()
    
    
    main(args)
