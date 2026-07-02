# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Control robot to move to target joint positions using move_to_joint_state

Features:
- Support degree and radian input
- Automatic joint angle interpolation
- Adjustable motion speed
"""

import alicia_d_sdk
import time

def main(args):
    """Control robot joint movements.
    
    :param args: Command line arguments
    """
    # Initialize robot instance
    robot = alicia_d_sdk.create_robot(port=args.port)


    try:
        # Set target joint positions in degrees
        target_joints_deg = [-30, 30.0, 30.0, 20.0, -20.0, 10.0]
        robot.set_home(speed_deg_s=args.speed_deg_s)
        time.sleep(1)
        # Use unified joint and gripper target interface
        robot.set_robot_state(
            target_joints=target_joints_deg,
            joint_format='deg',
            speed_deg_s=args.speed_deg_s,
            wait_for_completion=True
        )
        time.sleep(1)
        robot.set_home(speed_deg_s=args.speed_deg_s)

    except KeyboardInterrupt:
        print("\n✗ Processing interrupted")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="机械臂运动控制示例")
    
    parser.add_argument('--port', type=str, default="", help="串口端口 (例如: /dev/ttyUSB0 或 COM3)")
    parser.add_argument('--speed_deg_s', type=int, default=10,  help="关节运动速度 (单位: 度/秒，默认: 10，范围: 10-400度/秒)")
    args = parser.parse_args()
    main(args)