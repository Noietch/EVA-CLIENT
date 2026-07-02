# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Set new zero configuration

Warning: 
- Ensure no obstacles around the robot arm before calibration
- When torque is disabled, manually support the robot arm
- 一旦设置新零点，无法恢复至出厂零点！！！！如果想恢复需购买标定工具！！！
    Once the new zero point is set, it CANNOT be restored to the factory zero point! 
    If you want to restore it, you need to purchase the calibration tool!
"""

import alicia_d_sdk


def main(args):
    robot = alicia_d_sdk.create_robot(port=args.port)
    
    try:
        robot.zero_calibration()
        
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        robot.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Robot zero calibration program")
    
    # Robot configuration
    parser.add_argument('--port', type=str, default="", help="串口端口 (例如: /dev/ttyUSB0 或 COM3)")
    args = parser.parse_args()

    main(args)
