# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Read and print robot state

Features:
- Read joint angles (radians or degrees)
- Read end-effector pose
- Read gripper state
- Support single or continuous printing
- Configurable reading frequency (FPS) for continuous mode
"""

import alicia_d_sdk


def main(args):
    """Read and print robot state.
    
    :param args: Command line arguments
    """
    # Initialize robot instance
    robot = alicia_d_sdk.create_robot(
        port=args.port,
        gripper_type=args.gripper_type
    )
    
    try:
        # Print robot state once
        if args.single:
            robot.print_state(continuous=False, output_format=args.format)
        else:
            # Print robot state continuously with specified FPS
            robot.print_state(continuous=True, output_format=args.format, fps=args.fps)
        
    except KeyboardInterrupt:
        print("\n✗ Reading interrupted")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        robot.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Read robot state")
    
    # Serial port settings
    parser.add_argument('--port', type=str, default="", help="串口端口 (例如: /dev/ttyUSB0 或 COM3)")
    parser.add_argument('--gripper_type', type=str, default="50mm",  help="夹爪型号 (默认: 50mm)")
    # Display settings
    parser.add_argument('--format', type=str, default='deg', choices=['rad', 'deg'], help="Angle display format: rad(radians) or deg(degrees)")
    parser.add_argument('--single', action='store_true',  help="Print state once (default: continuous print)")
    parser.add_argument('--fps', type=float, default=30.0, help="Target frames per second for continuous mode (default: 200 Hz)")         
    
    args = parser.parse_args()
    
    main(args)
