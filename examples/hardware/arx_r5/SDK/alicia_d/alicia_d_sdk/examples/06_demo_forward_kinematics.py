# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Forward kinematics

Features:
- Read current joint angles
- Calculate end-effector pose
- Display position, rotation matrix, Euler angles, quaternion
"""

import numpy as np
import alicia_d_sdk
from robocore.utils.beauty_logger import beauty_print_array, beauty_print


def main(args):
    robot = alicia_d_sdk.create_robot(port=args.port, 
                                      version=args.version, 
                                      variant=args.variant, 
                                      model_format=args.model_format,
                                      base_link=args.base_link, 
                                      end_link=args.end_link)

    robot_model = robot.robot_model
    
    # Display robot model information
    robot_model.print_tree(show_fixed=True)
    
    # Use API's detailed pose retrieval method (backend is set internally)
    pose_info = robot.get_pose()
    
    # Extract each component
    position_fk = pose_info['position']
    rotation_fk = pose_info['rotation']
    euler_fk = pose_info['euler_xyz']
    quat_fk = pose_info['quaternion_xyzw']
    T_fk = pose_info['transform']
    
    # Display results
    beauty_print("End-Effector Position (m):")
    print(f"  p = {beauty_print_array(position_fk)}")
    beauty_print("End-Effector Orientation (Euler XYZ, radians):")
    print(f"  rpy = {beauty_print_array(euler_fk)}")
    beauty_print("End-Effector Orientation (Euler XYZ, degrees):")
    print(f"  rpy = {beauty_print_array(np.rad2deg(euler_fk))}")
    beauty_print("End-Effector Orientation (Quaternion xyzw):")
    print(f"  quat = {beauty_print_array(quat_fk, precision=6)}")
    # Add note about quaternion sign ambiguity
    quat_neg = -quat_fk
    print("  Note: q and -q represent the same rotation")
    print(f"  -quat = {beauty_print_array(quat_neg, precision=6)} (equivalent)")
    beauty_print("Rotation Matrix:")
    print(beauty_print_array(rotation_fk, precision=6))
    beauty_print("Homogeneous Transformation Matrix:")
    print(beauty_print_array(T_fk, precision=6))
    
    robot.disconnect()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Forward kinematics demo")
    
    # Robot configuration
    parser.add_argument('--port', type=str, default="",   help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    
    parser.add_argument('--version', type=str, default="v5_6", choices=["v5_6"], help="Version")
    parser.add_argument('--variant', type=str, default="leader", choices=["gripper_50mm", "gripper_100mm", "leader_ur", "leader", "vertical_50mm"], help="Variant")
    parser.add_argument('--model_format', type=str, default="urdf", choices=["urdf", "mjcf"], help="Model format")

    parser.add_argument('--base_link', type=str, default="base_link", help="Base link name, world or base_link etc.")
    parser.add_argument('--end_link', type=str, default="tool0", help="End effector link name, tool0 or link6 etc.")
    args = parser.parse_args()
    

    main(args)
