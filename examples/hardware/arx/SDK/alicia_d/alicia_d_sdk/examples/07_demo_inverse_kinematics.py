# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Inverse kinematics

Features:
- Specify target end-effector pose
- Solve for joint angles
- Move to target position
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import argparse
import alicia_d_sdk
from robocore.utils.beauty_logger import beauty_print_array, beauty_print


def main(args):
    robot = alicia_d_sdk.create_robot(port=args.port,
                                      gripper_type=args.gripper_type,
                                      base_link=args.base_link,
                                      end_link=args.end_link,
                                      backend=args.backend)

    if not robot.is_connected():
        print("✗ Connection failed, please check serial port settings")
        return

    ik_result = robot.set_pose(
        target_pose=args.end_pose,
        backend=args.backend,
        method=args.method,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        max_iters=args.max_iters,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
        speed_deg_s=args.speed_deg_s,
        execute=args.execute,
        force_execute=args.force_execute
    )
    
    beauty_print("Detailed IK Solution Results", type="module")
    print(f"  Success: {ik_result['success']}")
    if ik_result['success']:
        print(f"  Iterations: {ik_result['iters']}")
        print(f"  Position error: {ik_result['pos_err']:.6e} m")
        print(f"  Orientation error: {ik_result['ori_err']:.6e} rad")
        if 'err_norm' in ik_result and ik_result['err_norm'] is not None:
            print(f"  Total error: {ik_result['err_norm']:.6e}")
        if 'computation_time' in ik_result:
            print(f"  Computation time: {ik_result['computation_time'] * 1000:.4f} ms")
        beauty_print("Joint Angles (radians):")
        print(f"  q_ik = {beauty_print_array(ik_result['q'])}")
        beauty_print("Joint Angles (degrees):")
        print(f"  q_ik = {beauty_print_array(np.rad2deg(ik_result['q']))}")
        if ik_result.get('motion_executed', False):
            beauty_print("✓ Robot arm moved to target position")
        else:
            beauty_print("(Motion not executed)")
    else:
        print(f"  Iterations: {ik_result.get('iters', 0)}/{args.max_iters}")
        print(f"  Position error: {ik_result.get('pos_err', float('inf')):.6e} m (Tolerance: {args.pos_tol:.6e} m)")
        print(f"  Orientation error: {ik_result.get('ori_err', float('inf')):.6e} rad (Tolerance: {args.ori_tol:.6e} rad)")
        print(f"  Error message: {ik_result.get('message', 'Unknown error')}")
    print("=" * 60 + "\n")
    
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverse kinematics demo")
    
    # Robot connection settings
    parser.add_argument('--port', type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument('--speed_deg_s', type=int, default=10,  help="Joint motion speed (degrees/second, default: 10, range: 5-400 degrees/second)")
    parser.add_argument('--gripper_type', type=str, default="50mm", help="Gripper type")
    parser.add_argument('--base_link', type=str, default="base_link", help="Base link name, world or base_link etc.")
    parser.add_argument('--end_link', type=str, default="tool0", help="End effector link name, tool0 or link6 etc.")

    # IK Configuration
    parser.add_argument('--end-pose', type=float, nargs=7, 
                       default=[0.26336, -0.17054, +0.4051, -0.560276, +0.357632, +0.745837, +0.043783],
                       help='Target pose (7 floats: px py pz qx qy qz qw)')
    parser.add_argument('--method', type=str, default='dls', 
                       choices=['dls', 'pinv', 'transpose'],
                       help='IK method: dls(damped least squares), pinv(pseudo-inverse), transpose(Jacobian transpose)')
    parser.add_argument('--max-iters', type=int, default=500,  help='Maximum iterations (default: 500)')
    parser.add_argument('--pos-tol', type=float, default=1e-3, help='Position tolerance (default: 1e-3)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (default: 1e-3)')
    parser.add_argument('--num-inits', type=int, default=10,  help='Number of initial guesses (default: 10)')
    parser.add_argument('--init-strategy', type=str, default='current',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform', 'current'],
                        help='Initial guess strategy (default: current=use current joint angles)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Joint limit scale factor (0.0 to 1.0, default: 1.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--backend', type=str, default='cpp',
                        choices=['cpp', 'numpy', 'torch'],
                        help='Computation backend (default: cpp)')
    parser.add_argument('--execute', action='store_true', help='Execute motion to the solved position')
    parser.add_argument('--force-execute', action='store_true', help='Force execute motion to the solved position, regardless of success')
    
    args = parser.parse_args()
    
    main(args)
