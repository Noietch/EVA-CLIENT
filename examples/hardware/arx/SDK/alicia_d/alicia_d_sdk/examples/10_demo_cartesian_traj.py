#!/usr/bin/env python3
# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""Cartesian Spline Trajectory Planning with Inverse Kinematics

This demo demonstrates:
1. Recording Cartesian waypoints by manually dragging the robot (to obtain poses)
2. Generating random Cartesian waypoints automatically
3. Loading Cartesian waypoints from file
4. Generating a smooth spline trajectory in Cartesian space through multiple waypoints
5. Solving inverse kinematics for all poses in the trajectory (batch IK)
6. Executing the trajectory on the robot

Note: Manual dragging here is only to obtain Cartesian poses. 
      If you have known joint configurations, use 09_demo_joint_traj.py directly.

Usage:
python 10_demo_cartesian_traj.py

The program will prompt you to select one of three modes:
1. Manual drag recording mode - Drag robot arm to record Cartesian poses
2. Auto generation mode - Randomly generate Cartesian waypoints
3. Load file mode - Load waypoints from file

# Get help
python 10_demo_cartesian_traj.py --help
"""

import numpy as np
import argparse

import alicia_d_sdk
from alicia_d_sdk.execution import CartesianTrajectoryExecutor
from robocore.transform import make_transform
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy

from alicia_d_sdk.utils.trajectory_utils import (
    handle_manual_record_mode,
    handle_load_file_mode,
    prompt_num_waypoints,
    select_mode,
    display_cartesian_waypoints,
    display_cartesian_trajectory_stats,
    verify_cartesian_waypoints,
    display_ik_results,
    plot_trajectory
)


def mode_manual_record_cartesian(robot):
    """Manual recording mode: drag robot to record Cartesian waypoints.
    
    :param robot: Robot controller instance
    :return: Array of waypoints or None if cancelled
    """
    additional_info = [
        "Note: Manual dragging here is only to obtain Cartesian poses",
        "If you have known joint configurations, you can use 09_demo_joint_traj.py directly"
    ]
    waypoints, _ = handle_manual_record_mode(robot, waypoint_type='cartesian', additional_info=additional_info)
    return waypoints


def mode_auto_generate_cartesian(robot_model, args):
    """Auto generation mode: generate random Cartesian waypoints.
    
    :param robot_model: Robot model instance
    :param args: Command line arguments
    :return: Array of waypoints
    """
    beauty_print("=== Auto Generation Mode ===", type="module", centered=False)

    # Ask for number of waypoints
    num_waypoints = prompt_num_waypoints(default=args.num_waypoints, min_value=2)
    beauty_print(f"Generating {num_waypoints} random waypoints...")

    # Generate random waypoints
    waypoints_list = robot_model.random_pose_batch(
        batch_size=num_waypoints,
        seed=args.seed,
        scale=args.workspace_scale
    )
    waypoints = np.array([to_numpy(wp) for wp in waypoints_list])

    beauty_print(f"Successfully generated {len(waypoints)} waypoints", type="success")

    return waypoints


def mode_load_file_cartesian():
    """Load file mode: load Cartesian waypoints from file.
    
    :return: Array of waypoints or None if failed
    """
    waypoints, _ = handle_load_file_mode(waypoint_type='cartesian')
    return waypoints


def main(args):
    """Main function for Cartesian space trajectory planning and execution."""
    # [0] Initialize robot connection
    beauty_print("Cartesian Spline Planning with IK Batch Solver", type="module")
    
    robot = alicia_d_sdk.create_robot(
        port=args.port,
        gripper_type=args.gripper_type,
        base_link=args.base_link,
        end_link=args.end_link,
        backend=args.backend,
        device=args.device
    )
    robot_model = robot.robot_model

    try:
        # [1] Select mode and get waypoints
        mode_descriptions = [
            "Manual drag recording mode - Drag robot arm to record Cartesian poses",
            "Auto generation mode - Randomly generate Cartesian waypoints",
            "Load file mode - Load waypoints from file"
        ]
        mode = select_mode(mode_descriptions=mode_descriptions)

        if mode == '1':
            # Manual recording mode
            waypoints = mode_manual_record_cartesian(robot)
        elif mode == '2':
            # Auto generation mode
            waypoints = mode_auto_generate_cartesian(robot_model, args)
        else:  # mode == '3'
            # Load file mode
            waypoints = mode_load_file_cartesian()

        if waypoints is None:
            beauty_print("No waypoints obtained, exiting", type="warning")
            robot.disconnect()
            return

        # Display waypoints
        display_cartesian_waypoints(waypoints)

        # [2] Generate Cartesian trajectory
        beauty_print("[2] Generating Spline Trajectory", type="module", centered=False)

        trajectory = robot.plan_cartesian_trajectory(
            waypoints=waypoints,
            duration=args.duration,
            num_points=args.num_points,
        )

        display_cartesian_trajectory_stats(trajectory)
        verify_cartesian_waypoints(trajectory, waypoints)

        # [3] Extract poses and solve IK
        if 'poses' in trajectory:
            target_poses = trajectory['poses']
        else:
            # Build poses from positions and orientations
            positions = trajectory['positions']
            orientations = trajectory['orientations']
            target_poses = np.array([make_transform(orientations[i], positions[i])
                                    for i in range(len(positions))])

        beauty_print("[3] Solving Inverse Kinematics", type="module", centered=False)

        # Get initial joint configuration
        q0 = robot.get_robot_state("joint") if args.init_strategy == 'current' else None
        actual_strategy = 'random' if (q0 is not None and args.init_strategy == 'current') else args.init_strategy

        if q0 is not None:
            beauty_print(f"Using current joint angles as initial guess:")
            print(f"  Current joints (rad): {beauty_print_array(q0)}")
            print(f"  Current joints (deg): {beauty_print_array(np.rad2deg(q0))}")

        # Solve IK for all poses
        ik_result = robot.solve_ik_for_trajectory(
            target_poses=target_poses,
            q_init=q0,
            method=args.method,
            max_iters=args.max_iters,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            num_initial_guesses=args.num_inits,
            initial_guess_strategy=actual_strategy,
            initial_guess_scale=args.init_scale,
            random_seed=args.seed,
            backend=None,  # Use backend set at initialization
            use_previous_solution=True
        )

        display_ik_results(ik_result, trajectory)
        beauty_print("✓ Cartesian spline planning with IK batch solver completed!", type="success")

        joint_angles = ik_result['joint_angles']
        ik_results = ik_result['ik_results']
        success_rate = ik_result['success_rate']

        # [4] Plot trajectory (optional)
        if args.plot:
            beauty_print("[4] Plotting Trajectory", type="module", centered=False)
            plot_trajectory(trajectory, waypoints, plot_type='cartesian',
                            joint_angles=joint_angles, ik_results=ik_results)

        if not args.execute:
            input("\nPress Enter to start trajectory execution...")

        # [5] Execute trajectory
        beauty_print("[5] Executing Trajectory on Robot", type="module", centered=False)

        executor = CartesianTrajectoryExecutor(
            robot=robot,
            speed_deg_s=args.speed_deg_s,
            tolerance=0.5,
            timeout=args.timeout,
            progress_interval=50,
            initial_delay=0.1,
            wait_for_completion=False,
            use_timing=True  # If you want the execution follow the trajectory duration you set, turn to True
        )

        exec_result = executor.execute(
            joint_angles=joint_angles,
            trajectory_times=to_numpy(trajectory['t']),
            gripper_values=None,
            initial_tolerance=0.5,
            ik_success_rate=success_rate,
            min_success_rate=0.8
        )

        if exec_result.get('cancelled', False):
            robot.disconnect()
            return

        beauty_print("Trajectory execution completed", type="success")

    except KeyboardInterrupt:
        beauty_print("\nProgram interrupted by user", type="warning")
    except Exception as e:
        beauty_print(f"Error: {e}", type="error")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
        return {'trajectory': trajectory if 'trajectory' in locals() else None,
                'waypoints': waypoints if 'waypoints' in locals() else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cartesian Spline Planning with IK Batch Solver')
    
    # Robot connection
    parser.add_argument('--port', type=str, default="", help="串口端口 (例如: /dev/ttyUSB0 或 COM3)")
    parser.add_argument('--gripper_type', type=str, default="50mm", help="夹爪类型")
    parser.add_argument('--base_link', type=str, default="base_link", help="基座链路名称")
    parser.add_argument('--end_link', type=str, default="tool0", help="末端执行器链路名称")
    
    # Waypoint settings
    parser.add_argument('--num-waypoints', type=int, default=2, help='Number of waypoints for random generation (default)')
    parser.add_argument('--workspace-scale', type=float, default=0.6, help='Workspace scale (0.0-1.0)')
    
    # Trajectory planning
    parser.add_argument('--duration', type=float, default=5.0, help='Trajectory duration (seconds)')
    parser.add_argument('--num-points', type=int, default=500, help='Number of trajectory points')
    
    # IK settings
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'], help='IK method')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum IK iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-2, help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-2, help='Orientation tolerance (rad)')
    parser.add_argument('--init-scale', type=float, default=0.6, help='Initial guess scale (0.0-1.0)')
    parser.add_argument('--num-inits', type=int, default=2, help='Number of initial guesses')
    parser.add_argument('--init-strategy', type=str, default='current',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform', 'current'],
                        help='Initial guess strategy')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    
    # Execution
    parser.add_argument('--execute', action='store_true', help='Execute trajectory immediately without waiting for user input')
    parser.add_argument('--speed-deg-s', type=int, default=30, help="关节运动速度 (度/秒)")
    parser.add_argument('--timeout', type=float, default=10.0, help='Timeout per command (seconds)')
    
    # Other
    parser.add_argument('--backend', type=str, default='cpp', choices=['cpp', 'numpy', 'torch'],
                        help='Backend (default: cpp)')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--plot', action='store_false', help='Plot trajectory visualization')
    
    main(parser.parse_args())
