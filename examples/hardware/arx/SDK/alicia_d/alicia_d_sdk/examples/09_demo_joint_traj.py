#!/usr/bin/env python3
# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""Joint Space Trajectory Planning and Execution

This demo demonstrates:
1. Recording waypoints by manually dragging the robot
2. Generating random waypoints automatically
3. Loading waypoints from file
4. Planning smooth trajectories through waypoints
5. Executing trajectories on the robot

Usage:
python 09_demo_joint_traj.py
"""

import numpy as np
import argparse

import alicia_d_sdk
from alicia_d_sdk.execution import JointTrajectoryExecutor
from robocore.utils.beauty_logger import beauty_print
from robocore.utils.backend import to_numpy

from alicia_d_sdk.utils.trajectory_utils import (
    handle_manual_record_mode,
    handle_load_file_mode,
    prompt_num_waypoints,
    select_mode,
    display_joint_waypoints,
    display_joint_trajectory_stats,
    plot_trajectory
)


def mode_manual_record(robot):
    """Manual recording mode: drag robot to record waypoints.
    
    :param robot: Robot controller instance
    :return: Tuple of (waypoints, gripper_waypoints) or (None, None) if cancelled
    """
    return handle_manual_record_mode(robot, waypoint_type='joint')


def mode_auto_generate(robot, robot_model, args):
    """Auto generation mode: generate random waypoints.
    
    :param robot: Robot controller instance
    :param robot_model: Robot model instance
    :param args: Command line arguments
    :return: Tuple of (waypoints, gripper_waypoints)
    """
    beauty_print("=== Auto Generation Mode ===", type="module", centered=False)

    # Ask for number of waypoints
    num_waypoints = prompt_num_waypoints(default=args.num_waypoints, min_value=2)
    beauty_print(f"Generating {num_waypoints} random waypoints...")

    # Generate random waypoints
    waypoints = []
    gripper_waypoints = []

    # Optionally use current joints as first waypoint
    if args.use_current_joints:
        robot_state = robot.get_robot_state("joint_gripper")
        if robot_state is not None:
            q_start = robot_state.angles
            g_start = robot_state.gripper
            waypoints.append(to_numpy(q_start))
            gripper_waypoints.append(float(g_start) if g_start is not None else 500.0)
            beauty_print(f"Using current joint angles as first waypoint")
            num_random = num_waypoints - 1
        else:
            beauty_print("Failed to get current joint angles, using random generation", type="warning")
            num_random = num_waypoints
    else:
        num_random = num_waypoints

    # Generate random waypoints
    for i in range(num_random):
        waypoint_seed = args.seed + i if args.seed is not None else None
        q = to_numpy(robot_model.random_q(seed=waypoint_seed, scale=args.joint_scale))
        waypoints.append(q)
        # Random gripper value between 0 and 1000
        if waypoint_seed is not None:
            np.random.seed(waypoint_seed)
        gripper_waypoints.append(float(np.random.uniform(0, 1000)))

    waypoints_array = np.array(waypoints)
    gripper_array = np.array(gripper_waypoints) if gripper_waypoints else None

    beauty_print(f"Successfully generated {len(waypoints_array)} waypoints", type="success")

    return waypoints_array, gripper_array


def mode_load_file():
    """Load file mode: load waypoints from file.
    
    :return: Tuple of (waypoints, gripper_waypoints) or (None, None) if failed
    """
    return handle_load_file_mode(waypoint_type='joint')


def main(args):
    """Main function for joint space trajectory planning and execution."""
    # [0] Initialize robot connection
    beauty_print("Joint Space Trajectory Planning", type="module")
    
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
        mode = select_mode()

        if mode == '1':
            # Manual recording mode
            waypoints, gripper_waypoints = mode_manual_record(robot)
        elif mode == '2':
            # Auto generation mode
            waypoints, gripper_waypoints = mode_auto_generate(robot, robot_model, args)
        else:  # mode == '3'
            # Load file mode
            waypoints, gripper_waypoints = mode_load_file()

        if waypoints is None:
            beauty_print("No waypoints obtained, exiting", type="warning")
            robot.disconnect()
            return

        # Display waypoints
        display_joint_waypoints(waypoints, gripper_waypoints)

        # [2] Generate trajectory
        beauty_print("[2] Generating Joint Space Trajectory", type="module", centered=False)

        planner_name = f"B-Spline (degree={args.bspline_degree})" if args.planner == 'b_spline' else f"Multi-Segment (method={args.segment_method})"
        beauty_print(f"Using {planner_name} planner")

        trajectory = robot.plan_joint_trajectory(
            waypoints=waypoints,
            planner_type=args.planner,
            duration=args.duration if args.planner == 'b_spline' else None,
            num_points=args.num_points if args.planner == 'b_spline' else None,
            bspline_degree=args.bspline_degree,
            segment_method=args.segment_method,
            duration_per_segment=args.duration_per_segment if args.planner == 'multi_segment' else None,
            num_points_per_segment=args.num_points_per_segment if args.planner == 'multi_segment' else None,
            gripper_waypoints=gripper_waypoints
        )

        display_joint_trajectory_stats(trajectory)
        gripper_trajectory = trajectory.get('gripper', None)

        # [3] Plot trajectory (optional)
        if args.plot:
            beauty_print("[3] Plotting Trajectory", type="module", centered=False)
            plot_trajectory(trajectory, waypoints, plot_type='joint')

        # [4] Execute trajectory
        beauty_print("[4] Executing Trajectory on Robot", type="module", centered=False)

        if not args.execute:
            input("\nPress Enter to start trajectory execution...")

        executor = JointTrajectoryExecutor(
            robot=robot,
            speed_deg_s=args.speed_deg_s,
            tolerance=0.5,
            timeout=args.timeout,
            progress_interval=50,
            initial_delay=2.0,
            wait_for_completion=False,
            use_timing=True
        )

        executor.execute(
            joint_angles=to_numpy(trajectory['q']),
            trajectory_times=to_numpy(trajectory['t']),
            gripper_values=gripper_trajectory,
            initial_tolerance=0.1,
        )

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
    parser = argparse.ArgumentParser(description='Joint Space Trajectory Planning and Execution')
    
    # Robot connection
    parser.add_argument('--port', type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument('--gripper_type', type=str, default="50mm", help="Gripper type")
    parser.add_argument('--base_link', type=str, default="base_link", help="Base link name, world or base_link etc.")
    parser.add_argument('--end_link', type=str, default="tool0", help="End effector link name, tool0 or link6 etc.")

    # Trajectory planning
    parser.add_argument('--planner', type=str, default='b_spline', choices=['b_spline', 'multi_segment'],
                        help='Planner type (default: b_spline)')
    parser.add_argument('--duration', type=float, default=10.0, help='Trajectory duration (B-Spline)')
    parser.add_argument('--duration-per-segment', type=float, default=1.0, help='Duration per segment (Multi-Segment)')
    parser.add_argument('--num-points', type=int, default=500, help='Number of points (B-Spline)')
    parser.add_argument('--num-points-per-segment', type=int, default=100, help='Points per segment (Multi-Segment)')
    parser.add_argument('--bspline-degree', type=int, default=5, choices=[3, 5], help='B-Spline degree')
    parser.add_argument('--segment-method', type=str, default='quintic', choices=['cubic', 'quintic'],
                        help='Multi-segment method')
    
    # Auto generation settings
    parser.add_argument('--num-waypoints', type=int, default=6, help='Number of waypoints for random generation (default)')
    parser.add_argument('--joint-scale', type=float, default=0.6, help='Scale factor for random joints (0.0-1.0)')
    parser.add_argument('--use-current-joints', action='store_true', help='Use current joints as first waypoint in auto generation')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')

    # Execution
    parser.add_argument('--execute', action='store_true', help='Execute trajectory immediately without waiting for user input')
    parser.add_argument('--speed-deg-s', type=int, default=100, help="Joint motion speed (degrees/second)")
    parser.add_argument('--timeout', type=float, default=10.0, help='Timeout per command (seconds)')
    
    # Other
    parser.add_argument('--backend', type=str, default='cpp', choices=['cpp', 'numpy', 'torch'],
                        help='Backend (default: cpp)')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--plot', action='store_false', help='Plot trajectory visualization')
    
    main(parser.parse_args())
