# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""Trajectory Executor for Robot Control

This module provides specialized executors for joint space and Cartesian space trajectories.
"""

from __future__ import annotations

import time
from typing import Optional, Callable, Dict, Any
import numpy as np

from robocore.utils.beauty_logger import beauty_print
from robocore.utils.backend import to_numpy


class _BaseTrajectoryExecutor:
    """Base class with shared trajectory execution logic."""
    
    def __init__(
        self,
        robot,
        speed_deg_s: int = 20,
        tolerance: float = 0.5,
        timeout: float = 2.0,
        progress_interval: int = 50,
        initial_delay: float = 0.1,
        wait_for_completion: bool = False,
        use_timing: bool = True
    ):
        """Initialize base trajectory executor.
        
        :param robot: Robot controller instance (must have set_robot_state method)
        :param speed_deg_s: Joint speed in degrees per second
        :param tolerance: Joint tolerance in radians for reaching target
        :param timeout: Timeout for each motion command in seconds
        :param progress_interval: Print progress every N points (0 to disable)
        :param initial_delay: Delay in seconds before starting trajectory execution
        :param wait_for_completion: If True, wait for each point to complete before sending next.
                                    Note: When use_timing=True, this should typically be False to avoid conflicts.
        :param use_timing: If True, use trajectory timing to synchronize execution.
                          When True, execution follows trajectory duration regardless of wait_for_completion.
        """
        # Validate parameter combination
        if wait_for_completion and use_timing:
            beauty_print(
                "Warning: wait_for_completion=True with use_timing=True may cause conflicts. "
                "use_timing controls execution speed based on trajectory times, while "
                "wait_for_completion waits for each point to reach target. "
                "Consider setting wait_for_completion=False when use_timing=True.",
                type="warning"
            )

        self.robot = robot
        self.speed_deg_s = speed_deg_s
        self.tolerance = tolerance
        self.timeout = timeout
        self.progress_interval = progress_interval
        self.initial_delay = initial_delay
        self.wait_for_completion = wait_for_completion
        self.use_timing = use_timing
        
        # Execution state
        self.executed_count = 0
        self.failed_count = 0
        self.start_time = None
        self.trajectory_start_time = None
    
    def _execute_trajectory(
        self,
        joint_angles: np.ndarray,
        trajectory_times: Optional[np.ndarray] = None,
        gripper_values: Optional[np.ndarray] = None,
        initial_tolerance: float = 0.1,
        initial_wait: bool = True,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        on_failure: Optional[Callable[[int, str], bool]] = None
    ) -> Dict[str, Any]:
        """Internal method to execute trajectory (shared by both executors).
        
        :param joint_angles: Array of joint angles [n_points, n_dof] in radians
        :param trajectory_times: Optional array of time points [n_points] in seconds.
                                 If None, points are executed sequentially without timing.
        :param gripper_values: Optional array of gripper values [n_points] (0-1000)
        :param initial_tolerance: Tolerance for initial position (usually tighter)
        :param initial_wait: If True, wait for initial position to be reached
        :param on_progress: Optional callback function(point_index, total_points, time)
        :param on_failure: Optional callback function(point_index, error_message) -> bool
                          Return True to continue, False to stop
        :return: Dictionary with execution statistics
        """
        joint_angles = to_numpy(joint_angles)
        n_points = len(joint_angles)
        
        if trajectory_times is not None:
            trajectory_times = to_numpy(trajectory_times)
            if len(trajectory_times) != n_points:
                raise ValueError(f"trajectory_times length ({len(trajectory_times)}) must match joint_angles length ({n_points})")
        
        if gripper_values is not None:
            gripper_values = to_numpy(gripper_values)
            if len(gripper_values) != n_points:
                raise ValueError(f"gripper_values length ({len(gripper_values)}) must match joint_angles length ({n_points})")
        
        # Display execution info
        duration = trajectory_times[-1] if trajectory_times is not None else 0.0
        control_freq = n_points / duration if duration > 0 else 0.0
        
        beauty_print(f"Executing trajectory with {n_points} points...")
        beauty_print(f"Speed: {self.speed_deg_s} deg/s")
        if duration > 0:
            beauty_print(f"Trajectory duration: {duration:.3f} s")
            beauty_print(f"Control frequency: {control_freq:.1f} Hz")
        
        # Move to first point with wait
        beauty_print("Moving to starting position...")
        first_gripper = int(gripper_values[0]) if gripper_values is not None else None
        
        success = self.robot.set_robot_state(
            target_joints=joint_angles[0],
            gripper_value=first_gripper,
            joint_format='rad',
            speed_deg_s=self.speed_deg_s,
            tolerance=initial_tolerance,
            wait_for_completion=initial_wait,
            timeout=self.timeout
        )
        
        if not success:
            beauty_print("Failed to reach starting position", type="error")
            return {
                'success': False,
                'executed': 0,
                'failed': 1,
                'total': n_points,
                'duration': 0.0
            }
        
        time.sleep(self.initial_delay)
        # Start execution
        self.start_time = time.time()
        self.trajectory_start_time = time.time()
        self.executed_count = 0
        self.failed_count = 0
        
        for i in range(n_points):
            # Get gripper value for this point
            gripper_val = int(gripper_values[i]) if gripper_values is not None else None
            
            # Calculate timing if using trajectory timing
            if self.use_timing and trajectory_times is not None:
                if i == 0:
                    point_target_time = self.trajectory_start_time
                else:
                    point_target_time = self.trajectory_start_time + trajectory_times[i]
            
            # Send command
            success = self.robot.set_robot_state(
                target_joints=joint_angles[i],
                gripper_value=gripper_val,
                joint_format='rad',
                speed_deg_s=self.speed_deg_s,
                tolerance=self.tolerance,
                wait_for_completion=self.wait_for_completion,
                timeout=self.timeout
            )
            # time.sleep(0.1)
            if success:
                self.executed_count += 1
                
                # Progress reporting
                if self.progress_interval > 0 and ((i + 1) % self.progress_interval == 0 or i == 0):
                    t = trajectory_times[i] if trajectory_times is not None else 0.0
                    print(f"  Executed point {i+1}/{n_points} (t={t:.3f}s)")
                
                # Progress callback
                if on_progress is not None:
                    t = trajectory_times[i] if trajectory_times is not None else 0.0
                    on_progress(i, n_points, t)
            else:
                self.failed_count += 1
                error_msg = f"Failed to execute point {i+1}"
                beauty_print(error_msg, type="warning")
                
                # Failure callback
                if on_failure is not None:
                    should_continue = on_failure(i, error_msg)
                    if not should_continue:
                        beauty_print("Stopping execution.", type="warning")
                        break
                else:
                    # # Default behavior: ask user
                    # user_input = input("  Continue execution? (y/n): ").strip().lower()
                    # if user_input != 'y':
                    #     beauty_print("  Stopping execution.", type="warning")
                    #     break
                    pass
            
            # Wait until next point's target time (if using timing and not last point)
            if self.use_timing and trajectory_times is not None and i < n_points - 1:
                current_time = time.time()
                next_target_time = self.trajectory_start_time + trajectory_times[i + 1]
                wait_time = next_target_time - current_time
                
                if wait_time > 0:
                    time.sleep(wait_time)
                elif wait_time < -0.01:  # If we're more than 10ms behind, warn occasionally
                    if (i + 1) % 100 == 0:
                        beauty_print(f"  Warning: Running {abs(wait_time)*1000:.1f}ms behind schedule at point {i+1}", type="warning")
        
        # Calculate statistics
        exec_time = time.time() - self.start_time
        
        beauty_print(f"Trajectory execution completed:")
        print(f"  Executed: {self.executed_count}/{n_points} points")
        print(f"  Failed: {self.failed_count}/{n_points} points")
        print(f"  Total time: {exec_time:.4f} s (target: {duration:.3f} s)" if duration > 0 else f"  Total time: {exec_time:.4f} s")
        if self.executed_count > 0:
            print(f"  Average time per point: {exec_time/self.executed_count*1000:.4f} ms")
        
        return {
            'success': self.failed_count == 0,
            'executed': self.executed_count,
            'failed': self.failed_count,
            'total': n_points,
            'duration': exec_time,
            'target_duration': duration
        }


class JointTrajectoryExecutor(_BaseTrajectoryExecutor):
    """Execute joint space trajectories on the robot.
    
    Optimized for joint space trajectories with sequential execution and gripper support.
    """
    
    def __init__(
        self,
        robot,
        speed_deg_s: int = 20,
        tolerance: float = 0.5,
        timeout: float = 2.0,
        progress_interval: int = 50,
        initial_delay: float = 1.0,
        wait_for_completion: bool = True,
        use_timing: bool = False
    ):
        """Initialize joint trajectory executor.
        
        :param robot: Robot controller instance (must have set_robot_state method)
        :param speed_deg_s: Joint speed in degrees per second
        :param tolerance: Joint tolerance in radians for reaching target
        :param timeout: Timeout for each motion command in seconds
        :param progress_interval: Print progress every N points (0 to disable)
        :param initial_delay: Delay in seconds before starting trajectory execution (default: 1.0)
        :param wait_for_completion: If True, wait for each point to complete (default: True for joint space).
                                    When use_timing=True, consider setting this to False.
        :param use_timing: If True, use trajectory timing to follow planned duration (default: False).
                          When True, execution speed is controlled by trajectory times, not wait_for_completion.
        """
        super().__init__(
            robot=robot,
            speed_deg_s=speed_deg_s,
            tolerance=tolerance,
            timeout=timeout,
            progress_interval=progress_interval,
            initial_delay=initial_delay,
            wait_for_completion=wait_for_completion,
            use_timing=use_timing
        )
    
    def execute(
        self,
        joint_angles: np.ndarray,
        trajectory_times: Optional[np.ndarray] = None,
        gripper_values: Optional[np.ndarray] = None,
        initial_tolerance: float = 0.1,
        initial_wait: bool = True,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        on_failure: Optional[Callable[[int, str], bool]] = None
    ) -> Dict[str, Any]:
        """Execute a joint space trajectory.
        
        :param joint_angles: Array of joint angles [n_points, n_dof] in radians
        :param trajectory_times: Optional array of time points [n_points] in seconds.
                                 If None, points are executed sequentially without timing.
        :param gripper_values: Optional array of gripper values [n_points] (0-1000)
        :param initial_tolerance: Tolerance for initial position (usually tighter)
        :param initial_wait: If True, wait for initial position to be reached
        :param on_progress: Optional callback function(point_index, total_points, time)
        :param on_failure: Optional callback function(point_index, error_message) -> bool
                          Return True to continue, False to stop
        :return: Dictionary with execution statistics
        """
        return self._execute_trajectory(
            joint_angles=joint_angles,
            trajectory_times=trajectory_times,
            gripper_values=gripper_values,
            initial_tolerance=initial_tolerance,
            initial_wait=initial_wait,
            on_progress=on_progress,
            on_failure=on_failure
        )


class CartesianTrajectoryExecutor(_BaseTrajectoryExecutor):
    """Execute Cartesian space trajectories on the robot.
    
    Optimized for Cartesian space trajectories converted to joint space via IK,
    with timing synchronization and IK success rate validation.
    """
    
    def __init__(
        self,
        robot,
        speed_deg_s: int = 20,
        tolerance: float = 0.5,
        timeout: float = 10.0,
        progress_interval: int = 50,
        initial_delay: float = 0.1,
        wait_for_completion: bool = False,
        use_timing: bool = True
    ):
        """Initialize Cartesian trajectory executor.
        
        :param robot: Robot controller instance (must have set_robot_state method)
        :param speed_deg_s: Joint speed in degrees per second
        :param tolerance: Joint tolerance in radians for reaching target
        :param timeout: Timeout for each motion command in seconds
        :param progress_interval: Print progress every N points (0 to disable)
        :param initial_delay: Delay in seconds before starting trajectory execution (default: 0.1)
        :param wait_for_completion: If True, wait for each point to complete (default: False).
                                    When use_timing=True, this should typically be False to avoid conflicts.
        :param use_timing: If True, use trajectory timing to follow planned duration (default: True).
                          When True, execution speed is controlled by trajectory times, not wait_for_completion.
        """
        super().__init__(
            robot=robot,
            speed_deg_s=speed_deg_s,
            tolerance=tolerance,
            timeout=timeout,
            progress_interval=progress_interval,
            initial_delay=initial_delay,
            wait_for_completion=wait_for_completion,
            use_timing=use_timing
        )
    
    def execute(
        self,
        joint_angles: np.ndarray,
        trajectory_times: np.ndarray,
        gripper_values: Optional[np.ndarray] = None,
        initial_tolerance: float = 0.5,
        initial_wait: bool = True,
        ik_success_rate: Optional[float] = None,
        min_success_rate: float = 0.8,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        on_failure: Optional[Callable[[int, str], bool]] = None
    ) -> Dict[str, Any]:
        """Execute a Cartesian space trajectory (converted to joint space via IK).
        
        :param joint_angles: Array of joint angles [n_points, n_dof] in radians (from IK)
        :param trajectory_times: Array of time points [n_points] in seconds (required)
        :param gripper_values: Optional array of gripper values [n_points] (0-1000)
        :param initial_tolerance: Tolerance for initial position
        :param initial_wait: If True, wait for initial position to be reached (default: True)
        :param ik_success_rate: IK success rate (0.0 to 1.0) for validation
        :param min_success_rate: Minimum acceptable IK success rate (default: 0.8)
        :param on_progress: Optional callback function(point_index, total_points, time)
        :param on_failure: Optional callback function(point_index, error_message) -> bool
        :return: Dictionary with execution statistics
        """
        # Check IK success rate if provided
        if ik_success_rate is not None and ik_success_rate < min_success_rate:
            beauty_print(f"Warning: IK success rate is only {ik_success_rate*100:.1f}%", type="warning")
            beauty_print("Some poses may not be executed correctly", type="warning")
            user_input = input("Continue execution anyway? (y/n): ").strip().lower()
            if user_input != 'y':
                beauty_print("Execution cancelled by user", type="warning")
                return {
                    'success': False,
                    'executed': 0,
                    'failed': 0,
                    'total': len(joint_angles),
                    'duration': 0.0,
                    'cancelled': True
                }
        
        # Use timing for Cartesian trajectories
        return self._execute_trajectory(
            joint_angles=joint_angles,
            trajectory_times=trajectory_times,
            gripper_values=gripper_values,
            initial_tolerance=initial_tolerance,
            initial_wait=initial_wait,
            on_progress=on_progress,
            on_failure=on_failure
        )
