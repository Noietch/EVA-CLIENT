# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
SynriaRobotAPI - User-level API

Responsibilities:
- Provide concise unified user interface
- High-level motion command encapsulation
- State query interface
- System control functions
- Parameter validation and error handling
"""

from alicia_d_sdk.utils import precise_sleep
from alicia_d_sdk.utils import logger
from alicia_d_sdk.hardware.data_parser import JointState
from alicia_d_sdk.hardware import ServoDriver
import robocore as rc
from synriard import get_model_path
from robocore.transform import matrix_to_euler, matrix_to_quaternion
from robocore.utils.backend import to_numpy
from robocore.kinematics import forward_kinematics
from robocore.transform import make_transform, quaternion_to_matrix
from robocore.modeling import RobotModel
from robocore.kinematics import inverse_kinematics
import json
import numpy as np
from typing import List, Optional, Dict, Union, Tuple, Any, Literal
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


BackendName = Literal['cpp', 'numpy', 'torch']


class SynriaRobotAPI:
    """Synria robot arm API - provides unified user interface"""

    def __init__(self,
                 servo_driver: ServoDriver,
                 robot_model: RobotModel,
                 auto_connect: bool = True,
                 backend: Optional[BackendName] = None,
                 device: str = "cpu"):
        """Initialize robot API.

        :param servo_driver: Servo driver instance (low-level hardware)
        :param robot_model: Pre-loaded robot model (RoboCore RobotModel)
        :param auto_connect: Auto connect to robot on initialization
        :param backend: Computation backend, 'cpp', 'numpy', or 'torch' (default: None, uses 'cpp')
        :param device: Device for torch backend, 'cpu' or 'cuda' (default: 'cpu', ignored for 'cpp' and 'numpy')
        """
        self.servo_driver = servo_driver
        self.data_parser = servo_driver.data_parser  # Direct access to data parser
        self.robot_model = robot_model
        self.debug_mode = servo_driver.debug_mode  # Access debug mode from servo driver

        # Set backend if provided, otherwise use default 'cpp'
        if backend is not None:
            rc.set_backend(backend, device=device)
        else:
            rc.set_backend('cpp')

        # Higher-level helpers
        self.robot_type = None
        # Cache for joint angles to maintain position when only gripper is set
        self._cached_joint_angles: Optional[List[float]] = None
        if auto_connect:
            self.connect()

        # ==================== Connection Management ===================

    def connect(self) -> bool:
        """Connect to robot and detect firmware version."""
        if self.is_connected():
            return True

        result = self.servo_driver.connect()
        if result:
            try:
                # Initialize state
                self.get_robot_state("joint_gripper")
                self._robot_type()
                # Initialize joint angle cache with current joint angles
                current_joints = self.get_robot_state("joint")
                if current_joints is not None:
                    self._cached_joint_angles = list(current_joints)
                logger.info("Synria Robot Connected successfully.")
                return True
            except Exception as e:
                logger.error(f"Hardware initialization failed after serial connection: {e}")
                return False
        return False

    def disconnect(self):
        """Disconnect from robot and stop update threads."""
        self.servo_driver.stop_update_thread()
        self.servo_driver.disconnect()

    def is_connected(self) -> bool:
        """Check if robot is connected.
        """
        return self.servo_driver.serial_comm.is_connected()

    # ==================== Get Robot Information ====================

    def get_robot_state(self, info_type: str = "joint_gripper", timeout: float = 1.0, cache: bool = True) -> Optional[Union[JointState, Dict, List[float], str, float]]:
        """
        Unified API to get robot state information.

        :param info_type: Type of information to get. Options:
            - "joint_gripper": Returns JointState (arm joint angles, gripper value, timestamp, run_status_text)
            - "joint": Returns List[float] of arm joint angles (radians) only
            - "gripper": Returns float gripper value (0-1000) only
            - "version": Returns Dict with serial_number, hardware_version, firmware_version
            - "temperature": Returns List[float] of temperatures in Celsius
            - "velocity": Returns List[float] of velocities in degrees per second
            - "gripper_type": Returns str (e.g., "50mm" or "100mm") or None if unavailable
            - "self_check": Returns Dict with self-check data (or None if failed)
        :param timeout: Maximum time to wait for response in seconds
        :return: Requested data or None if failed
        """
        # Special handling for gripper_type: try cache first, then hardware query
        if info_type == "gripper_type" and cache:
            return self._get_gripper_type_with_cache(timeout)

        # Joint and gripper are acquired together from hardware using the "joint" command
        if info_type in ("joint_gripper", "joint", "gripper"):
            if not self.servo_driver.acquire_info("joint_gripper", wait=True, timeout=timeout):
                logger.error(f"Failed to get joint/gripper data within timeout period")
                return None
            return self.data_parser.get_info(info_type)

        # Other info types map directly to hardware commands
        if not self.servo_driver.acquire_info(info_type, wait=True, timeout=timeout):
            logger.error(f"Failed to get {info_type} data within timeout period")
            return None

        result = self.data_parser.get_info(info_type)

        return result

    def get_pose(self, backend: Optional[BackendName] = None) -> Optional[Union[List[float], Dict]]:
        """Get current end-effector pose.

        :param backend: Computation backend, 'cpp', 'numpy', or 'torch' (default: None, uses backend set at initialization)
        :return: Dictionary with position, rotation, euler_xyz, quaternion_xyzw, transform, or None if failed
        """
        # Set backend globally (forward_kinematics uses global backend)
        if backend is not None:
            rc.set_backend(backend)
        joint_angles = self.get_robot_state("joint")
        if joint_angles is None:
            logger.error("Failed to get joint angles")
            return None

        T_fk = forward_kinematics(
            self.robot_model,
            joint_angles,
            return_end=True
        )

        position_fk = T_fk[:3, 3]
        rotation_fk = T_fk[:3, :3]
        euler_fk = matrix_to_euler(rotation_fk, seq='xyz')
        quat_fk = matrix_to_quaternion(rotation_fk)

        return {
            'transform': T_fk,
            'position': position_fk,
            'rotation': rotation_fk,
            'euler_xyz': euler_fk,
            'quaternion_xyzw': quat_fk
        }

    # ==================== Robot Control ====================

    def set_home(self, speed_deg_s: Union[int, float, List[float], np.ndarray] = 10, gripper_speed_deg_s: Optional[float] = 483.4):
        """Move robot to home position and wait until near zero.

        :param speed_deg_s: Speed in degrees per second. Can be int/float (same for all joints) or list/array (per-joint speeds), default 10
        :param gripper_speed_deg_s: Gripper speed in degrees per second. If None, uses default 5500 ticks/s (≈483.4 deg/s)
        """
        # time.sleep(0.1)
        home_joints = [0.0] * 6
        self.set_robot_state(
            target_joints=home_joints,
            gripper_value=1000,
            speed_deg_s=speed_deg_s,
            gripper_speed_deg_s=gripper_speed_deg_s,
            wait_for_completion=True
        )

    def set_robot_state(self,
                        target_joints: Optional[List[float]] = None,
                        gripper_value: Optional[int] = None,
                        joint_format: str = 'rad',
                        speed_deg_s: Union[int, float, List[float], np.ndarray] = 10,
                        gripper_speed_deg_s: Optional[float] = 483.4,
                        tolerance: float = 0.1,
                        timeout: float = 10.0,
                        wait_for_completion: bool = True) -> bool:
        """Set joint angles and/or gripper in a single combined command.

        :param target_joints: Optional target joint angles. If None, keeps current
        :param gripper_value: Optional gripper value (0-1000). If None, keeps current
        :param joint_format: Unit format for joints, 'rad' or 'deg'
        :param speed_deg_s: Speed in degrees per second. Can be int/float (same for all joints) or list/array (per-joint speeds, 4.39-439.45 deg/s), default 10
        :param gripper_speed_deg_s: Gripper speed in degrees per second. If None, uses default 5500 ticks/s (≈483.4 deg/s)
        :param tolerance: Rad, acceptable abs distance to target for joints
        :param timeout: Seconds, maximum wait time
        :param wait_for_completion: If True, wait until target reached
        :return: True if successful, False otherwise
        """
        # Convert joint format if needed
        if target_joints is not None:
            if joint_format == 'deg':
                target_joints = [a * np.pi / 180.0 for a in target_joints]
            # Update cache when joints are explicitly set
            self._cached_joint_angles = list(target_joints)
        else:
            # When joints are not explicitly set, use cached joint angles to maintain position
            # This prevents arm from dropping due to gravity when only gripper is set
            if self._cached_joint_angles is None:
                # If cache is empty, try to get current joints from hardware
                current_joints = self.get_robot_state("joint")
                if current_joints is not None:
                    self._cached_joint_angles = list(current_joints)
                else:
                    logger.warning("No cached joint angles available and cannot read from hardware, using zeros")
                    self._cached_joint_angles = [0.0] * 6
            target_joints = self._cached_joint_angles

        # Use unified method
        success = self.servo_driver.set_joint_and_gripper(
            joint_angles=target_joints,
            gripper_value=gripper_value,
            speed_deg_s=speed_deg_s,
            gripper_speed_deg_s=gripper_speed_deg_s
        )

        if not success:
            logger.error("Failed to set robot target")
            return False

        # Wait for completion if requested
        if wait_for_completion and (target_joints is not None or gripper_value is not None):
            joint_result = True
            gripper_result = True

            # Wait for joints if target_joints is provided
            if target_joints is not None:
                joint_result = self._wait_for_joint_target(
                    target_joints=target_joints,
                    tolerance=tolerance,
                    timeout=timeout,
                    log_prefix="等待关节接近目标"
                )

            # Wait for gripper if gripper_value is provided
            if gripper_value is not None:
                gripper_tolerance = 5.0  # Increase tolerance to 5% for more reliable completion
                gripper_timeout = timeout  # Use max 4 seconds for gripper wait
                start_time = time.time()

                # Give hardware some time to start responding
                time.sleep(0.05)

                # Check gripper position with timeout
                gripper_reached = False
                while time.time() - start_time < gripper_timeout:
                    current_gripper = self.get_robot_state("gripper")
                    if current_gripper is not None:
                        if abs(current_gripper - gripper_value) <= gripper_tolerance:
                            gripper_reached = True
                            break
                    time.sleep(0.05)

                # If we didn't verify position but command was sent, still consider success
                if not gripper_reached:
                    final_check = self.get_robot_state("gripper")
                    if final_check is None:
                        # State unavailable, but command was sent
                        gripper_result = True
                    else:
                        # Check one more time with tolerance
                        gripper_result = abs(final_check - gripper_value) <= gripper_tolerance
                else:
                    gripper_result = True

            return joint_result and gripper_result

        # Either waiting was not requested, or no targets were provided.
        return True

    def set_pose(self,
                 target_pose: List[float],
                 backend: Optional[BackendName] = None,
                 method: str = 'dls',
                 pos_tol: float = 1e-3,
                 ori_tol: float = 1e-3,
                 max_iters: int = 500,
                 num_initial_guesses: int = 10,
                 initial_guess_strategy: str = 'current',
                 initial_guess_scale: float = 1.0,
                 random_seed: Optional[int] = None,
                 speed_deg_s: Union[int, float, List[float], np.ndarray] = 10,
                 gripper_speed_deg_s: Optional[float] = 483.4,
                 execute: bool = True,
                 force_execute: bool = False) -> Dict:
        """Move end-effector to target pose using inverse kinematics.

        :param target_pose: Target pose as [x, y, z, qx, qy, qz, qw]
        :param backend: Computation backend, 'cpp', 'numpy', or 'torch' (default: None, uses backend set at initialization)
        :param method: IK solver method, 'dls', 'pinv', or 'transpose'
        :param pos_tol: Position tolerance in meters
        :param ori_tol: Orientation tolerance in radians
        :param max_iters: Maximum number of iterations
        :param num_initial_guesses: Number of initial guesses for multi-start
        :param initial_guess_strategy: Initial guess strategy ('zero', 'random', 'sobol', 'latin', 'center', 'uniform', 'current')
        :param initial_guess_scale: Scale factor for initial guesses (0.0 to 1.0)
        :param random_seed: Random seed for reproducibility
        :param speed_deg_s: Motion speed in degrees per second. Can be int/float (same for all joints) or list/array (per-joint speeds)
        :param gripper_speed_deg_s: Gripper speed in degrees per second. If None, uses default 5500 ticks/s (≈483.4 deg/s)
        :param execute: Execute motion if True and IK succeeds
        :param force_execute: Force execute motion even if IK failed (requires q to be available)
        :return: Dictionary with success, q, iters, pos_err, ori_err, message, motion_executed, computation_time
        """

        # Set backend globally (inverse_kinematics uses global backend)
        if backend is not None:
            rc.set_backend(backend)

        # Get initial guess based on strategy
        if initial_guess_strategy == 'current':
            q0 = self.get_robot_state("joint")
            if q0 is None:
                return {
                    'success': False,
                    'message': '无法获取当前关节角度',
                    'q': None,
                    'motion_executed': False
                }
            # Use 'random' strategy with current joints as base
            actual_strategy = 'random'
        else:
            q0 = None
            actual_strategy = initial_guess_strategy

        # Convert pose to transformation matrix
        position = np.array(target_pose[:3])
        quaternion = np.array(target_pose[3:])
        rotation_matrix = quaternion_to_matrix(quaternion)
        pose_matrix = make_transform(rotation_matrix, position)

        # Solve inverse kinematics
        start_time = time.time()
        ik_result = inverse_kinematics(
            self.robot_model,
            pose_matrix,
            q0=q0,
            method=method,
            max_iters=max_iters,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            num_initial_guesses=num_initial_guesses,
            initial_guess_strategy=actual_strategy,
            initial_guess_scale=initial_guess_scale,
            random_seed=random_seed,
            use_analytic_jacobian=True
        )
        elapsed_time = time.time() - start_time

        # Convert to numpy and extract results
        q_ik = to_numpy(ik_result['q']) if ik_result.get('q') is not None else None

        # Extract error information (handle both single value and list)
        iters = ik_result.get('iters', 0)
        pos_err = ik_result.get('pos_err', float('inf'))
        ori_err = ik_result.get('ori_err', float('inf'))
        err_norm = ik_result.get('err_norm', None)

        # Handle list results (from batch processing)
        if isinstance(iters, list):
            iters = iters[0] if iters else 0
        if isinstance(pos_err, list):
            pos_err = pos_err[0] if pos_err else float('inf')
        if isinstance(ori_err, list):
            ori_err = ori_err[0] if ori_err else float('inf')
        if isinstance(err_norm, list):
            err_norm = err_norm[0] if err_norm else None

        # Check if IK succeeded
        ik_success = ik_result.get('success', False)
        if isinstance(ik_success, list):
            ik_success = ik_success[0] if ik_success else False

        # Update result with normalized values
        ik_result['iters'] = iters
        ik_result['pos_err'] = pos_err
        ik_result['ori_err'] = ori_err
        if err_norm is not None:
            ik_result['err_norm'] = err_norm
        ik_result['computation_time'] = elapsed_time

        # Execute motion if requested
        if ik_success and execute:
            result = self.set_robot_state(
                target_joints=q_ik,
                joint_format='rad',
                speed_deg_s=speed_deg_s,
                gripper_speed_deg_s=gripper_speed_deg_s,
                wait_for_completion=True
            )
            ik_result['motion_executed'] = result
        elif force_execute and q_ik is not None:
            # Force execute even if IK failed
            result = self.set_robot_state(
                target_joints=q_ik,
                joint_format='rad',
                speed_deg_s=speed_deg_s,
                gripper_speed_deg_s=gripper_speed_deg_s,
                wait_for_completion=True,
                timeout=10
            )
            ik_result['motion_executed'] = result
        else:
            ik_result['motion_executed'] = False

        return ik_result

    # ==================== Advanced Trajectory Methods ====================

    def plan_joint_trajectory(
        self,
        waypoints: np.ndarray,
        planner_type: str = 'b_spline',
        duration: Optional[float] = None,
        num_points: int = 800,
        bspline_degree: int = 5,
        segment_method: str = 'quintic',
        duration_per_segment: Optional[float] = None,
        num_points_per_segment: int = 100,
        gripper_waypoints: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Plan joint space trajectory through waypoints.

        :param waypoints: Array of joint waypoints [n_waypoints, n_dof] in radians
        :param planner_type: Planner type, 'b_spline' or 'multi_segment'
        :param duration: Total trajectory duration in seconds (for B-Spline)
        :param num_points: Number of points in trajectory (for B-Spline)
        :param bspline_degree: B-Spline degree, 3 (cubic) or 5 (quintic)
        :param segment_method: Multi-segment method, 'cubic' or 'quintic'
        :param duration_per_segment: Duration per segment in seconds (for Multi-Segment)
        :param num_points_per_segment: Number of points per segment (for Multi-Segment)
        :param gripper_waypoints: Optional array of gripper values [n_waypoints] (0-1000)
        :return: Dictionary with trajectory data including 't', 'q', 'qd', 'qdd', and optionally 'gripper'
        """
        from robocore.planning import BSplinePlanner, MultiSegmentPlanner
        from robocore.utils.backend import to_numpy

        waypoints = to_numpy(waypoints)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)

        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        # Create planner
        if planner_type == 'b_spline':
            planner = BSplinePlanner(degree=bspline_degree)
        elif planner_type == 'multi_segment':
            planner = MultiSegmentPlanner(method=segment_method)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}. Must be 'b_spline' or 'multi_segment'")

        # Plan trajectory
        if planner_type == 'b_spline':
            trajectory = planner.plan(
                waypoints=waypoints,
                duration=duration,
                num_points=num_points
            )
        else:  # multi_segment
            if duration_per_segment is None:
                duration_per_segment = 1.0
            trajectory = planner.plan(
                waypoints=waypoints,
                durations=duration_per_segment,
                num_points_per_segment=num_points_per_segment
            )

        # Interpolate gripper values if provided
        if gripper_waypoints is not None:
            gripper_waypoints = to_numpy(gripper_waypoints)
            t_waypoints = np.linspace(0, trajectory['t'][-1], len(gripper_waypoints))
            t_traj = to_numpy(trajectory['t'])
            # Use linear interpolation for gripper
            gripper_trajectory = np.interp(t_traj, t_waypoints, gripper_waypoints)
            # Clip to valid range [0, 1000]
            gripper_trajectory = np.clip(gripper_trajectory, 0, 1000)
            trajectory['gripper'] = gripper_trajectory

        # Add waypoints to trajectory for reference
        trajectory['waypoints'] = waypoints

        return trajectory

    def plan_cartesian_trajectory(
        self,
        waypoints: np.ndarray,
        duration: Optional[float] = None,
        num_points: int = 100,
        backend: Optional[BackendName] = None
    ) -> Dict[str, Any]:
        """Plan Cartesian space spline trajectory through waypoints.

        :param waypoints: Array of waypoint poses [n_waypoints, 4, 4] (transformation matrices)
                          or [n_waypoints, 3] (positions only, will use identity orientation)
        :param duration: Total trajectory duration in seconds (optional, auto-estimated if None)
        :param num_points: Number of points in trajectory
        :param backend: Computation backend, 'cpp', 'numpy', or 'torch' (default: None, uses backend set at initialization)
        :return: Dictionary with 't', 'poses', 'positions', 'orientations', 'velocities', 'accelerations'
        """
        import robocore as rc
        from robocore.planning import SplineCurvePlanner
        from robocore.utils.backend import to_numpy

        # Set backend for planning (forward_kinematics uses global backend)
        if backend is not None:
            rc.set_backend(backend)

        # Ensure waypoints are numpy arrays
        if isinstance(waypoints, list):
            waypoints = np.array([to_numpy(wp) for wp in waypoints])
        else:
            waypoints = to_numpy(waypoints)

        # Create planner
        planner = SplineCurvePlanner()

        # Plan trajectory
        trajectory = planner.plan(
            waypoints=waypoints,
            duration=duration,
            num_points=num_points
        )

        # Convert to numpy if needed
        for key in ['t', 'positions', 'orientations', 'velocities', 'accelerations']:
            if key in trajectory:
                trajectory[key] = to_numpy(trajectory[key])

        if 'poses' in trajectory:
            trajectory['poses'] = np.array([to_numpy(pose) for pose in trajectory['poses']])

        # Add waypoint positions for reference
        if waypoints.ndim == 3 and waypoints.shape[1:] == (4, 4):
            waypoint_positions = np.array([wp[:3, 3] for wp in waypoints])
        elif waypoints.ndim == 2 and waypoints.shape[1] == 3:
            waypoint_positions = waypoints
        else:
            waypoint_positions = None

        if waypoint_positions is not None:
            trajectory['waypoints'] = waypoint_positions

        return trajectory

    def solve_ik_for_trajectory(
        self,
        target_poses: np.ndarray,
        q_init: Optional[List[float]] = None,
        method: str = 'dls',
        max_iters: int = 100,
        pos_tol: float = 1e-2,
        ori_tol: float = 1e-2,
        num_initial_guesses: int = 5,
        initial_guess_strategy: str = 'random',
        initial_guess_scale: float = 0.6,
        random_seed: Optional[int] = None,
        backend: Optional[BackendName] = None,
        use_previous_solution: bool = True
    ) -> Dict[str, Any]:
        """Solve inverse kinematics for a sequence of Cartesian poses.

        :param target_poses: Array of target poses [n_poses, 4, 4] (transformation matrices)
        :param q_init: Initial joint configuration (uses current joints if None)
        :param method: IK solver method, 'dls', 'pinv', or 'transpose'
        :param max_iters: Maximum IK iterations per pose
        :param pos_tol: Position tolerance in meters
        :param ori_tol: Orientation tolerance in radians
        :param num_initial_guesses: Number of initial guesses for multi-start
        :param initial_guess_strategy: Initial guess strategy ('zero', 'random', 'sobol', 'latin', 'center', 'uniform')
        :param initial_guess_scale: Scale factor for initial guesses (0.0 to 1.0)
        :param random_seed: Random seed for reproducibility
        :param backend: Computation backend, 'cpp', 'numpy', or 'torch' (default: None, uses backend set at initialization)
        :param use_previous_solution: If True, use previous solution as initial guess (ensures continuity)
        :return: Dictionary with 'joint_angles', 'ik_results', 'success_rate', 'statistics'
        """
        import robocore as rc
        from robocore.kinematics.ik import inverse_kinematics
        from robocore.utils.backend import to_numpy
        import time

        # Set backend if specified (inverse_kinematics uses global backend)
        if backend is not None:
            rc.set_backend(backend)

        target_poses = to_numpy(target_poses)
        n_poses = len(target_poses)

        # Get initial joint configuration
        if q_init is None:
            q_init = self.get_robot_state("joint")
            if q_init is None:
                raise ValueError("Cannot get current joint angles. Please provide q_init.")

        q_init = np.array(q_init)
        q_current = q_init.copy()

        # Solve IK for each pose
        ik_results = []
        joint_angles = []
        success_count = 0

        start_time = time.time()

        for i, target_pose in enumerate(target_poses):
            # Use previous solution as initial guess if enabled
            if use_previous_solution and i > 0:
                q0 = q_current
                # Use fewer initial guesses for subsequent poses (we have a good initial guess)
                num_inits = min(5, num_initial_guesses) if num_initial_guesses > 1 else 1
                strategy = 'random'
            else:
                q0 = q_init if i == 0 else q_current
                num_inits = num_initial_guesses
                strategy = initial_guess_strategy if i == 0 else 'random'

            # Solve IK (max_iters, pos_tol, ori_tol are passed via solver_kwargs)
            result = inverse_kinematics(
                self.robot_model,
                target_pose,
                q0=q0,
                method=method,
                num_initial_guesses=num_inits,
                initial_guess_strategy=strategy,
                initial_guess_scale=initial_guess_scale,
                random_seed=random_seed,
                max_iters=max_iters,
                pos_tol=pos_tol,
                ori_tol=ori_tol
            )

            ik_results.append(result)

            if result['success']:
                joint_angles.append(result['q'])
                q_current = np.array(result['q'])
                success_count += 1
            else:
                # Use previous solution if available, otherwise use zeros
                if len(joint_angles) > 0:
                    joint_angles.append(joint_angles[-1])
                else:
                    joint_angles.append(np.zeros(len(self.robot_model._chain_actuated)))

        ik_time = time.time() - start_time

        joint_angles = np.array(joint_angles)
        success_rate = success_count / n_poses if n_poses > 0 else 0.0

        # Calculate statistics
        pos_errors = [r['pos_err'] for r in ik_results if r['success']]
        ori_errors = [r['ori_err'] for r in ik_results if r['success']]

        statistics = {
            'total_poses': n_poses,
            'successful': success_count,
            'failed': n_poses - success_count,
            'success_rate': success_rate,
            'computation_time': ik_time,
            'avg_time_per_pose': ik_time / n_poses if n_poses > 0 else 0.0,
            'pos_error_mean': np.mean(pos_errors) if pos_errors else None,
            'pos_error_max': np.max(pos_errors) if pos_errors else None,
            'ori_error_mean': np.mean(ori_errors) if ori_errors else None,
            'ori_error_max': np.max(ori_errors) if ori_errors else None
        }

        return {
            'joint_angles': joint_angles,
            'ik_results': ik_results,
            'success_rate': success_rate,
            'statistics': statistics
        }

    def torque_control(self, command: str, timeout: float = 1.0) -> bool:
        """Enable or disable robot torque.

        :param command: 'on' or 'off'
        :param timeout: Maximum time to wait for response in seconds
        :return: True if successful
        """
        if command == "on":
            return self.servo_driver.acquire_info("torque_on", wait=True, timeout=timeout)
        elif command == "off":
            return self.servo_driver.acquire_info("torque_off", wait=True, timeout=timeout)
        else:
            logger.error("command parameter must be 'on' or 'off'")
            return False

    def zero_calibration(self) -> bool:
        """Execute zero position calibration procedure.

        :return: True if calibration successful
        """
        logger.warning("This operation is irreversible and will change the factory zero position, please operate with caution")
        logger.info("Starting zero calibration, robot arm will lose torque")
        logger.info("Press Enter to continue, Ctrl+C to cancel...")
        input()
        if not self.torque_control('off'):
            logger.error("Failed to disable torque")
            return False
        logger.info("Please manually drag the robot arm to zero position, then press Enter to continue...")
        input()

        if not self.servo_driver.acquire_info("zero_cali", wait=True, timeout=2.0):
            logger.error("Zero calibration failed")
            return False
        time.sleep(0.1)
        self.servo_driver.acquire_info("torque_on", wait=True, timeout=1.0)
        return True

    def print_state(self, continuous: bool = False, output_format: str = "deg", fps: float = 200.0):
        """Print current robot state.

        :param continuous: Print continuously if True, once if False
        :param output_format: Angle format, 'deg' or 'rad'
        :param fps: Target frames per second for continuous mode. Default 200 Hz
        """
        robot_type = self._robot_type()

        def _print_once(robot_type):
            pose = None
            state = self.get_robot_state("joint_gripper")
            # get the gripper type

            temperature = self.get_robot_state("temperature", timeout=5.0)
            velocity = self.get_robot_state("velocity")
            self.get_robot_state("self_check")
            if state is None:
                logger.warning("Failed to get joint state")
                return

            joints = state.angles
            gripper = state.gripper
            status = state.run_status_text

            # Extract pose only for follower
            if robot_type == "follower":
                pose = self.get_pose()

            # Format joints for printing
            if output_format == 'deg':
                joint_out = np.round(np.array(joints) * 180.0 / np.pi, 2)
                unit = "°"
            else:
                joint_out = np.round(joints, 3)
                unit = "rad"
            logger.info(f"Joint angles ({unit}): {joint_out.tolist()}, Gripper (0-1000): {gripper}")
            if status != "idle":
                logger.info(f"Button status: {status}")

            if pose is not None:
                quaternion = pose['quaternion_xyzw']
                position = pose['position']
                logger.info(f"Position (xyz /m): {np.round(position, 3).tolist()}, Quaternion (qx, qy, qz, qw): {np.round(quaternion, 3).tolist()}")

            if temperature is not None:
                logger.info(f"Servo temperature (°C): {np.round(temperature, 1).tolist()}")
            if velocity is not None:
                logger.info(f"Servo velocity (deg/s): {np.round(velocity, 1).tolist()}")

            print("\n")
        if continuous:
            # For high frequency (>= 100 Hz), use smaller spin_threshold for better efficiency
            # For 200 Hz (5ms interval), use 2ms spin_threshold to allow some sleep time
            interval = 1 / fps
            spin_threshold = 0.002 if interval <= 0.010 else 0.010  # 2ms for high freq, 10ms for low freq
            logger.info(f"Starting continuous state printing, press Ctrl+C to stop (target FPS: {fps})")
            try:
                while True:
                    start_time = time.perf_counter()
                    _print_once(robot_type)
                    dt_time = time.perf_counter() - start_time
                    precise_sleep(interval - dt_time, spin_threshold=spin_threshold)
            except KeyboardInterrupt:
                logger.info("Stopped continuous state printing")
        else:
            _print_once(robot_type)

    def _generate_random_q(self, scale: float = 0.5) -> List[float]:
        """Generate random joint configuration within limits.

        :param scale: Range scale factor within joint limits
        :return: Random joint angles in radians
        """
        rng = np.random.default_rng()
        q = [0.0] * self.robot_model.num_dof()

        for js in self.robot_model._actuated:
            lo, hi = -1.0, 1.0
            if js.limit:
                if js.limit[0] is not None:
                    lo = js.limit[0]
                if js.limit[1] is not None:
                    hi = js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.5 * (hi - lo) * scale
            q[js.index] = float(rng.uniform(mid - span, mid + span))

        return q

    def _wait_for_joint_target(self,
                               target_joints: Optional[List[float]],
                               tolerance: float,
                               timeout: float,
                               log_prefix: str = "等待关节接近目标") -> bool:
        """Wait until all joints reach target angles.

        :param target_joints: Target joint angles in radians. If None, returns True immediately
        :param tolerance: Rad, acceptable abs distance to target for all joints
        :param timeout: Seconds, maximum wait time
        :param log_prefix: Log message prefix
        :return: True if target reached, False if timeout
        """
        # If no joint target is specified, there is nothing to wait for.
        if target_joints is None:
            logger.debug("No joint target specified, skip joint waiting.")
            return True

        start_time = time.time()
        # logger.info(f"{log_prefix}...")

        while time.time() - start_time < timeout:
            current_joints = self.get_robot_state("joint")
            if current_joints is not None:
                if all(abs(a - b) <= tolerance for a, b in zip(current_joints, target_joints)):
                    # logger.info("Reached target position")
                    return True
        logger.warning("Timeout waiting for joints to reach target")
        joints = self.get_robot_state("joint")
        logger.warning(f"Target joint angles: {target_joints}")
        logger.warning(f"Current joint angles: {joints}")

        return False

    def __del__(self):
        try:
            self.disconnect()
        except Exception as e:
            logger.error(f"SynriaRobotAPI destructor exception: {e}")

    def _robot_type(self) -> str:
        if self.robot_type is not None:
            return self.robot_type

        version = self.get_robot_state("version")
        if version is None:
            return None

        serial_number = version.get("serial_number")
        # ADFS for follower, ADLS for leader
        if serial_number.startswith("ADF"):
            self.robot_type = "follower"
        elif serial_number.startswith("ADL"):
            self.robot_type = "leader"

        return self.robot_type

    def _get_gripper_type_with_cache(self, timeout: float = 1.0) -> Optional[str]:
        """Get gripper type with caching support.

        First tries to load from JSON cache, then queries hardware if needed.
        Returns None with warning if hardware query fails (non-critical).

        :param timeout: Maximum time to wait for hardware response in seconds
        :return: Gripper type name (e.g., "50mm" or "100mm"), or None if unavailable
        """
        # JSON file path in the same folder as this module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, "gripper_type.json")

        # 1) Try to load cached gripper type from JSON file (no serial communication)
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                cached_type = data.get("type_name")
                if isinstance(cached_type, str) and cached_type:
                    return cached_type
            except Exception as e:
                logger.warning(f"Failed to load cached gripper type from JSON, will try hardware query: {e}")

        # 2) If no valid cache, actively query hardware
        # Try with wait=True first to get response reliably
        if not self.servo_driver.acquire_info("gripper_type", wait=True, timeout=timeout):
            logger.warning("Failed to get gripper_type data within timeout period")
            return None

        result = self.data_parser.get_info("gripper_type")
        if result is None:
            logger.warning("Gripper type (50mm or 100mm) should be defined by parameters")
            return None

        # Save to JSON file for future use
        self._save_gripper_type_to_json(result)

        return result

    def _save_gripper_type_to_json(self, gripper_type: str):
        """Save gripper type to JSON file in the same folder as this module."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, "gripper_type.json")

        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump({"type_name": gripper_type}, f, indent=2, ensure_ascii=False)
            if self.debug_mode:
                logger.debug(f"Saved gripper type '{gripper_type}' to {json_file_path}")
        except Exception as e:
            logger.error(f"Failed to save gripper type to JSON file: {e}")
