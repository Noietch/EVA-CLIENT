# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
API Layer - User Interface

Provides a concise and unified user interface, including:
- High-level motion commands (joint control, Cartesian trajectories)
- State query interfaces (joint angles, gripper state, end-effector pose)
- System control functions (torque control, zero calibration)
"""

from alicia_d_sdk.api.synria_robot_api import SynriaRobotAPI

__all__ = [
    "SynriaRobotAPI"
]