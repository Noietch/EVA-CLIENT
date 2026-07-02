# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Alicia-D SDK v6.1.0rc5 - Bridged with RoboCore

Architecture Layers:
- User Layer: SynriaRobotAPI (unified user interface)
- Execution Layer: TrajectoryExecutor, DragTeaching (trajectory execution)
- Hardware Layer: ServoDriver, SerialComm, DataParser (low-level hardware drivers)
- Kinematics Layer: RoboCore kinematics functions (FK/IK/Jacobian)
- Planning Layer: RoboCore trajectory planning functions

RoboCore Integration:
- Default compute backend: RoboCore cpp backend when available
- robocore.kinematics: Provides FK/IK/Jacobian calculations
- robocore.planning: Provides trajectory planning functionality
- robocore.modeling: Provides RobotModel for robot representation
"""

from alicia_d_sdk.api import SynriaRobotAPI
from alicia_d_sdk.hardware import ServoDriver
from alicia_d_sdk.api.synria_robot_api import BackendName

# Import from RoboCore for kinematics and modeling
from robocore.modeling import RobotModel
from robocore.kinematics import forward_kinematics, inverse_kinematics, jacobian
from synriard import get_model_path
import json
from pathlib import Path
from typing import Optional


__version__ = "6.1.0rc5"
__author__ = "Synria Robotics"
__description__ = "Alicia-D Robot Arm SDK v6.1.0rc5 - Bridged with RoboCore"


def _get_gripper_type_from_json() -> str:
    """Read gripper type from JSON file, return default if not found."""
    json_path = Path(__file__).parent / "api" / "gripper_type.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_type = data.get("type_name")
            if isinstance(cached_type, str) and cached_type:
                return cached_type
        except Exception:
            pass
    return "50mm"

# Re-export RoboCore components for convenience
__all__ = [
    # Core API
    "SynriaRobotAPI",
    "create_robot",

    # Hardware Layer
    "ServoDriver",

    # RoboCore - Modeling
    "RobotModel",

    # RoboCore - Kinematics
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",

]


def create_robot(
    port: str = "",
    version: str = "v5_6",
    variant: str = None,
    model_format: str = "urdf",
    debug_mode: bool = False,
    auto_connect: bool = True,
    base_link: str = "base_link",
    end_link: str = "tool0",
    backend: Optional[BackendName] = None,
    device: str = "cpu",
    model_path: str = None,
    gripper_type: str = None,
) -> SynriaRobotAPI:
    """
    Create robot instance.

    :param port: Serial port
    :param version: Version name, e.g., "v5_6", "v7_0", etc.
    :param variant: Variant name, e.g., "gripper_50mm", "gripper_100mm", "leader_ur", etc.
    :param model_format: Model format, 'urdf' or 'mjcf', default is 'urdf'
    :param debug_mode: Debug mode
    :param base_link: Base link name in the robot model (default 'base_link')
    :param end_link: End link name in the robot model (default 'tool0')
    :param backend: Computation backend, 'cpp', 'numpy', or 'torch' (default: None, uses 'cpp')
    :param device: Device for torch backend, 'cpu' or 'cuda' (default: 'cpu', ignored for 'cpp' and 'numpy')
    :param model_path: Model path, if None, use default model path
    :param gripper_type: Gripper type:  deprecated, use variant instead please
        - explicit value such as "50mm" / "100mm" for user-defined configuration
        - None to auto-select from saved JSON (if available) or default to "50mm"
    :return: SynriaRobotAPI instance
    """
    servo_driver = ServoDriver(port=port, debug_mode=debug_mode)

    effective_gripper_type = gripper_type if gripper_type is not None else _get_gripper_type_from_json()
    variant = variant if variant is not None else f"gripper_{effective_gripper_type}"

    if model_path is None:
        model_path = get_model_path(
            "Alicia_D",
            version=version,
            variant=variant,
            model_format=model_format
        )
    robot_model = RobotModel(str(model_path), base_link=base_link, end_link=end_link)

    robot = SynriaRobotAPI(
        servo_driver=servo_driver,
        robot_model=robot_model,
        auto_connect=auto_connect,
        backend=backend,
        device=device
    )

    return robot
