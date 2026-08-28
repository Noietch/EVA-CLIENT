# Alicia-D SDK


[English Version](README_EN.md) | [Chinese Version](README.md) | [Official Taobao Store](https://g84gtpygdv6trpvdhcsy0kfr73avcip.taobao.com/shop/view_shop.htm?appUid=RAzN8HWKU5B7MfX6JjEWgkuNfftNVbnrjbjx6fPjY9KqXB46Rvy&spm=a21n57.1.hoverItem.2) | [Alicia-D Product Manual](https://docs.sparklingrobo.com/)

<p align="center"><img src="./imgs/Alicia_D_v5_5.jpg" width="500" /></p>



**Alicia-D SDK** is a Python toolkit for controlling the six-axis arms in the
Lingdong Alicia-D series, including their grippers. It is built on the
`RoboCore` library and provides serial communication for arm motion and gripper
control, as well as pose and state queries.


## RoboCore: Unified High-Throughput Robotics Library 

![](./imgs/logo.jpeg)

This SDK is powered by [RoboCore (Unified High-Throughput Robotics Library)](https://github.com/Synria-Robotics/RoboCore), developed by [Synria Robotics Co., Ltd.](https://synriarobotics.ai).


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)


---

### ✨ Core Features

| Module | Function | Status |
|---|---|---|
| **Modeling** | URDF/MJCF parsing and robot-model abstraction | ✅ Stable |
| **Forward kinematics** | C++/NumPy/PyTorch backends and batching | ✅ Stable |
| **Inverse kinematics** | DLS/Pinv/Transpose solvers and multi-start solving | ✅ Stable |
| **Jacobian** | Analytic, numerical, and automatic-differentiation methods | ✅ Stable |
| **Coordinate transforms** | SE(3)/SO(3) rigid-body transforms and format conversion | ✅ Stable |
| **Kinematic analysis** | Workspace and singularity analysis | ✅ Beta |
| **Trajectory planning** | Trajectory generation | ✅ Beta |
| **Visualization** | Kinematic-chain visualization | ✅ Stable |
| **Configuration management** | YAML-based configuration management | ✅ Stable |






## Main Features

*   **Joint control**: Set and read the six joint angles with smooth interpolated execution.
*   **End-effector trajectories**: Plan and execute Cartesian end-effector pose trajectories.
*   **Gripper control**: Precise angle control or one-click open/close switching.
*   **Torque control**: Enable or disable joint motor torque for free dragging and teaching.
*   **Zero setting**: Save the current position as a new zero position.
*   **State reading**: Read joint angles, gripper angle, and end-effector pose in real time.
*   **Automatic serial connection**: Search for a serial port automatically or specify one manually.
*   **Drag teaching**: Record pose points by dragging and execute the resulting trajectory.
*   **Logging system**: Filter log levels and control console verbosity.
*   **RoboCore integration**: Use the RoboCore C++ backend by default while retaining explicit `numpy` / `torch` switching.

## Project Structure

```
├── alicia_d_sdk
│   ├── api
│   │   └── synria_robot_api.py      # User-facing API
│   ├── execution
│   │   └── hardware_executor.py     # Execution layer
│   ├── hardware
│   │   ├── serial_comm.py           # Serial communication
│   │   ├── data_parser.py           # Data parsing
│   │   └── servo_driver.py          # Servo data transmission
│   ├── __init__.py
│   └── utils
│       ├── calculate.py             # Control calculation functions
│       └── logger/                  # Logging system
├── docs
│   ├── api_reference.md             # API reference
│   ├── examples.md                  # Example guide
│   ├── installation.md              # Installation guide
│   └── logger_levels.md             # Log levels
├── examples
│   ├── 00_demo_read_version.py      # Read firmware version
│   ├── 01_torque_switch.py          # Torque switch
│   ├── 02_demo_set_new_zero.py      # Set a new zero
│   ├── 03_demo_read_state.py        # Read state
│   ├── 04_demo_move_gripper.py      # Gripper control
│   ├── 05_demo_move_joint.py        # Joint motion
│   ├── 06_demo_forward_kinematics.py  # Forward kinematics
│   ├── 07_demo_inverse_kinematics.py  # Inverse kinematics
│   ├── 08_demo_drag_teaching.py     # Drag teaching
│   ├── 09_demo_joint_traj.py        # Joint-space trajectory planning
│   ├── 10_demo_cartesian_traj.py    # Cartesian-space trajectory planning
│   ├── 11_benchmark_read_joints.py  # Joint-read performance benchmark
│   └── 12_utmostFPS.py              # Maximum frame-rate test
```

## Installation

```bash
pip install alicia_d_sdk
```

## Quick Start

1.  Install with `pip install alicia_d_sdk` or see the [Installation Guide](docs/installation.md).
2.  Run an example:
```bash
cd examples
python3 00_demo_read_version.py    # Read firmware version
python3 03_demo_read_state.py      # Read state
python3 04_demo_move_gripper.py    # Gripper control
python3 05_demo_move_joint.py      # Joint motion
```

## Documentation

**Chinese documentation:**
*   [Installation Guide](docs/installation.md)
*   [Examples Guide](docs/examples.md)
*   [API Reference](docs/api_reference.md)
*   [Logger Levels](docs/logger_levels.md)

**English Documentation:**
*   [Installation Guide](docs/installation_en.md)
*   [Examples Guide](docs/examples_en.md)
*   [API Reference](docs/api_reference_en.md)
*   [Logger Levels](docs/logger_levels_en.md)
