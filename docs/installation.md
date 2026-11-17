[← Back to README](../README.md)

# Installation

The EVA application requires **Python 3.11** (`>=3.11,<3.12`). Its environment
is separate from the hardware SDK environments under `examples/hardware/`.
The project's `uv` platform configuration targets Linux x86_64.

## First installation in a new checkout

Install [uv](https://github.com/astral-sh/uv) first, then run:

```bash
git clone https://github.com/Noietch/EVA-CLIENT.git
cd EVA-CLIENT
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

These commands assume `.venv` does not exist yet. For an existing checkout,
reuse its environment. Do not rebuild it or run synchronization, upgrade, or
reinstallation commands. Before adding a missing dependency, inspect the
installation plan and proceed only if existing packages and versions stay intact.
Run EVA and checks through `.venv/bin/` directly.

The only root-project extras currently declared in `pyproject.toml` are:

| Extra | Contents |
|-------|----------|
| `dev` | pytest, pytest-cov, Ruff, Pyright |
| `viz` | `fast-simplification` for mesh simplification |

Hardware SDKs, CAN drivers, ROS packages, and camera SDKs belong to their
respective hardware environments. See the [hardware guide](../examples/hardware/SUPPOTED_HARDWARE.md)
and each robot's README before configuring a launcher. The root project has no
`franka`, `arx`, `ur5e`, or `camera` extras.

## Transport requirements

| Transport | Requirements |
|-----------|--------------|
| `zmq` | A robot, fake-hardware, or simulator node speaking EVA's wire protocol; no ROS |
| `dataset` | A local compatible dataset; no ROS or hardware node |
| `ros1` | A compatible ROS 1 installation and access to its Python modules |
| `ros2` | A compatible ROS 2 installation and access to `rclpy` and message packages |

ROS requirements depend on the selected transport even when the robot SDK runs
in another process. Do not assume a system ROS Python installation is compatible
with Python 3.11; use the robot-specific setup instructions.

## Open the application

From the repository root:

```bash
.venv/bin/eva --language en
```

With no `--config`, EVA loads `configs/00_base/defaults.py` and opens DEVICE.
The default browser address is `http://127.0.0.1:8080`. This starts the application;
hardware readiness and policy inference require their own setup.

For an offline dataset workflow, use the bundled sample preset:

```bash
.venv/bin/eva --config configs/00_openloop/dual_agilex_piper_openloop.py
```

The preset points to `examples/agilex_dataset`. Inspect recorded data through
REPLAY; policy inference in DEBUG uses the configured policy backend. Continue
with [Quick start](quick-start.md) for devices, business presets, and headless control.
