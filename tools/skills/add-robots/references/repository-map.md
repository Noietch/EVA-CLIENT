# EVA-CLIENT Robot Integration Map

Choose the closest current adapter instead of inventing a new structure.

| Template | Pattern | Useful paths |
| --- | --- | --- |
| YAM | ZMQ, external vendor SDK, mixed camera SDKs | `examples/hardware/yam`, `src/robots/zoo/dual_yam`, `configs/01_deploy/dual_yam` |
| ARX X5 | ZMQ, Python 3.12, vendored X5 SDK | `examples/hardware/arx_x5`, `src/robots/zoo/arx_x5`, `configs/02_collection/arx_x5_vr.py` |
| ARX R5 | ZMQ, Python 3.11, vendored ARX R5 and Alicia-D SDKs | `examples/hardware/arx_r5`, `src/robots/zoo/arx_r5` |
| AgiBot G2 | ZMQ, external GDK or mock | `examples/hardware/agibot_g2`, `src/robots/zoo/agibot_g2` |
| AgileX Piper | Direct ROS 1 integration | `examples/hardware/agilex_piper`, `src/robots/zoo/agilex_piper` |
| Franka | ZMQ, external Franka and camera SDKs | `examples/hardware/franka`, `src/robots/zoo/dual_franka` |
| UR5e | ZMQ, local gripper and teleoperation adapters | `examples/hardware/ur5e`, `src/robots/zoo/ur5e` |

## Required Touchpoints

- Registration: `src/robots/__init__.py` and `src/robots/zoo/<slug>/__init__.py`.
- Visual assets: `src/robots/zoo/<slug>/assets/` for URDF and meshes.
- Isolated runtime: `examples/hardware/<slug>/pyproject.toml`, `setup_env.sh`, launchers, node, robot, camera, SDK wrapper, and fake node as applicable.
- Configs: `configs/01_deploy`, `configs/02_collection`, and only the supported evaluation, open-loop, or RL groups.
- Shared fake/HIL behavior: `examples/hardware/fake_common.py` when the new adapter needs it.
- Tests: startup, fake node, adapter, wire format, config, visualization, and `tests/hardware/test_hardware_environments.py`.

Each top-level hardware adapter owns its Python environment. Launchers must resolve their sibling `.venv` and must not activate the repository root `.venv` or install a root hardware extra.
