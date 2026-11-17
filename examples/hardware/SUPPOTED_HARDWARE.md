# Supported Hardware Control Matrix

This table describes the DEVICE catalog assembled from each hardware
`config.yaml` and `examples/input_sources/*/config.yaml`. Collection/deploy
configs choose the task; DEVICE chooses the robot, operation, camera combination,
and Real/Fake mode. Apply the selection before starting its components.

| Hardware | Default and available Operation | Controller/leader mapping | Current config entrypoints | Notes |
| --- | --- | --- | --- | --- |
| AgiBot G2 | VR (default), EVA-VR (PICO) | `left_arm=left`, `right_arm=right` | `configs/02_collection/agibot_g2.py`, `configs/04_rl/agibot_g2_rl.py` | Real GDK node starts separately; no real Robot launcher is declared. |
| ARX X5 | VR (default), EVA-VR (PICO) | `left_arm=left`, `right_arm=right` | `configs/02_collection/arx_x5.py`, `configs/02_collection/arx_x5_tasks_set.py`, `configs/04_rl/arx_x5_rl.py` | Dedicated Python 3.12 hardware environment. |
| Dual Franka | VR (default), EVA-VR (PICO) | `left_arm=left`, `right_arm=right` | `configs/02_collection/dual_franka.py`, `configs/04_rl/dual_franka_rl.py` | Real robot and optional Orbbec camera processes are separate. |
| UR5e | VR (default), EVA-VR (PICO), Leader | VR: `arm=right`; Leader: Alicia-D single arm | `configs/02_collection/ur5e.py`, `configs/04_rl/ur5e_rl.py` | Leader gripper uses its analog trigger. |
| AgileX Piper | Piper Leader (default) | each arm uses its own ROS 1 master/puppet topic pair | `configs/02_collection/dual_agilex_piper.py`, `configs/04_rl/dual_agilex_piper_rl.py` | No VR option; DEVICE Robot launch passes `--skip-camera`. |
| ARX R5 | VR (default), EVA-VR (PICO), Leader | VR: `left_arm=left`, `right_arm=right`; Leader: two Alicia-D ports | `configs/02_collection/arx_r5.py`, `configs/04_rl/arx_r5_rl.py` | DEVICE Robot launch uses passive collection; see the adapter README for active standalone collection. |
| R1 Lite | VR (default), EVA-VR (PICO) | `left_arm=left`, `right_arm=right` | `configs/02_collection/r1lite.py`, `configs/04_rl/r1lite_rl.py` | Real ROS2/CAT launch scripts are separate; CAT/GELLO is not a current DEVICE Operation option. |
| YAM | VR (default), EVA-VR (PICO), Leader | `left_arm=left`, `right_arm=right`; Leader: two CAN teaching handles | `configs/02_collection/dual_yam.py`, `configs/01_deploy/dual_yam/openpi_qpos.py` | Standalone shell launcher defaults to dual leaders; no YAM `configs/04_rl` entry. |

`EVA-VR (PICO)` uses the native APK and the same host bridge and arm mappings as
WebXR. Physical-leader options use transport control; VR options use client
retargeting. See the adapter README before launching real hardware.
