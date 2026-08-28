# Supported Hardware Control Matrix

This table records the current operator input contract for `examples/hardware`.
The left/right controller mappings below are explicit, not inferred. Any live
motion, gripper action, or camera action still needs operator confirmation in
the corresponding launcher or UI.

| Hardware | Primary control contract | Left/right mapping | Current config entrypoints | Notes |
| --- | --- | --- | --- | --- |
| AgiBot G2 | WebXR / handheld controller | `left_arm=left`, `right_arm=right` | `configs/02_collection/agibot_g2_vr.py`, `configs/04_rl/agibot_g2_rl.py` | WebXR is the primary collection and RL input path. |
| ARX X5 | WebXR / handheld controller | `left_arm=left`, `right_arm=right` | `configs/02_collection/arx_x5_vr.py`, `configs/04_rl/arx_x5_rl.py` | VR teleop; no physical leader adapter is wired in the hardware node. |
| Dual Franka | WebXR / handheld controller | `left_arm=left`, `right_arm=right` | `configs/02_collection/dual_franka_vr.py`, `configs/04_rl/dual_franka_rl.py` | WebXR is the primary collection and RL input path. |
| UR5e | Alicia-D master arm + analog gripper trigger | single arm: `arm=arm` | `configs/02_collection/ur5e.py`, `configs/04_rl/ur5e_rl.py` | Primary leader path; the gripper stays on the analog trigger. |
| AgileX Piper | ROS 1 master/puppet transport | `left_arm` and `right_arm` each map to their own master/puppet topic pair | `configs/02_collection/dual_agilex_piper.py`, `configs/04_rl/dual_agilex_piper_rl.py` | Leader-follower stays upstream ROS 1 transport, not an EVA teleop client. |
| ARX R5 | Dual Alicia-D leaders | `left_arm=left`, `right_arm=right` | `configs/02_collection/arx_r5.py`, `configs/04_rl/arx_r5_rl.py` | Left and right leaders are separate Alicia-D ports and channels. |
| R1 Lite | CAT/GELLO leaders + JoyCon buttons | `left_arm=left`, `right_arm=right` | `configs/02_collection/r1lite.py`, `configs/04_rl/r1lite_rl.py` | JoyCon buttons gate rollout and intervention; gripper buttons stay side-specific. |
| YAM | Dual CAN teaching handles | `left_arm=left`, `right_arm=right` | `configs/02_collection/dual_yam.py`, `configs/01_deploy/dual_yam/openpi_qpos.py` | No `configs/04_rl` entry in this repo; collection and HIL stay on the leader-follower transport. |

WebXR input currently applies only to AgiBot G2, ARX X5, and Dual Franka. Every
other listed adapter stays on physical leader-follower or ROS transport input.
