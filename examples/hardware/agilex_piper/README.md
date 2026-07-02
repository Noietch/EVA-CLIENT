# Agilex Piper (Dual) Hardware

This directory contains the launch helpers for the dual Agilex Piper arms used
by the deploy configs (`configs/01_deploy/dual_agilex_piper/*.py`) and the
collection config (`configs/02_collection/dual_agilex_piper.py`).

Unlike the ZMQ robots (ARX / Franka / AgiBot / UR5e), Piper deploy and collection
run over **ROS 1 (Noetic)**: EVA's `ros1` transport connects directly to the
Piper ROS topics, so there is no `node.py` in this directory. The scripts here
bring up the upstream Piper ROS workspace (and its cameras); a ZMQ fake node is
provided for no-hardware dev.

## Files

- `run_agilex_ros.sh`: brings up the real Piper ROS 1 stack — `roscore`, the
  Astra camera launch, CAN configuration, and the `piper start_ms_piper.launch`
  node. Run this on the Piper host before starting EVA.
- `run_agilex_visualize.sh`: launches the `aloha_des` RViz visualization, showing
  the live arm state and an action preview overlay.
- `fake_node.py` / `run_fake_node.sh`: software-only fake node over ZMQ
  (built on `examples/hardware/fake_common.py`); needs no ROS and no real arms.

## Requirements

`run_agilex_ros.sh` and `run_agilex_visualize.sh` assume the upstream Agilex
`cobot_magic` workspaces and ROS Noetic are installed on the robot host. Each
path can be overridden by the matching environment variable:

```text
/opt/ros/noetic/setup.bash
/home/agilex/cobot_magic/Piper_ros_private-ros-noetic   # PIPER_ROOT
/home/agilex/cobot_magic/camera_ws                      # CAMERA_ROOT
/home/agilex/cobot_magic/aloha_des                      # ALOHA_DES_ROOT (visualize)
```

The fake node only needs the EVA base package in the virtual environment:

```bash
source .venv/bin/activate
```

## Run the Fake Node (no hardware)

For development without arms or ROS, the fake node speaks ZMQ. Point EVA at a
`transport.type='zmq'` config (the openloop config
`configs/00_openloop/dual_agilex_piper_openloop.py` is ROS-free):

```bash
bash examples/hardware/agilex_piper/run_fake_node.sh
# then, in another shell, with a zmq-transport config:
eva --config configs/00_openloop/dual_agilex_piper_openloop.py --web-port 8080
```

Endpoint / rate overrides:

```bash
OBS_ENDPOINT=tcp://127.0.0.1:5555 \
ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
PUBLISH_RATE=30 \
  bash examples/hardware/agilex_piper/run_fake_node.sh
```

Extra arguments are forwarded to `fake_node.py` (e.g. `--image-height`,
`--image-width`). The fake node builds the `agilex_piper` robot from its zoo
config, so its state/camera layout matches the real one.

## Run on Real Hardware

```bash
bash examples/hardware/agilex_piper/run_agilex_ros.sh
# then, on the EVA host (ROS 1 transport):
eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py --web-port 8080
```

`run_agilex_ros.sh` starts `roscore` (unless `--no-roscore`), launches the camera
stack and waits for the three color topics (`/camera_f|l|r/color/image_raw`),
runs `can_config.sh` (unless `--skip-can`), then launches the Piper node. On exit
it kills the camera and roscore processes it started.

Options:

```text
--mode <0|1>          puppet_mode passed to the piper launch. Default: 1
--auto-enable <bool>  auto_enable passed to roslaunch. Default: true
--skip-can            Skip running can_config.sh
--skip-camera         Do not launch the camera workspace
--build               Run catkin_make before launch
--no-roscore          Assume roscore is already running
-h, --help            Show help
```

Environment overrides:

```text
PIPER_ROOT           Piper workspace path
CAMERA_ROOT          Camera workspace path
CAMERA_LAUNCH_PKG    Camera launch package (default: astra_camera)
CAMERA_LAUNCH_FILE   Camera launch file   (default: multi_camera.launch)
```

Example — reuse an existing roscore, skip CAN setup, build first:

```bash
bash examples/hardware/agilex_piper/run_agilex_ros.sh \
  --no-roscore --skip-can --build --mode 1 --auto-enable true
```

### Visualize in RViz

```bash
bash examples/hardware/agilex_piper/run_agilex_visualize.sh
```

Options:

```text
--build       Run catkin_make in aloha_des before launch
-h, --help    Show help
```

Any extra `name:=value` arguments are forwarded to `roslaunch`. Topic overrides
(real state vs. action-preview overlay) come from environment variables:

```text
MASTER_LEFT_TOPIC  / MASTER_RIGHT_TOPIC    real arm-state topics  (default /puppet/joint_{left,right})
PUPPET_LEFT_TOPIC  / PUPPET_RIGHT_TOPIC     action preview topics  (default /sim_joint_{left,right})
ACTION_LEFT_TOPIC  / ACTION_RIGHT_TOPIC     backward-compatible aliases of the PUPPET_* vars
ALOHA_DES_ROOT     RViz workspace path
RVIZ_LAUNCH_PKG    launch package (default aloha_new_description)
RVIZ_LAUNCH_FILE   launch file    (default display_aloha_live.launch)
```

## Action / State Layout

The 14D action/state layout is:

```text
[left_arm  7D: joint1..6 + gripper_scalar,
 right_arm 7D: joint1..6 + gripper_scalar]
```

## Cameras

Collection records cameras `cam_high` / `cam_left_wrist` / `cam_right_wrist`
from the master/puppet ROS topics declared in
`configs/02_collection/dual_agilex_piper.py`.

## Teleop Collection

Teleop is the upstream Agilex master/puppet (leader/follower) scheme over ROS 1,
not an EVA process: the operator moves the master arms and EVA's `ros1` collection
transport records both the puppet state and the master command directly from the
ROS topics. There is nothing extra to launch here beyond `run_agilex_ros.sh`.

`configs/02_collection/dual_agilex_piper.py` maps the topics per arm:

```text
left_arm   qpos        /puppet/joint_left          (follower state)
           action_qpos /puppet/master_joint_left   (leader command)
           eef         /puppet/end_pose_left
right_arm  qpos        /puppet/joint_right
           action_qpos /puppet/master_joint_right
           eef         /puppet/end_pose_right
```

Then run collection from the EVA host:

```bash
eva --config configs/02_collection/dual_agilex_piper.py --web-port 8080
```
