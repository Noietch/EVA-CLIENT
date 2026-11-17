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

The fake launcher uses the existing Python 3.11 environment at
`examples/hardware/agilex_piper/.venv`, containing the dependencies declared in
this directory's `pyproject.toml`. Override its location with
`AGILEX_PIPER_VENV_DIR`. Activating the root `.venv` does not select the fake
node's interpreter. Run EVA itself with the root `.venv/bin/eva`.

The legacy `setup_env.sh` runs dependency synchronization. Preserve existing
environments; add only missing dependencies after checking that the installation
plan changes no installed package or file.

## Run the Fake Node (no hardware)

For development without arms or ROS, the fake node speaks ZMQ. Point EVA at a
`transport.type='zmq'` config. In DEVICE select `Dual AgileX Piper`, switch
Robot to `Fake`, choose Camera `None` if images are unnecessary, and apply. The
saved Fake mode changes a live workflow's runtime transport to ZMQ. Dataset
transport presets remain offline and do not connect to this node:

```bash
bash examples/hardware/agilex_piper/run_fake_node.sh
# then, in another shell, using the saved Fake device selection:
.venv/bin/eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py --web-port 8080
```

The deployment preset requires its policy server for inference. The fake node
supplies synthetic robot observations, not a policy.

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
.venv/bin/eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py --web-port 8080
```

When the EVA DEVICE page has `agilex_piper` selected in `Real` mode, its Robot
Start button launches this script with `--skip-camera`; the standalone command
above also launches the upstream cameras. Configure external cameras separately
when using DEVICE. CAN setup follows the YAM launcher: it
first tries non-interactive `sudo` (for an existing authorization), then falls
back to `pkexec` so the desktop PolicyKit agent can show the system password
dialog. The EVA page does not collect or store a separate Piper password.
For a local ROS1 config, EVA starts a lightweight `roscore` automatically when
the ROS master is absent; the Piper hardware and cameras still start only after
Robot Start is pressed in DEVICE.

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
PIPER_PYTHON_ENV     Python environment containing `piper_sdk` (default: `/home/agilex/miniconda3/envs/aloha`)
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
from the ROS topic mappings declared in this directory's `config.yaml`.

## Teleop Collection

Teleop is the upstream Agilex master/puppet (leader/follower) scheme over ROS 1,
not a separate EVA process. Select `Piper Leader` under Operation. The operator
moves the master arms on `/master/joint_{left,right}`; while Collection control is
enabled, EVA relays those messages to `/puppet/master_joint_{left,right}` and
records the raw master positions as the action. Disabling Collection control stops
the relay. There is nothing extra to launch beyond `run_agilex_ros.sh`.

EVA already subscribes to `/master/joint_left` and `/master/joint_right` as the
gated leader input. Do not run a separate unconditional master-to-follower relay:
EVA forwards those messages to `/puppet/master_joint_left` and
`/puppet/master_joint_right` only while Collection control or HIL takeover is
active.

`examples/hardware/agilex_piper/config.yaml` maps the topics per arm:

```text
left_arm   qpos        /puppet/joint_left          (follower state)
           action_qpos /master/joint_left          (leader state/action)
           command     /puppet/master_joint_left   (follower command)
           eef         /puppet/end_pose_left
right_arm  qpos        /puppet/joint_right
           action_qpos /master/joint_right
           command     /puppet/master_joint_right
           eef         /puppet/end_pose_right
```

Then run collection from the EVA host:

```bash
.venv/bin/eva --config configs/02_collection/dual_agilex_piper.py --web-port 8080
```
