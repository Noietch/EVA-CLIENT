# R1 Lite Hardware

This directory contains the launch helpers for the dual-arm R1 Lite, used by the
deploy configs (`configs/01_deploy/r1lite/*.py`) and the collection config
(`configs/02_collection/r1lite.py`).

Unlike the ZMQ robots (ARX / Franka / AgiBot / UR5e), R1 Lite runs over
**ROS 2 (Humble)**: EVA's `ros2` transport subscribes to the `/hdas/...` feedback
topics and publishes to the `/motion_target/...` command topics declared in
`configs/01_deploy/r1lite/_base.py`, so there is no `node.py` in this directory.
`run_hardware.sh` starts the robot's own bringup remotely over SSH and then
launches EVA locally; a ZMQ fake node is provided for no-hardware dev.

## Files

- `run_hardware.sh`: starts the on-robot bringup in a remote tmux session over
  SSH, sets up the local ROS 2 / FastDDS environment, and `exec`s `eva` with the
  R1 Lite config.
- `kill_hardware.sh`: kills the remote tmux sessions started by `run_hardware.sh`.
- `fake_node.py` / `run_fake_node.sh`: software-only fake node over ZMQ
  (built on `examples/hardware/fake_common.py`); needs no ROS and no real robot.

## Requirements

`run_hardware.sh` against the real robot assumes:

- ROS 2 Humble at `/opt/ros/humble/setup.bash` (sourced if present).
- A FastDDS super-client profile at `~/.ros/fastdds_r1lite_super_client.xml`
  (overridable via `FASTRTPS_DEFAULT_PROFILES_FILE`). The script **errors out**
  if this file is missing.
- Passwordless SSH access to the robot host (`r1lite` by default) where
  `~/r1lite/robot/start_robot.sh` brings up the robot, with `tmux` installed
  there.

The fake node only needs the EVA base package in the virtual environment:

```bash
source .venv/bin/activate
```

## Run the Fake Node (no hardware)

For development without the robot or ROS 2, the fake node speaks ZMQ. Point EVA
at a `transport.type='zmq'` config (the openloop config
`configs/00_openloop/r1lite_openloop.py` is ROS-free):

```bash
bash examples/hardware/r1_lite/run_fake_node.sh
# then, in another shell, with a zmq-transport config:
eva --config configs/00_openloop/r1lite_openloop.py --web-port 8080
```

Endpoint / rate overrides:

```bash
OBS_ENDPOINT=tcp://127.0.0.1:5555 \
ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
PUBLISH_RATE=30 \
  bash examples/hardware/r1_lite/run_fake_node.sh
```

Extra arguments are forwarded to `fake_node.py` (e.g. `--image-height`,
`--image-width`). The fake node builds the `r1_lite` robot from its zoo config,
so its state/camera layout matches the real one.

## Run on Real Hardware

```bash
bash examples/hardware/r1_lite/run_hardware.sh
```

This starts the remote robot bringup in tmux session `eva_r1lite_robot` on host
`r1lite` (idempotent — it skips if the session already exists), then launches
EVA. By default it uses `configs/02_collection/r1lite.py`. To use a different
config, pass `--config`; it is forwarded to `eva` verbatim along with any other
arguments:

```bash
bash examples/hardware/r1_lite/run_hardware.sh \
  --config configs/01_deploy/r1lite/openpi_qpos.py --web-port 8080
```

Environment overrides:

```text
R1LITE_HOST                       Robot SSH host                     (default: r1lite)
R1LITE_SESSION                    Remote tmux session for the robot  (default: eva_r1lite_robot)
CAT_HOST / CAT_SESSION            Teleop host / session (teleop line is commented out by default)
EVA_CONFIG                        Default config when no --config is passed (default: configs/02_collection/r1lite.py)
FASTRTPS_DEFAULT_PROFILES_FILE    FastDDS profile path
RMW_IMPLEMENTATION                RMW backend (default: rmw_fastrtps_cpp)
```

The teleop bringup line (host `cat`, session `eva_r1lite_teleop`) is commented
out in `run_hardware.sh`; uncomment it if you also need the leader/teleop stack.

### Stop the Robot

```bash
bash examples/hardware/r1_lite/kill_hardware.sh
```

This kills both the robot and teleop tmux sessions on their respective hosts.
Honors the same `R1LITE_HOST` / `R1LITE_SESSION` / `CAT_HOST` / `CAT_SESSION`
overrides.

## Action / State Layout

Dual arm on a fixed torso, 14D total:

```text
[left_arm  7D: left_arm_joint1..6 + left_gripper_joint,
 right_arm 7D: right_arm_joint1..6 + right_gripper_joint]
```

The gripper open position is `100.0`, and EEF poses default to `base_link`,
switchable to `torso_link3`.

## Cameras

Cameras `cam_high` / `cam_left_wrist` / `cam_right_wrist` map to the R1 Lite
head/wrist camera topics declared in `configs/01_deploy/r1lite/_base.py`.

## Teleop Collection

Teleop runs on the robot's own ROS 2 stack, not as an EVA process: a leader/teleop
bringup drives the arms and EVA's `ros2` collection transport records both the
feedback state and the motion-target command directly from the ROS topics.

`run_hardware.sh` already starts the robot bringup; the teleop bringup is the
second `start_remote_tmux` line (host `cat`, session `eva_r1lite_teleop`,
`/home/cat/start_tele.sh`), commented out by default — uncomment it to also start
the leader stack. `run_hardware.sh` defaults to `configs/02_collection/r1lite.py`,
which maps the topics per arm:

```text
left_arm   qpos        /hdas/feedback_arm_left                      (feedback state)
           action_qpos /motion_target/target_joint_state_arm_left  (commanded)
           eef         /relaxed_ik/motion_control/pose_ee_arm_left
right_arm  qpos        /hdas/feedback_arm_right
           action_qpos /motion_target/target_joint_state_arm_right
           eef         /relaxed_ik/motion_control/pose_ee_arm_right
```

`kill_hardware.sh` stops both the robot and teleop sessions.
