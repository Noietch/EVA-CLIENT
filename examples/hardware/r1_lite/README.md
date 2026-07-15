# R1 Lite Hardware

This directory keeps the R1 Lite hardware entrypoints that are specific to this
robot. EVA core owns the generic ROS2 transport, HIL, collection recording, and
fixed-clock dataset save logic; robot startup details stay here.

## Files

- `run_hardware.sh`: starts the remote R1 Lite robot stack and CAT teleop stack.
- `kill_hardware.sh`: runs the remote cleanup commands on the R1 Lite and CAT hosts.
- `run_fake_node.sh`: starts the local ROS2 fake camera and robot processes.
- `fake_node.py`: ROS2-only R1 Lite fake publishers/subscribers and control UI.
- `cat_operator_button.py`: CAT-side ROS2 bridge from Joy-Con buttons to
  `/eva/operator_button`.
- `reset_torso.py`: publishes the fixed R1 Lite torso pose.

## Real Hardware

The scripts assume passwordless SSH from the EVA machine to:

```text
r1lite@192.168.12.12
cat@192.168.12.13
```

Start the robot and CAT teleop stacks from the repo root:

```bash
source .venv/bin/activate
bash examples/hardware/r1_lite/run_hardware.sh
```

By default this starts:

```text
R1 Lite host: tmux session eva_r1lite_robot
  /tmp/eva_r1lite_body_no_teleop_bootstrap
  copy R1LITEBody.d hdas/mobiman/ros2_discovery/system/tools yaml to
    /tmp/eva_r1lite_body_no_teleop.d
  tmuxp load those yaml files without R1LITEBody.d/teleop.yaml

CAT host: tmux session eva_r1lite_teleop
  source /opt/ros/humble/setup.bash
  source /home/cat/galaxea/install/setup.bash
  python3 /home/cat/galaxea/install/mobiman/lib/DynamixelTeleop/GelloTeleop/gello_teleop.py
    /tabletop/hdas/feedback_arm_left:=/eva/hil/input_joint_state_arm_left
    /tabletop/hdas/feedback_arm_right:=/eva/hil/input_joint_state_arm_right
  ros2 launch joycon_evdev_publisher joy_evdev_launch.py
  python3 /tmp/eva_cat_operator_button.py --topic /eva/operator_button
```

`run_hardware.sh` writes `/tmp/eva_r1lite_body_no_teleop_bootstrap` on the R1
Lite host. That bootstrap copies the vendor body session yaml files into
`/tmp/eva_r1lite_body_no_teleop.d` and starts only `hdas`, `mobiman`,
`ros_discovery`, `system`, and `tools`; it does not start vendor
`R1LITEBody.d/teleop.yaml`, so robot-side CAT teleop cannot publish directly to
the real motion target topics.

The script also copies `cat_operator_button.py` to
`/tmp/eva_cat_operator_button.py` on the CAT host, writes
`/tmp/eva_cat_fastdds_super_client.xml` and
`/tmp/eva_cat_hil_teleop_bootstrap`, and then starts the CAT tmux session. It
does not patch vendor files. The generated CAT FastDDS profile uses
`SUPER_CLIENT` discovery through `192.168.12.12:11811`, matching the EVA and
R1 Lite ROS2 graph. CAT arm motion is published only to EVA HIL input topics;
EVA forwards HIL to `/motion_target/target_joint_state_arm_left` and
`/motion_target/target_joint_state_arm_right` only while COLLECT ARM is on or
while rollout intervention is active with HIL ON.

Useful overrides:

```bash
START_R1LITE=0 bash examples/hardware/r1_lite/run_hardware.sh
START_CAT=0 bash examples/hardware/r1_lite/run_hardware.sh
R1LITE_HOST=r1lite@192.168.12.12 CAT_HOST=cat@192.168.12.13 \
  bash examples/hardware/r1_lite/run_hardware.sh
CAT_OPERATOR_BUTTON=0 bash examples/hardware/r1_lite/run_hardware.sh
```

Command overrides:

```bash
R1LITE_BODY_SESSION_REMOTE='/tmp/eva_r1lite_body_no_teleop.d' \
CAT_SETUP_CMD='source /opt/ros/humble/setup.bash && source /home/cat/galaxea/install/setup.bash' \
CAT_FASTRTPS_PROFILE='/tmp/eva_cat_fastdds_super_client.xml' \
CAT_HIL_TELEOP_REMOTE='/tmp/eva_cat_hil_teleop_bootstrap' \
  bash examples/hardware/r1_lite/run_hardware.sh
```

### EVA FastDDS Profile

The repository provides a sanitized workstation profile template at
`examples/hardware/r1_lite/fastdds/fastdds_super_client.example.xml`. Copy it to
the local ROS configuration directory:

```bash
mkdir -p "$HOME/.ros"
cp examples/hardware/r1_lite/fastdds/fastdds_super_client.example.xml \
  "$HOME/.ros/fastdds_r1lite_super_client.xml"
```

Replace every placeholder with values from the R1 Lite ROS2 network:

```text
__EVA_LOCAL_IP__                  EVA workstation address on the robot LAN
__DISCOVERY_SERVER_GUID_PREFIX__  FastDDS discovery server GUID prefix
__DISCOVERY_SERVER_IP__           FastDDS discovery server address
__DISCOVERY_SERVER_PORT__         FastDDS discovery server port
```

The GUID prefix must match the running discovery server; it is not derived from
the server IP. Verify that no placeholders remain:

```bash
profile="$HOME/.ros/fastdds_r1lite_super_client.xml"
if rg -n '__[A-Z0-9_]+__' "$profile"; then
  echo "FastDDS profile still contains placeholders" >&2
  exit 1
fi
export FASTRTPS_DEFAULT_PROFILES_FILE="$profile"
```

After hardware is up, start EVA locally in another shell:

```bash
source .venv/bin/activate
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/.ros/fastdds_r1lite_super_client.xml"
eva --config configs/02_collection/r1lite.py --web-port 8080
```

### FastDDS Camera Receive Buffer

Three 60 Hz compressed camera topics can overflow Linux's default 212,992-byte
UDP receive buffer. A dropped UDP fragment discards the complete best-effort ROS2
image sample and appears later as `image_skew_exceeded`, even when the short live
rate display has returned to 60 Hz.

Configure the EVA workstation's persistent receive-buffer limit in
`/etc/sysctl.d/90-eva-fastdds-image-buffer.conf`:

```text
net.core.rmem_default=16777216
net.core.rmem_max=16777216
```

Apply it before starting EVA:

```bash
sudo sysctl -p /etc/sysctl.d/90-eva-fastdds-image-buffer.conf
```

The UDPv4 `transport_descriptor` in the EVA workstation profile selected by
`FASTRTPS_DEFAULT_PROFILES_FILE` must also request that buffer:

```xml
<transport_descriptor>
  <transport_id>udp_transport</transport_id>
  <type>UDPv4</type>
  <receiveBufferSize>16777216</receiveBufferSize>
  <!-- Keep the site-specific interfaceWhiteList and discovery settings. -->
</transport_descriptor>
```

Both settings are required. Restart EVA after changing them, then inspect its
FastDDS user-data UDP socket:

```bash
ss -u -a -m -p | grep -A1 eva
```

The socket `rb` value must be larger than `212992`, and its `d` drop counter must
not increase during collection. Do not compensate for an increasing counter by
widening the collection image-skew tolerance.

For policy rollout:

```bash
source .venv/bin/activate
eva --config configs/01_deploy/r1lite/openpi_qpos.py --web-port 8080
```

## Operator Buttons

`cat_operator_button.py` subscribes to CAT Joy topics and publishes string events
to `/eva/operator_button`:

```text
right Joy button 2 -> x
right Joy button 3 -> y
left Joy button 4  -> left_gripper_open
left Joy button 5  -> left_gripper_close
right Joy button 4 -> right_gripper_open
right Joy button 5 -> right_gripper_close
```

EVA maps `x` / `y` according to the current state: collect start/stop/cancel,
rollout intervention continue/abandon, or rollout save. Gripper labels route to
the normal EVA gripper commands. During rollout, HIL is off by default; turn HIL
ON before pressing CAT X to enter intervention.

## Cleanup

Stop both remote stacks:

```bash
bash examples/hardware/r1_lite/kill_hardware.sh
```

Defaults:

```text
R1 Lite host: bash ~/workspace/robot/kill_robot.sh
CAT host:     bash ~/kill_tele.sh
```

Useful overrides:

```bash
START_R1LITE=0 bash examples/hardware/r1_lite/kill_hardware.sh
START_CAT=0 bash examples/hardware/r1_lite/kill_hardware.sh
R1LITE_KILL_CMD='bash ~/workspace/robot/kill_robot.sh' \
CAT_KILL_CMD='bash ~/kill_tele.sh' \
  bash examples/hardware/r1_lite/kill_hardware.sh
```

## ROS2 Fake Node

For local development without physical R1 Lite/CAT hardware, start the local ROS2
fake node with its embedded control UI:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
bash examples/hardware/r1_lite/run_fake_node.sh
```

`run_fake_node.sh` forces local ROS2 discovery by setting `ROS_LOCALHOST_ONLY=1`
and unsetting `FASTRTPS_DEFAULT_PROFILES_FILE`, `ROS_DISCOVERY_SERVER`, and
`RMW_IMPLEMENTATION`. This is intentional for fake/EVA local testing. Do not use
that local discovery setup for real R1 Lite hardware; real hardware runs should
keep the r1lite FastDDS SUPER_CLIENT profile that discovers
`192.168.12.12:11811`.

By default `run_fake_node.sh` starts two `fake_node.py` subprocesses:

```text
--role cameras: publishes the three compressed camera topics
--role robot:   publishes robot feedback, subscribes to EVA motion targets,
                serves the local UI, and publishes operator/HIL events
```

Splitting camera and robot roles keeps image publishing from starving control
callbacks during recording. The fake uses the same ROS2 topics as the R1 Lite
configs and exposes a local UI for X/Y, gripper, and HIL joint controls.

Start EVA against the fake in another shell with the same local ROS2 discovery
environment:

```bash
cd /home/ps/VLA-RL/EVA-CLIENT
source /opt/ros/humble/setup.bash
source .venv/bin/activate
unset FASTRTPS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER RMW_IMPLEMENTATION
export ROS_LOCALHOST_ONLY=1
eva --config /home/ps/VLA-RL/EVA-CLIENT/configs/02_collection/r1lite.py --web-port 8080
```

For policy rollout:

```bash
cd /home/ps/VLA-RL/EVA-CLIENT
source /opt/ros/humble/setup.bash
source .venv/bin/activate
unset FASTRTPS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER RMW_IMPLEMENTATION
export ROS_LOCALHOST_ONLY=1
eva --config /home/ps/VLA-RL/EVA-CLIENT/configs/01_deploy/r1lite/openpi_qpos.py --web-port 8080
```

UI, image, and rate overrides:

```bash
PUBLISH_RATE=30 UI_HOST=127.0.0.1 UI_PORT=8765 IMAGE_HEIGHT=360 IMAGE_WIDTH=640 \
  bash examples/hardware/r1_lite/run_fake_node.sh --no-open-ui
```

Extra arguments are forwarded to the `--role robot` `fake_node.py` process.
