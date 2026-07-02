# UR5e Hardware

This directory contains the hardware-side process for the UR5e deploy config
(`configs/01_deploy/ur5e/openpi_qpos.py`). This config uses
`transport.type: zmq`, so EVA talks to this node over ZMQ; the device SDKs are
not transport backends.

The real node drives the UR5e through UR RTDE, controls a Dahuan AG95 gripper,
optionally runs Alicia-D teleop for collection, reads OpenCV cameras, publishes
EVA ZMQ `WireObservation` frames, and consumes EVA `WireAction` commands.

## Files

- `node.py`: real UR5e ZMQ execution node — action/observation loop and teleop
  collection lifecycle.
- `robot.py`: UR RTDE and AG95 hardware adapter.
- `camera.py`: OpenCV camera helpers.
- `teleop.py`: Alicia-D teleop helper.
- `gripper.py`: Dahuan AG95 smoke-test helper and normalized gripper wrapper.
- `run_hardware.sh`: launcher for the real node (robot IP, gripper/teleop ports,
  cameras, env).
- `fake_node.py` / `run_fake_node.sh`: software-only fake node over ZMQ
  (built on `examples/hardware/fake_common.py`); needs no SDK and no real robot.
- `tests/`: AG95 and Alicia-D smoke-test scripts.

## Requirements

Install the EVA base package and UR5e hardware SDK dependencies in the project
virtual environment:

```bash
source .venv/bin/activate
uv pip install -e ".[ur5e]"
```

The `ur5e` extra installs:

```text
alicia_d_sdk==6.1.0rc4
ur-rtde==1.6.3
pydhgripper==1.0.2
```

OpenCV camera access uses the base project `opencv-python` dependency. The robot
host must also have network access to the UR controller and permission to open
the AG95 and Alicia-D serial ports.

## Run the Fake Node (no hardware)

For development without the robot or the SDK, the fake node speaks the same ZMQ
protocol:

```bash
bash examples/hardware/ur5e/run_fake_node.sh
# then, in another shell:
eva --config configs/01_deploy/ur5e/openpi_qpos.py --web-port 8080
```

Endpoint / rate overrides:

```bash
OBS_ENDPOINT=tcp://127.0.0.1:5555 \
ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
PUBLISH_RATE=25 \
  bash examples/hardware/ur5e/run_fake_node.sh
```

Extra arguments are forwarded to `fake_node.py` (e.g. `--image-height`,
`--image-width`). The fake node builds the `ur5e` robot from its zoo config, so
its state/camera layout matches the real one.

## Run on Real Hardware

Run it on the robot host before starting EVA:

```bash
bash examples/hardware/ur5e/run_hardware.sh
eva --config configs/01_deploy/ur5e/openpi_qpos.py --web-port 8080
```

Environment overrides:

```bash
UR5E_ROBOT_IP=192.168.31.123 AG95_PORT=/dev/ttyUSB1 \
OBS_ENDPOINT=tcp://127.0.0.1:5555 ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
  bash examples/hardware/ur5e/run_hardware.sh
```

Enable Alicia-D teleop in the same UR5e node process:

```bash
USE_TELEOP=1 TELEOP_PORT=/dev/ttyACM0 \
  bash examples/hardware/ur5e/run_hardware.sh
```

Override the teleop joint signs when needed:

```bash
USE_TELEOP=1 TELEOP_JOINT_COEF=1,-1,-1,-1,-1,-1 \
  bash examples/hardware/ur5e/run_hardware.sh
```

## Endpoints

Default endpoints match the EVA preset:

```yaml
transport:
  type: zmq
  sub_endpoint: "tcp://127.0.0.1:5555"
  pub_endpoint: "tcp://127.0.0.1:5556"
```

The node binds both endpoints. EVA connects to them.

- `tcp://127.0.0.1:5555`: node publishes observations, EVA subscribes.
- `tcp://127.0.0.1:5556`: node subscribes to actions, EVA publishes.

## Cameras

Camera arguments use:

```text
--camera name:source:width:height:fps:rotation
```

`run_hardware.sh` defaults (overridable via `UR5E_WRIST_CAMERA` /
`UR5E_EXTERIOR_CAMERA`, or by passing `--camera` directly):

```text
wrist_image:2:1280:720:25:0
exterior_image:0:1280:720:25:0
```

## Smoke Tests

AG95 gripper:

```bash
bash examples/hardware/ur5e/tests/run_gripper_test.sh --command 1.0
bash examples/hardware/ur5e/tests/run_gripper_test.sh --command 0.0
```

Use `AG95_PORT=/dev/ttyUSB1` to override the gripper serial port.

Alicia-D teleop:

```bash
bash examples/hardware/ur5e/tests/run_teleop_test.sh
```

Use `TELEOP_PORT=/dev/ttyACM0` to override the teleop serial port.

## Teleop Collection

For `configs/02_collection/ur5e.py`, `node.py` runs an Alicia-D master arm
(`teleop.type='ur5e_master_arm'`) inside the same process. It listens for EVA's
collection start/stop control actions; on start it connects the leader and
calibrates a delta against the current UR5e qpos, and during an active episode it
reads the leader command and publishes it as `action.qpos` (with FK
`action.eef`) alongside the follower's `observation.state`.

Enable teleop in the node and override the port / joint signs as needed:

```bash
USE_TELEOP=1 TELEOP_PORT=/dev/ttyACM0 \
TELEOP_JOINT_COEF=1,-1,-1,-1,-1,-1 \
  bash examples/hardware/ur5e/run_hardware.sh
```

The gripper command source is the leader's analog trigger
(`teleop.gripper.source='analog'`).
