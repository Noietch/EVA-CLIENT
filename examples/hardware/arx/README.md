# ARX R5 Hardware

This directory contains the hardware-side process for the ARX R5 deploy config
(`configs/01_deploy/arx_r5/openpi_qpos.py`) and the collection config
(`configs/02_collection/arx_r5.py`). These configs use `transport.type: zmq`, so
EVA talks to this node over ZMQ.

The real node drives a dual ARX R5 pair through the ARX R5 Python SDK
(`bimanual.SingleArm`, one instance per arm over CAN), reads one or more Orbbec
cameras through the Orbbec Python SDK, publishes EVA ZMQ `WireObservation`
frames, and consumes EVA `WireAction` commands.

## Files

- `node.py`: real ARX R5 ZMQ execution node — action/observation loop and teleop
  collection lifecycle.
- `robot.py`: dual `SingleArm` hardware adapter (CAN, joints, gripper).
- `camera.py`: Orbbec SDK camera source.
- `teleop.py`: Alicia-D leader-arm teleop helper for collection.
- `run_hardware.sh`: launcher for the real node (CAN setup, env, SDK paths).
- `fake_node.py` / `run_fake_node.sh`: software-only fake node over ZMQ
  (built on `examples/hardware/fake_common.py`); needs no SDK and no real arms.
- `SDK/`: vendored ARX R5 (`ARX_R5_python`) and Alicia-D (`alicia_d`) SDKs.
- `utils/`: Orbbec white-balance calibration helpers.

## Requirements

Run from the EVA client virtual environment:

```bash
source .venv/bin/activate
```

The base EVA package provides the common runtime dependencies used by this node:

```text
numpy
opencv-python
pyyaml
pyzmq
msgpack / msgpack-numpy
```

The ARX R5 Python SDK is vendored under `SDK/ARX_R5_python` (cloned from
`ARXroboticsX/R5`, `py/ARX_R5_python`). It ships a prebuilt pybind module
(`arx_r5_python`) plus shared libraries, so it must be built once on the robot
host before use:

```bash
cd examples/hardware/arx/SDK/ARX_R5_python
bash build.sh          # builds bimanual/build -> installs the pybind .so
source setup.sh        # exports LD_LIBRARY_PATH for the shared libs
```

`run_hardware.sh` adds `SDK/ARX_R5_python` to `PYTHONPATH` and the SDK `lib` dirs
to `LD_LIBRARY_PATH`, so `import bimanual` resolves at launch.

`pyorbbecsdk` is not declared in `pyproject.toml`. It is expected to already be
available on the real robot host that has the Orbbec SDK installed. The node
imports `pyorbbecsdk` lazily, so deployments with all Orbbec cameras disabled can
still start without that package.

```bash
bash examples/hardware/arx/run_hardware.sh \
  --disabled-camera cam_high \
  --disabled-camera cam_left_wrist \
  --disabled-camera cam_right_wrist
```

## Run the Fake Node (no hardware)

For development without arms or the SDK, the fake node speaks the same ZMQ
protocol:

```bash
bash examples/hardware/arx/run_fake_node.sh
# then, in another shell:
eva --config configs/01_deploy/arx_r5/openpi_qpos.py --web-port 8080
```

Endpoint / rate overrides:

```bash
OBS_ENDPOINT=tcp://127.0.0.1:5555 \
ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
PUBLISH_RATE=30 \
  bash examples/hardware/arx/run_fake_node.sh
```

Extra arguments are forwarded to `fake_node.py` (e.g. `--image-height`,
`--image-width`). The fake node builds the `arx_r5` robot from its zoo config, so
its state/camera layout matches the real one.

## Run on Real Hardware

Run it on the robot host before starting EVA:

```bash
bash examples/hardware/arx/run_hardware.sh
eva --config configs/01_deploy/arx_r5/openpi_qpos.py --web-port 8080
```

By default `run_hardware.sh` reads `configs/01_deploy/arx_r5/openpi_qpos.py`. Use
`EVA_CONFIG=/path/to/config.yaml` if the EVA config lives elsewhere. Open the web
console at `http://127.0.0.1:8080`.

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

These are ZMQ protocol ports, not HTTP ports. Opening `http://127.0.0.1:5556`
in a browser will show `ERR_INVALID_HTTP_RESPONSE`; use the EVA web console port
instead, usually `http://127.0.0.1:8080`.

## Action / State Layout

The 14D action/state layout is:

```text
[left_arm 7D: joint1..6 + gripper_scalar,
 right_arm 7D: joint1..6 + gripper_scalar]
```

The six arm joints are sent through `set_joint_positions` (radians); the gripper
scalar in `[0, 1]` is rescaled to the ARX catch position and sent through
`set_catch`. State is read back from `get_joint_positions` (six joints) and the
gripper position is normalized back into `[0, 1]`.

## Cameras

The Orbbec camera selector can be either a serial number or `index:N`. ARX
cameras have no default serials, so pass each one on the CLI. Resolution and FPS
are shared across configured cameras:

```bash
bash examples/hardware/arx/run_hardware.sh \
  --orbbec-camera cam_high=index:0 \
  --orbbec-camera cam_left_wrist=<LEFT_WRIST_SERIAL> \
  --orbbec-camera cam_right_wrist=<RIGHT_WRIST_SERIAL> \
  --orbbec-resolution 640x480 \
  --orbbec-fps 30 \
  --orbbec-color-format MJPG
```

On the current ARX Orbbec set, the SDK default color profiles are high
resolution but only about 10 Hz. The measured 30 Hz color profile shared by all
three cameras is `640x480 MJPG@30`.

To calibrate color white balance, point each camera at the same white/gray
reference under the task lighting and run the ARX calibration CLI. By default it
first asks which camera to calibrate: enter `1`, `2`, or `3` for one camera, or
press Enter to run all cameras. It pauses before each selected camera; with all
three cameras, press Enter three more times, once after aiming each camera. It
then lets Orbbec auto white balance settle and writes the manual values to YAML:

```bash
bash examples/hardware/arx/utils/calibrate_white_balance_x3.sh
```

You can also select one camera from the command line:

```bash
bash examples/hardware/arx/utils/calibrate_white_balance_x3.sh 2
```

Each run also saves one calibration frame per camera under
`work_dirs/arx_orbbec_white_balance/<timestamp>/`.

Then pass the generated file to the node. The node disables auto white balance
for cameras listed in the YAML and applies the saved manual value at startup:

```bash
bash examples/hardware/arx/run_hardware.sh \
  --orbbec-white-balance-file examples/hardware/arx/utils/white_balance.yaml
```

If a camera selector changes, override it with an environment variable:

```bash
CAM_HIGH_SELECTOR=index:0 \
CAM_LEFT_WRIST_SELECTOR=<LEFT_WRIST_SERIAL> \
CAM_RIGHT_WRIST_SELECTOR=<RIGHT_WRIST_SERIAL> \
  bash examples/hardware/arx/utils/calibrate_white_balance_x3.sh
```

To choose a different image output directory, set `ORBBEC_WB_IMAGE_DIR`.

A camera with no selector is skipped. To also tell EVA to skip a missing camera,
add its key under `transport.disabled_cameras` in the EVA config:

```yaml
transport:
  disabled_cameras:
    - cam_left_wrist
```

The valid camera keys are `cam_high`, `cam_left_wrist`, and `cam_right_wrist`.

## Arms and CAN

Each arm is one ARX `SingleArm` over a CAN interface. Default CAN ports:

- `left_arm`: `can0`
- `right_arm`: `can1`

`run_hardware.sh` initializes the default CANable devices before starting the node:

- `/dev/arxcan0` -> `can0` -> `left_arm`
- `/dev/arxcan1` -> `can1` -> `right_arm`

Override them with environment variables:

```bash
LEFT_CAN_DEVICE=/dev/arxcan0 LEFT_CAN_PORT=can0 \
RIGHT_CAN_DEVICE=/dev/arxcan1 RIGHT_CAN_PORT=can1 \
  bash examples/hardware/arx/run_hardware.sh
```

Set `INIT_ARX_CAN=0` to skip CAN setup when the interfaces are already managed
outside this script.

`ARM_TYPE` selects the ARX SDK URDF/type code (`0` = X5lite, `1` = R5 master);
`GRIPPER_OPEN_POS` is the ARX catch position that maps to a fully open gripper
(EVA gripper scalar `1.0`):

```bash
ARM_TYPE=0 GRIPPER_OPEN_POS=5.0 bash examples/hardware/arx/run_hardware.sh
```

`node.py` enables each connected arm from the long-lived `ArxDualArm` instance by
reading its current joint positions and writing them back through
`set_joint_positions`. `run_hardware.sh` also keeps an optional preflight enable path
for diagnosis; set `ENABLE_ARX_ARMS=1` to run it before launching the node.
The launcher filters the repeated SDK banner line `ARX方舟无限` by default; set
`ARX_FILTER_SDK_BANNER=0` to keep raw SDK stdout/stderr.

If one arm is intentionally absent or powered off, disable it explicitly so the
script does not initialize, enable, connect, or command it:

```bash
DISABLED_ARMS=left_arm bash examples/hardware/arx/run_hardware.sh

# Equivalent CLI form:
bash examples/hardware/arx/run_hardware.sh --disabled-arm left_arm
```

The disabled arm's state stays at the cached initial qpos, while the available
arm continues to read state and execute actions. For EVA-side observations you
can also add the same arm under `transport.disabled_groups` in the EVA config:

```yaml
transport:
  disabled_groups:
    - left_arm
```

## Teleop Collection

For `configs/02_collection/arx_r5.py`, `node.py` listens for EVA's
collection start/stop control actions. During an active collection episode it
reads the two Alicia-D leader arms, maps them to the ARX qpos action layout,
commands the ARX followers, and publishes the aligned collection fields:

- `observation.qpos`: ARX follower joint/gripper state.
- `observation.eef`: FK from the ARX follower qpos.
- `action`: Alicia leader command mapped into ARX joint/gripper space.
- `action_eef`: FK from that mapped Alicia action.

Default Alicia ports match `teleop.py`:

- `left_arm`: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C192742-if00`
- `right_arm`: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C192642-if00`

Override them on `run_hardware.sh` through normal node arguments:

```bash
bash examples/hardware/arx/run_hardware.sh \
  --left-alicia-port /dev/serial/by-id/<left> \
  --right-alicia-port /dev/serial/by-id/<right>
```

## Status

The node prints hardware status periodically:

```text
Hardware status:
  arms=[left_arm=online right_arm=disabled]
  cameras=[cam_high=online(age=0.0s)]
  rates=[obs=30.0Hz actions=2.0Hz]
```

Use `--status-log-interval 10` to change the interval, or set it to `0` to
disable periodic status logs.
