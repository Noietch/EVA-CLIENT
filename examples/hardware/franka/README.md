# Franka (Dual) Hardware

This directory contains the hardware-side process for the dual Franka deploy
config (`configs/01_deploy/dual_franka/openpi_qpos.py`). This config uses
`transport.type: zmq`, so EVA talks to this node over ZMQ.

The real node connects to a dual Franka pair through `franky`, reads one or more
Orbbec cameras through the Orbbec Python SDK, publishes EVA ZMQ `WireObservation`
frames, and consumes EVA `WireAction` commands.

## Files

- `node.py`: real Franka ZMQ execution node — action/observation loop.
- `robot.py`: dual Franka hardware adapter backed by `franky`.
- `camera.py`: Orbbec SDK camera source.
- `run_hardware.sh`: launcher for the real node (IPs, disabled arms, env).
- `fake_node.py` / `run_fake_node.sh`: software-only fake node over ZMQ
  (built on `examples/hardware/fake_common.py`); needs no SDK and no real arms.

## Requirements

Run from the EVA client virtual environment:

```bash
source .venv/bin/activate
uv pip install -e ".[franka]"
```

The base EVA package provides the common runtime dependencies used by this node:

```text
numpy
opencv-python
pyyaml
pyzmq
msgpack / msgpack-numpy
```

The `franka` extra adds the real robot control dependency:

```text
franky-control==1.1.1
```

`pyorbbecsdk` is not declared in `pyproject.toml`. It is expected to already be
available on the real Franka robot host that has the Orbbec SDK installed. The
node imports `pyorbbecsdk` lazily, so deployments with all Orbbec cameras
disabled can still start without that package.

```bash
bash examples/hardware/franka/run_hardware.sh \
  --disabled-camera cam_high \
  --disabled-camera cam_left_wrist \
  --disabled-camera cam_right_wrist
```

## Run the Fake Node (no hardware)

For development without arms or the SDK, the fake node speaks the same ZMQ
protocol:

```bash
bash examples/hardware/franka/run_fake_node.sh
# then, in another shell:
eva --config configs/01_deploy/dual_franka/openpi_qpos.py --web-port 8080
```

Endpoint / rate overrides:

```bash
OBS_ENDPOINT=tcp://127.0.0.1:5555 \
ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
PUBLISH_RATE=30 \
  bash examples/hardware/franka/run_fake_node.sh
```

Extra arguments are forwarded to `fake_node.py` (e.g. `--image-height`,
`--image-width`). The fake node builds the `dual_franka` robot from its zoo
config, so its state/camera layout matches the real one.

## Run on Real Hardware

Run it on the robot host before starting EVA:

```bash
bash examples/hardware/franka/run_hardware.sh
eva --config configs/01_deploy/dual_franka/openpi_qpos.py --web-port 8080
```

By default `run_hardware.sh` reads `configs/01_deploy/dual_franka/openpi_qpos.py`
and starts the known Franka workcell Orbbec cameras that are not listed in
`transport.disabled_cameras`:

```text
cam_high        <- Orbbec Femto Bolt CL8R353009V
cam_right_wrist <- Orbbec Gemini 335 CP053530008B
cam_left_wrist  <- optional, no default serial
```

Use `EVA_CONFIG=/path/to/config.yaml` if the EVA config lives elsewhere.

Default arm IPs:

- `left_arm`: `172.16.0.2`
- `right_arm`: `172.16.0.3`

Override them with environment variables:

```bash
LEFT_ROBOT_IP=172.16.0.12 RIGHT_ROBOT_IP=172.16.0.13 \
  bash examples/hardware/franka/run_hardware.sh
```

If one arm is intentionally absent or powered off, disable it explicitly so the
node does not try to connect or command it:

```bash
DISABLED_ARMS=left_arm bash examples/hardware/franka/run_hardware.sh

# Equivalent CLI form:
bash examples/hardware/franka/run_hardware.sh --disabled-arm left_arm
```

The disabled arm's state stays at the cached initial qpos, while the available
arm continues to read state and execute actions. For EVA-side observations you
can also add the same arm under `transport.disabled_groups` in the EVA config:

```yaml
transport:
  disabled_groups:
    - left_arm
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

## Action / State Layout

The 16D action/state layout is:

```text
[left_arm 8D: panda_joint1..7 + gripper_scalar,
 right_arm 8D: panda_joint1..7 + gripper_scalar]
```

## Cameras

The Orbbec camera selector can be either a serial number or `index:N`.
CLI camera specs override defaults or add optional cameras. Resolution and FPS
are shared across configured cameras:

```bash
bash examples/hardware/franka/run_hardware.sh \
  --orbbec-camera cam_left_wrist=<LEFT_WRIST_SERIAL> \
  --orbbec-resolution 1280x720 \
  --orbbec-fps 30
```

If a camera is not connected, add its key to the EVA config so both EVA and the
hardware node skip it:

```yaml
transport:
  disabled_cameras:
    - cam_left_wrist
    - cam_high
```

## Teleop Collection

Not yet implemented (还在整理). `node.py` currently handles only deploy —
publishing observations and executing actions — and does not respond to EVA's
collection start/stop control actions or read any leader device.
`configs/02_collection/dual_franka.py` defines the recording schema (cameras,
columns, `action.qpos` / `action.eef`), but the teleop source that would
populate those action fields is not wired in this directory yet.

## Status

The node prints hardware status periodically:

```text
Hardware status: arms=[left_arm=online right_arm=disabled] cameras=[cam_high=online(age=0.0s)] rates=[obs=30.0Hz actions=2.0Hz]
```

Use `--status-log-interval 10` to change the interval, or set it to `0` to
disable periodic status logs.
