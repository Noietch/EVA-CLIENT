# AgiBot G2 Hardware

This directory contains the hardware-side process for the AgiBot G2 deploy
configs (`configs/01_deploy/agibot_g2/openpi_qpos.py`,
`configs/01_deploy/agibot_g2/openpi_eef.py`) and the collection config
(`configs/02_collection/agibot_g2.py`). These configs use `transport.type: zmq`,
so EVA talks to this node over ZMQ.

The real node drives the dual G2 arms, head, and body through the AgiBot GDK
(`agibot_gdk`), reads the head/left-hand/right-hand color cameras through the
same SDK, publishes EVA ZMQ `WireObservation` frames, and consumes EVA
`WireAction` commands.

## Files

- `node.py`: real GDK ZMQ execution node. A 30 Hz observation thread publishes
  state + camera frames; a 100 Hz servo loop drains incoming actions and sends
  `joint_servo_control` to the robot.
- `agibot_gdk_mock.py`: drop-in stand-in for the real `agibot_gdk` module. It
  replays the first frame of three local `episode_000000.mp4` files as camera
  images and echoes commanded joints back as state, to dry-run `node.py` without
  the GDK installed (see "Mocking the GDK" below).
- `fake_node.py` / `run_fake_node.sh`: software-only fake node over ZMQ
  (built on `examples/hardware/fake_common.py`); needs no GDK and no real
  hardware. This is the normal no-hardware dev path.

## Requirements

Run from the EVA client virtual environment:

```bash
source .venv/bin/activate
```

The real node imports `agibot_gdk`, which is **not** declared in `pyproject.toml`.
It is expected to already be installed on the AgiBot robot host. If it is missing,
`node.py` raises `ImportError("Please install agibot_gdk first.")` at startup.

The fake node imports no robot SDK and runs anywhere the EVA base package is
installed.

## Run the Fake Node (no hardware)

For development without the robot or the GDK, the fake node speaks the same ZMQ
protocol:

```bash
bash examples/hardware/agibot_g2/run_fake_node.sh
# then, in another shell:
eva --config configs/01_deploy/agibot_g2/openpi_qpos.py --web-port 8080
```

Endpoint / rate overrides:

```bash
OBS_ENDPOINT=tcp://127.0.0.1:5555 \
ACTION_ENDPOINT=tcp://127.0.0.1:5556 \
PUBLISH_RATE=30 \
  bash examples/hardware/agibot_g2/run_fake_node.sh
```

Extra arguments are forwarded to `fake_node.py` (e.g. `--image-height`,
`--image-width`). The fake node builds the `agibot_g2` robot from its zoo config,
so its state/camera layout matches the real one.

## Run on Real Hardware

Run it on the robot host before starting EVA:

```bash
python examples/hardware/agibot_g2/node.py \
  --obs-endpoint tcp://127.0.0.1:5555 \
  --action-endpoint tcp://127.0.0.1:5556
eva --config configs/01_deploy/agibot_g2/openpi_qpos.py --web-port 8080
```

`node.py` takes only the two endpoint arguments, both defaulting to the EVA
presets. Open the EVA web console (e.g. `http://127.0.0.1:8080`) in a browser,
not the ZMQ ports.

### Mocking the GDK

To exercise `node.py` itself (its threading, ZMQ wiring, and joint mapping)
without the real SDK, swap the import to the bundled mock at the top of `node.py`:

```python
# import agibot_gdk
import agibot_gdk_mock as agibot_gdk
```

The mock expects three video files relative to the working directory:

```text
observation.images.top_head/episode_000000.mp4
observation.images.hand_left/episode_000000.mp4
observation.images.hand_right/episode_000000.mp4
```

For a pure no-hardware run with synthetic data and no video files, prefer
`run_fake_node.sh` instead.

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

These are ZMQ protocol ports, not HTTP ports; use the EVA web console port
(usually `http://127.0.0.1:8080`) in a browser.

## Action / State Layout

`node.py` reads and commands 24 joints in this fixed order:

```text
[left_arm  8D: arm_l_joint1..7 + gripper_l_inner_joint1,
 right_arm 8D: arm_r_joint1..7 + gripper_r_inner_joint1,
 head      3D: head_joint1..3,
 body      5D: body_joint1..5]
```

State is published per group (`left_arm`, `right_arm`, `head`, `body`); actions
with `target == "real"` overwrite the servo target. The servo loop runs at
100 Hz and the observation loop at 30 Hz.

## Cameras

The real node maps three GDK color cameras to EVA observation keys:

```text
kHeadColor      -> top_head
kHandLeftColor  -> hand_left
kHandRightColor -> hand_right
```

The collection config (`configs/02_collection/agibot_g2.py`) records these under
`cam_high` / `cam_left_wrist` / `cam_right_wrist`.

## Teleop Collection

Not yet implemented (还在整理). `node.py` currently handles only deploy — its
30 Hz observation thread and 100 Hz servo loop — and does not respond to EVA's
collection start/stop control actions or read any leader device.
`configs/02_collection/agibot_g2.py` defines the recording schema (cameras,
columns, `action.qpos` / `action.eef`), but the teleop source that would
populate those action fields is not wired in this directory yet.

## Shutdown

`SIGINT` / `SIGTERM` (Ctrl-C) stops the observation thread, closes the camera,
releases the GDK, and closes both sockets.
