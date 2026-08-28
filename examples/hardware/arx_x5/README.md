# ARX X5 Dual-Arm Hardware Startup

This directory starts the dual-arm X5, motor CAN, three Intel RealSense D405
cameras, and the EVA ZMQ hardware node. X5 and R5 use the same state, action,
and VR control flow; only the X5 SDK and X5A URDF differ.

## Fixed Configuration

The launcher has the following values configured for the current machine; no
environment variables are required:

```text
left_arm:       /dev/arxcan1 -> can1
right_arm:      /dev/arxcan3 -> can3
X5 type:        2 (2025 dual-track gripper)
state endpoint: tcp://127.0.0.1:5555
action endpoint: tcp://127.0.0.1:5556
cameras:        640x480 BGR8, 30 FPS
web port:       8080
```

D405 cameras use fixed serial numbers and do not depend on SDK enumeration
order:

```text
cam_high:        409122271504
cam_left_wrist:  352122272510
cam_right_wrist: 352122271326
```

All three D405 cameras automatically load `d405_profile.yaml` at startup. The
launcher uses the `day` profile by default: auto exposure is disabled and the
high/left/right cameras use fixed exposures of `6000/8000/7000 us`. Use
`--realsense-profile night` to switch to the night values
`12000/14000/14000 us`. Both profiles use white-balance values calibrated with
the whiteboard: high `4390 K`, left `4450 K`, and right `4410 K`. Each camera
also loads its small BGR correction gain.

The D405 auto white-balance mode works, but when the white desk or whiteboard
covers much of the frame, auto exposure pushes about 68%-86% of pixels into
the saturated region. X5 therefore disables auto exposure by default to avoid
overexposing the board and desk. Recalibrate the corresponding day/night
values in `d405_profile.yaml` when the collection environment changes.

## Initial Installation

X5 uses an independent Python 3.12 project and virtual environment without
modifying the repository root environment or R5. Run:

```bash
git submodule update --init --recursive
bash examples/hardware/arx_x5/setup_env.sh
```

`run_hardware.sh` and `run_fake.sh` use only
`examples/hardware/arx_x5/.venv/bin/python`; they neither activate nor read the
repository root `.venv`. If `python3.12` is not in `PATH`, specify the
interpreter during installation:

```bash
ARX_X5_PYTHON=/path/to/python3.12 bash examples/hardware/arx_x5/setup_env.sh
```

The script performs the following steps:

1. Create `examples/hardware/arx_x5/.venv`.
2. Install `pyrealsense2` and the X5 hardware-node dependencies.
3. Use the official bundled
   [ARARX_X5_beta SDK-V2](https://github.com/ARXroboticsX/ARARX_X5_beta),
   currently pinned to `1aa8a7d2ddefced2229f953feffc248be1c0b44d` at
   `examples/hardware/arx_x5/SDK/X5`.
4. Validate the official CPython 3.12 binary and `SingleArm` interface, then
   install dependencies required by vendor imports, including `pin`,
   `python-can`, and `ruckig`. Installation does not run FK/IK calculations or
   connect to the robot.

The new X5 integration code in EVA Client follows the repository root's
Apache-2.0 license. `ARARX_X5_beta` is an independent third-party submodule
outside EVA Client's Apache-2.0 license scope. The pinned version currently
does not include a license file; confirm upstream licensing before public use
or redistribution.

SDK-V2 uses SocketCAN and its own 200 Hz control thread directly, without the
legacy ROS/KDL pybind extensions. X5 FK, IK, and VR retargeting still use the
repository's `src/robots/kinematics/pyroki.py` and X5A URDF.

## Start X5 Hardware

After initial installation, run one command:

```bash
bash examples/hardware/arx_x5/run_hardware.sh
```

The script starts only the dual-arm motors, three D405 cameras, and the
hardware ZMQ node. On `Ctrl-C`, the arms enter protective mode and camera and
ZMQ ports are released. Start WebXR and EVA separately in two other terminals:

At each startup, the SDK-V2 CAN bus first sends a vendor `clear error` frame
only to X5-2025 gripper motor ID 8 and verifies that the fault code is 0. It
does not change the gripper zero or send position commands to the six-axis
motors at this stage. If the gripper fault remains or no feedback is received,
the hardware node does not continue startup.

```bash
python examples/input_sources/vr_webxr/node.py \
  --host 127.0.0.1 \
  --port 43876 \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766
```

```bash
eva --config configs/02_collection/arx_x5_vr.py
```

Open `http://127.0.0.1:8080` in a browser. The headset accesses the WebXR page
through the local host or ADB forwarding:

```text
http://127.0.0.1:43876/?token=<TOKEN_FROM_NODE_LOG>&mode=ar
```

## Hardware-Free Fake X5

Fake X5 uses the same EVA ZMQ action/state protocol as the real hardware and
does not require the X5 SDK, CAN, or RealSense. By default, it sets state
directly to each received action qpos and sends periodic color images for all
three cameras.

Start the fake node from the repository root:

```bash
bash examples/hardware/arx_x5/run_fake.sh
```

To simulate smoother mechanical response, switch to an independent second-order
system for each qpos:

```bash
bash examples/hardware/arx_x5/run_fake.sh --dynamics-mode second-order
```

Enter `r` followed by Return to reset qpos, velocity, and the action target to
the X5 initial values; enter `q` followed by Return to exit. The fake-plant
iteration lifecycle is independent of EVA ARM, Collect, and recording states.

In another terminal, start the WebXR node as described above, then start EVA:

```bash
eva --config configs/02_collection/arx_x5_vr.py
```

This configuration keeps all three camera streams, and collection output is
written to `work_dirs/collection/arx_x5_vr`.

## Startup Behavior

After X5 `_connect` reads the current six-axis position, it writes the same
values back to enter position control. The hardware node then uses the SDK-V2
trajectory interface to bring both arms synchronously to their initial poses
in `5 s`. EVA policy and dataset sampling remain at 30 Hz. The hardware node
uses `--rate 100` for action/state communication and passes approximately
`10 ms` durations to the official Ruckig/200 Hz control thread for
interpolation. Each new 30 FPS camera frame is attached once, while the saver
aligns samples to a 30 FPS fixed time grid using `collection.storage.fps`.
J2/J3 lower limits use
the official physical range starting at `0°`, so IK negative margin is not
continuously pushed against the mechanical zero. The default initial pose is
the official X5 home pose; use `--start-at-zero` below when all-zero startup is
needed.

Press `Ctrl-C` to stop; the node enters protective mode and releases the
cameras. To temporarily skip one arm, use:

```bash
bash examples/hardware/arx_x5/run_hardware.sh --disabled-arm left_arm
```

If a single joint does not power on at startup, run the fault-recovery startup:

```bash
bash examples/hardware/arx_x5/reset_hardware.sh
```

The script first stops any leftover hardware node, rebuilds the left and right
`slcand`/CAN interfaces, and performs SDK enable checks plus complete
`disable`/`close` operations for each arm. It also moves right-arm J2 by
`0.01 rad` to verify that it responds. It retries at most three times by
default. Startup and homing continue only when both arms have empty
`offline_joints`, `fault` is `None`, and the right J2 micro-motion succeeds.
Set `ARX_X5_RESET_ATTEMPTS` to change the retry count:

```bash
ARX_X5_RESET_ATTEMPTS=5 bash examples/hardware/arx_x5/reset_hardware.sh
```

To move both arms to all-zero six-axis positions when the node starts, use:

```bash
bash examples/hardware/arx_x5/run_hardware.sh --start-at-zero
```

The default startup still moves to the official X5 home pose.

Viewing the parameters does not connect to the motors or initialize CAN:

```bash
bash examples/hardware/arx_x5/run_hardware.sh --help
```
