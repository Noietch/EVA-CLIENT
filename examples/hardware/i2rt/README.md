# I2RT YAM Hardware

This adapter connects one YAM arm (`i2rt_yam`) or a two-follower YAM Cell-style
pair (`i2rt_dual_yam`) to EVA through the existing ZMQ transport. Intel
RealSense D405 cameras are bound by serial number so camera roles remain stable
across reboots. The adapter intentionally targets the base YAM arm; YAM Pro,
YAM Ultra, and Big YAM need their own limits and robot descriptions before they
can be enabled safely.

The official I2RT SDK pins NumPy 2.2.6 while EVA Client pins NumPy 1.26.4. Its
dependencies therefore live in this directory's own `pyproject.toml` and its
environment is created under `examples/hardware/i2rt/.venv`; the two processes
exchange only ZMQ messages.

## 1. One-time SDK setup

From the EVA Client root:

```bash
bash examples/hardware/i2rt/setup_sdk.sh
```

The SDK is an official Git submodule pinned to the latest stable release tested
with this adapter (`v1.2.4`, commit `5d47b358`). Setup initializes that exact
revision, applies the reviewed EVA compatibility/safety patches, and runs
`uv sync --project examples/hardware/i2rt`. The subproject
declares `i2rt`, `pyrealsense2`, `pyzmq`, and the SDK's NumPy/build constraints.

To install the official boot-time CAN udev rule:

```bash
sudo sh examples/hardware/i2rt/SDK/i2rt/devices/install_devices.sh
```

## 2. Inspect CAN and D405 cameras

The adapter uses 1 Mbit/s CAN. `run_hardware.sh` brings configured interfaces
up automatically; persistent I2RT names are preferred when present, otherwise
it falls back to `can0`, `can1`, and so on.

For a strictly read-only check that does not bring CAN up or start camera
streaming:

```bash
bash examples/hardware/i2rt/inspect_hardware.sh
```

To bring the three currently attached CAN adapters up at 1 Mbit/s, run this in
an interactive terminal so `sudo` can request the local password:

```bash
bash examples/hardware/i2rt/bring_up_can.sh
```

The script only configures the network interfaces; it does not create an I2RT
robot object or command any motors. Interface names can also be passed
explicitly, for example `bring_up_can.sh can0 can1 can2`.

To query camera model and serial through the RealSense SDK without starting a
stream:

```bash
bash examples/hardware/i2rt/run_hardware.sh --list-cameras
```

Use the serial reported on the actual machine rather than relying on USB
enumeration order.

If a CAN interface is stuck:

```bash
sh examples/hardware/i2rt/SDK/i2rt/scripts/reset_all_can.sh
```

## 3. Run a single YAM

Terminal 1, hardware node:

```bash
I2RT_ROBOT=i2rt_yam \
FOLLOWER_CAN=can0 \
D405_CAM_HIGH_SERIAL=<D405_SERIAL> \
  bash examples/hardware/i2rt/run_hardware.sh
```

Add a wrist D405 with `D405_CAM_WRIST_SERIAL=<SERIAL>`.
The supplied EVA config disables `cam_wrist` by default for the current
single-D405 setup; remove it from `transport.disabled_cameras` in a local config
when the wrist camera is added.

Terminal 2, EVA:

```bash
source .venv/bin/activate
eva --config configs/01_deploy/i2rt_yam/openpi_qpos.py --web-port 8080
```

## 4. Run dual YAM followers

```bash
I2RT_ROBOT=i2rt_dual_yam \
LEFT_FOLLOWER_CAN=can_follower_l \
RIGHT_FOLLOWER_CAN=can_follower_r \
D405_CAM_HIGH_SERIAL=<D405_SERIAL> \
  bash examples/hardware/i2rt/run_hardware.sh
```

Then launch EVA:

```bash
eva --config configs/01_deploy/i2rt_dual_yam/openpi_qpos.py --web-port 8080
```

Optional wrist D405 variables are `D405_CAM_LEFT_WRIST_SERIAL` and
`D405_CAM_RIGHT_WRIST_SERIAL`. You can also pass mappings directly:

```bash
bash examples/hardware/i2rt/run_hardware.sh \
  --camera cam_high=<SERIAL> \
  --camera cam_left_wrist=<SERIAL> \
  --camera cam_right_wrist=<SERIAL>
```

The supplied dual-arm EVA config likewise disables both wrist camera keys by
default. Enable only the keys whose serial mappings are passed to the node, and
add those keys to the collection schema when recording them.

## Leader-follower collection and HIL

Single-arm collection uses `LEADER_CAN`. Dual-arm collection enables both
teaching-handle leaders with `ENABLE_I2RT_LEADERS=1`:

```bash
I2RT_ROBOT=i2rt_dual_yam \
ENABLE_I2RT_LEADERS=1 \
LEFT_LEADER_CAN=can_leader_l \
RIGHT_LEADER_CAN=can_leader_r \
  bash examples/hardware/i2rt/run_hardware.sh

eva --config configs/02_collection/i2rt_dual_yam.py --web-port 8080
```

The real node publishes leader positions as `action.qpos` while commanding the
followers. Collection and relative HIL snapshot the current leader/follower
poses and apply leader deltas on top of the follower pose, so different motor
zero offsets do not cause a jump at takeover. Absolute HIL remains available
when both arms have been calibrated into the same joint coordinate frame.

Leader/follower control runs at 200 Hz by default while camera/state publishing
remains at 30 Hz. Override these independently with `CONTROL_RATE` and
`PUBLISH_RATE`. The periodic hardware status log includes the achieved control
rate and per-joint `target - current` tracking error in radians.

After selecting a collection task and switching `ARM ON` in the Collection page,
the leader's `RECORD` button toggles the recording episode: one debounced press
starts recording and the next ends/saves it. The button cannot arm motion by
itself; leaving Collection or switching `ARM OFF` remains the safety gate and
disconnects leader control. The `SYNC` button is intentionally unused.

The verified left-leader launcher also enables a bounded outer-loop integral
trim (`Ki=2.0`, maximum correction `0.12 rad`). It learns the static correction
after the startup zero move and only continues integrating after a runtime
target has stayed nearly stationary for 0.15 seconds. This removes gravity,
friction, and small encoder-zero steady-state errors without increasing the
SDK's inner-loop stiffness. Errors below `0.002 rad` are ignored to avoid
limit-cycle chatter. Override or disable it with `I2RT_TRACKING_KI=0`; the
limit, deadband, delay, and startup learning duration have matching
`I2RT_TRACKING_*` / `I2RT_STARTUP_TRIM_DURATION` environment variables.

When only the left teaching handle is available, enable it independently:

```bash
I2RT_ROBOT=i2rt_dual_yam \
ENABLE_I2RT_LEFT_LEADER=1 \
LEFT_LEADER_CAN=can_leader_l \
  bash examples/hardware/i2rt/run_hardware.sh
```

In this partial-leader mode, starting collection or HIL snapshots the unled
right follower's current joint position and holds that anchor while the left
leader drives only the left follower. Stopping collection/HIL releases both
followers back to gravity-compensation idle. `ENABLE_I2RT_RIGHT_LEADER=1`
enables the symmetric right-only setup; `ENABLE_I2RT_LEADERS=1` remains the
shorthand for both leaders.

For the currently verified workstation mapping (can0 left follower, can1 right
follower, can2 left leader, and D405 serial `260422275306`), use the ready-made
launcher:

```bash
bash examples/hardware/i2rt/run_left_leader.sh
```

This defaults to the physical `linear_4310` gripper, starts only the left
leader, keeps the right leader disabled, and moves both followers smoothly to
six all-zero arm joints over five seconds before holding that position. On the
first connection, the SDK may move each gripper while detecting its limits.
The CAN mappings and camera serial remain overridable through
the environment. CAN interface numbers can change after replugging or rebooting,
so inspect the devices again before using the defaults if the USB layout changes.
Do not reuse raw `I2RT_GRIPPER_LIMITS` values across process restarts unless
they have been converted into the SDK's post-wrap motor coordinate frame;
an unverified override is unsafe.
Motorized-gripper startup is interlocked: provide a verified
`I2RT_GRIPPER_LIMITS=CLOSED,OPEN` override, or deliberately set
`I2RT_ALLOW_GRIPPER_CALIBRATION=1` with both grippers clear and supervised.
Leave this terminal running; the leader connects lazily when EVA starts HIL or
collection. The D405 frame timeout defaults to three seconds and can be changed
with `D405_TIMEOUT_MS`.

Override `I2RT_STARTUP_DURATION` to change the zeroing time, or set
`I2RT_STARTUP_POSITION=current` to hold the measured startup pose instead.
The workstation launcher also raises both local fourth-joint position gains from
10 to the SDK's YAM Pro/Ultra value of 40, because flattened joints `joint03`
and `joint10` are those two local fourth joints. Override with
`I2RT_JOINT4_KP` if the hardware is retuned.

## Safety behavior

- Commands are clipped to the official YAM joint limits and gripper `[0, 1]`.
- If commands stop for 0.5 seconds, connected followers return to the configured
  idle mode. The workstation launcher uses `hold_position` because its current
  gravity-compensation calibration does not support the arm safely. Generic
  launches retain `gravity_comp`; override with `I2RT_IDLE_MODE` only after the
  gravity model and end-effector load have been calibrated.
- Closing the node calls the SDK's safe `close()` path.
- Setup applies a small SDK safety patch that closes partially initialized CAN
  connections and waits for the motor-control loop before closing its socket.
- Linear grippers may calibrate at first connection. Follow I2RT's requirement
  to start them fully closed or complete gripper calibration first.

## Software-only verification

No SDK, CAN, or D405 is required:

```bash
bash examples/hardware/i2rt/run_fake_node.sh
eva --config configs/01_deploy/i2rt_dual_yam/openpi_qpos.py --web-port 8080
```

The official I2RT simulator can also exercise the real execution-node protocol:

```bash
I2RT_SIM=1 I2RT_ROBOT=i2rt_yam \
  bash examples/hardware/i2rt/run_hardware.sh
```
