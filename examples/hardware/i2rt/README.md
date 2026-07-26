# I2RT YAM Hardware

This adapter connects a two-follower YAM Cell-style pair (`i2rt_dual_yam`) to
EVA through the existing ZMQ transport. Intel RealSense D405 cameras are bound
by serial number so camera roles remain stable across reboots. The adapter
intentionally targets the base YAM arm; YAM Pro, YAM Ultra, and Big YAM need
their own limits and robot descriptions before they can be enabled safely.

The official I2RT SDK pins NumPy 2.2.6 while EVA Client pins NumPy 1.26.4. Its
dependencies therefore live in this directory's own `pyproject.toml` and its
environment is created under `examples/hardware/i2rt/.venv`; the two processes
exchange only ZMQ messages.

## 1. One-time SDK setup

From the EVA Client root:

```bash
bash examples/hardware/i2rt/setup_sdk.sh
```

The SDK is a Git submodule pinned to the tested `Noietch/i2rt` fork
(`main`, commit `fd90afef`), based on the official v1.2.4 release. The fork
contains the reviewed EVA compatibility and safety changes, so setup initializes
that exact revision directly and runs `uv sync --project examples/hardware/i2rt`.
The subproject declares `i2rt`, `pyrealsense2`, `pyzmq`, the EVA PyRoki/JAX FK
stack, and the SDK's NumPy/build constraints. The hardware node uses the same
registered EVA kinematics model as inference, replay, and visualization.

To install the official boot-time CAN udev rule:

```bash
sudo sh examples/hardware/i2rt/SDK/i2rt/devices/install_devices.sh
```

## 2. CAN and D405 devices

The adapter uses 1 Mbit/s CAN. `run_hardware.sh` brings configured interfaces
up automatically; persistent I2RT names are preferred when present, otherwise
it falls back to `can0` through `can3`. Run it in an interactive terminal so
`sudo` can request the local password when the interfaces need configuration.

To query camera model and serial through the RealSense SDK without starting a
stream:

```bash
bash examples/hardware/i2rt/run_hardware.sh --list-cameras
```

The hardware launcher binds all three installed D405 cameras by serial rather
than relying on USB enumeration order.

If a CAN interface is stuck:

```bash
sh examples/hardware/i2rt/SDK/i2rt/scripts/reset_all_can.sh
```

## 3. Run dual YAM followers

`run_hardware.sh` defaults to the dual-YAM layout, enables both teaching-handle
leaders, and maps followers/leaders to `can0` through `can3` when persistent CAN
names are unavailable. No robot or leader enable switch is required:

```bash
LEFT_FOLLOWER_CAN=can_follower_l \
RIGHT_FOLLOWER_CAN=can_follower_r \
LEFT_LEADER_CAN=can_leader_l \
RIGHT_LEADER_CAN=can_leader_r \
  bash examples/hardware/i2rt/run_hardware.sh
```

With no `I2RT_GRIPPER_LIMITS` override, the launcher allows the SDK to calibrate
both motorized grippers automatically. Start both grippers in the SDK-required
closed position and keep the workspace clear while they move to their hard
stops. A verified `I2RT_GRIPPER_LIMITS=CLOSED,OPEN` override skips calibration.

Then launch EVA:

```bash
eva --config configs/01_deploy/i2rt_dual_yam/openpi_qpos.py --web-port 8080
```

The verified default D405 mapping is:

- `cam_high=260422275306`
- `cam_left_wrist=260422273576`
- `cam_right_wrist=260322279472`

All three streams default to `640x480` at 30 FPS. Override the mappings with
`D405_CAM_HIGH_SERIAL`, `D405_CAM_LEFT_WRIST_SERIAL`, and
`D405_CAM_RIGHT_WRIST_SERIAL`; override the stream with `D405_CAMERA_WIDTH`,
`D405_CAMERA_HEIGHT`, `D405_CAMERA_FPS`, and `D405_CAMERA_TIMEOUT_MS`.

The supplied dual-arm EVA deploy and collection configs enable and record all
three camera keys. The hardware node always starts all three streams.

## Dual-leader collection and HIL

Leader collection/HIL is available only for `i2rt_dual_yam`, with both teaching
handles enabled together by default. Running `run_hardware.sh` immediately
captures relative leader/follower anchors and starts direct dual-arm teleoperation:

```bash
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
Both leader CAN interfaces must be present and distinct; left-only and
right-only leader control are not supported.
Each teaching-handle leader starts in the SDK's gravity-compensation mode using
the YAM model and the teaching handle's modeled 0.258 kg end-effector load.
Set `ENABLE_I2RT_LEADERS=0` only for policy deployment without teaching
handles.

Leader/follower control runs at 200 Hz by default while an independent publisher
thread sends cached camera/state observations at 30 Hz. The publisher never
touches the CAN SDK. Override the rates independently with `CONTROL_RATE` and
`PUBLISH_RATE`. The periodic hardware status log includes the achieved control
rate and per-joint `target - current` tracking error in radians.

After selecting a collection task and switching `ARM ON` in the Collection page,
either leader's `RECORD` button starts an episode when idle and ends/saves it
while recording. `SYNC` is the cancel button: while recording it stops and
discards the current episode. The buttons cannot arm motion by themselves.
Leaving Collection or switching `ARM OFF` stops recording/HIL state, while
direct leader control remains active until the hardware node exits or
`ENABLE_I2RT_LEADERS=0` is used.

An optional bounded outer-loop integral trim can remove gravity, friction, and
small encoder-zero steady-state errors without increasing the SDK's inner-loop
stiffness. It is disabled by default; enable it with `I2RT_TRACKING_KI=2.0`.
The correction limit, deadband, settle delay, and startup learning duration have
matching `I2RT_TRACKING_*` / `I2RT_STARTUP_TRIM_DURATION` variables.

For the currently verified workstation mapping (can2 left follower, can1 right
follower, can3 left leader, can0 right leader, and the three D405 serials listed
above), use the sole hardware launcher directly:

```bash
bash examples/hardware/i2rt/run_hardware.sh
```

This defaults to the physical `linear_4310` gripper and starts the left and right
leaders as a required pair. The CAN mappings and camera serial remain
overridable through the environment. CAN interface numbers can change after
replugging or rebooting, so inspect the devices again before using the fallbacks
if the USB layout changes.
Do not reuse raw `I2RT_GRIPPER_LIMITS` values across process restarts unless
they have been converted into the SDK's post-wrap motor coordinate frame;
an unverified override is unsafe.
Leave this terminal running; both leaders connect when the hardware node starts.

Both followers and both leaders move smoothly to their all-zero arm position by
default over `I2RT_STARTUP_DURATION` seconds. The leaders then return to gravity
compensation before direct relative control starts. Set
`I2RT_STARTUP_POSITION=current` to skip the startup motion. `I2RT_JOINT4_KP`
remains available for a calibrated fourth-joint gain override.

## Safety behavior

- Commands are clipped to the official YAM joint limits and gripper `[0, 1]`.
- If commands stop for 0.5 seconds, connected followers return to the configured
  idle mode. The default is `gravity_comp`; override with `I2RT_IDLE_MODE` only
  after the gravity model and end-effector load have been calibrated.
- Closing the node calls the SDK's safe `close()` path.
- The pinned SDK fork closes partially initialized CAN connections and waits
  for the motor-control loop before closing its socket.
- Linear grippers calibrate by default when no verified limit override is
  supplied. Follow I2RT's requirement to start them fully closed.
