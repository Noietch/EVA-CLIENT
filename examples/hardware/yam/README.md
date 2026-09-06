# YAM YAM Hardware

## Startup and Normal Stop Poses

`config.yaml` defines `robot.config.robot.initial_qpos` and `safe_qpos` as 14 values:
left six joints, left gripper, right six joints, right gripper. Joint angles are
in radians. YAM gripper values use `1=open` and `0=closed`; the console remaps
that convention to the URDF so the 3D fingers match the physical grippers. The
console preserves measured gripper positions during these moves.
If `safe_qpos` is omitted, it defaults to `initial_qpos`. The configured YAM
initial joint targets match the X5 nominal
targets; this does not imply identical Cartesian poses between the two robots.
Validate the pose and unobstructed path on your installation before a real move.

Robot Start moves to `initial_qpos`. Normal Robot Stop locks teleoperation, moves
to `safe_qpos`, checks fresh feedback and settling, then closes the hardware node.
Failure or interruption withholds shutdown and leaves the process available for
recovery. Robot shutdown is not forcibly killed on timeout. The YAM SDK releases
motor torque when closing; this is not a mains-power relay operation. Zero joint
angles alone do not prove mechanical support or power-loss safety.

This normal shutdown sequence is not an emergency stop. Use the physical emergency
stop when immediate intervention is necessary; do not wait for a parking move.

This adapter connects a two-follower YAM Cell-style pair (`dual_yam`) to
EVA through the existing ZMQ transport. The overhead and two wrist Orbbec
cameras are bound by serial number so camera roles remain stable across reboots.
An Intel RealSense D405 remains available as an optional alternative overhead
camera. The adapter
intentionally targets the base YAM arm; YAM Pro, YAM Ultra, and Big YAM need
their own limits and robot descriptions before they can be enabled safely.

The official YAM SDK pins NumPy 2.2.6 while EVA Client pins NumPy 1.26.4. Its
dependencies therefore live in this directory's own `pyproject.toml` and its
environment is created under `examples/hardware/yam/.venv`; the two processes
exchange only ZMQ messages.

## 1. One-time SDK setup

From the EVA Client root:

```bash
bash examples/hardware/yam/setup_sdk.sh
```

The SDK is a Git submodule pinned to the tested `Noietch/i2rt` fork
(`main`, commit `70a053b`), based on the official v1.2.4 release. The fork
contains the reviewed EVA compatibility and safety changes, so setup initializes
that exact revision directly and runs `uv sync --project examples/hardware/yam`.
The subproject declares `i2rt`, `pyrealsense2`, `pyorbbecsdk2`, OpenCV, `pyzmq`,
the EVA PyRoki/JAX FK stack, and the SDK's NumPy/build constraints. The hardware node uses the same
registered EVA kinematics model as inference, replay, and visualization.

To install the official boot-time CAN udev rule:

```bash
sudo sh examples/hardware/yam/SDK/i2rt/devices/install_devices.sh
```

## 2. CAN and camera devices

The adapter uses 1 Mbit/s CAN. `run_hardware.sh` brings configured interfaces
up automatically; persistent YAM names are preferred when present, otherwise
it falls back to `can0` through `can3`. Run it in an interactive terminal so
`sudo` can request the local password when the interfaces need configuration.

To query camera model, serial, and USB connection through both vendor SDKs
without starting a stream:

```bash
bash examples/hardware/yam/run_hardware.sh --list-cameras
```

The hardware launcher binds both camera families by serial rather than relying
on USB enumeration order.

If a CAN interface is stuck:

```bash
sh examples/hardware/yam/SDK/i2rt/scripts/reset_all_can.sh
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
  bash examples/hardware/yam/run_hardware.sh
```

With no `YAM_GRIPPER_LIMITS` override, the launcher allows the SDK to calibrate
both motorized grippers automatically. Start both grippers in the SDK-required
closed position and keep the workspace clear while they move to their hard
stops. A verified `YAM_GRIPPER_LIMITS=CLOSED,OPEN` override skips calibration.

Then launch EVA:

```bash
eva --config configs/01_deploy/dual_yam/openpi_qpos.py --web-port 8080
```

The default three-camera mapping is:

- `cam_high=CP0HC530000Z` (Orbbec Gemini 335)
- `cam_left_wrist=CV2R1610003Z` (Orbbec Gemini 305)
- `cam_right_wrist=CV2L360000CL` (Orbbec Gemini 305)

The launcher enables all three Orbbec keys by default. For single-camera
diagnosis, set `ORBBEC_ENABLED_CAMERAS=cam_right_wrist`. Streams default to
`640x480` MJPG at 30 FPS, the highest rate verified with all three cameras on
this workstation. A 60 FPS override is available, but all three devices are not
reliably concurrent at that rate. Override mappings with
`ORBBEC_CAM_HIGH_SERIAL`, `ORBBEC_CAM_LEFT_WRIST_SERIAL`, and
`ORBBEC_CAM_RIGHT_WRIST_SERIAL`; tune the stream with `ORBBEC_CAMERA_WIDTH`,
`ORBBEC_CAMERA_HEIGHT`, `ORBBEC_CAMERA_FPS`, `ORBBEC_CAMERA_FORMAT`, and
`ORBBEC_CAMERA_TIMEOUT_MS`. Auto exposure remains enabled and a small default
brightness compensation of `+5` is applied; override it with
`ORBBEC_CAMERA_BRIGHTNESS` (`-64..64`).

All enabled Orbbec streams share one isolated SDK process and one SDK context.
The launcher starts the left, right, and overhead streams sequentially and
waits for all of them to be online before bringing up the motors.
By default, `run_hardware.sh` also resets each enabled Orbbec USB device once
before opening the SDK, which clears stale streams left by a previous run.
Set `ORBBEC_USB_RESET=0` to skip this recovery step.
The launcher holds a single-instance lock so a second invocation cannot reset
cameras that are already streaming from a live hardware node.

Before multi-camera capture, the launcher checks that the Linux USB transfer
buffer is at least 256 MB. Set it for the current boot with:

```bash
echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

On workstations with at least 32 logical CPUs, the launcher defaults to CPU
affinity `0-15` so camera work does not contend with the USB controller IRQ.
Set `YAM_CPU_AFFINITY=` to disable affinity or provide another CPU list.

The three Orbbec streams run in one isolated camera process, explicitly enable
color auto exposure, and discard 30 frames before publishing. Camera waits do
not run in the motor SDK process. Override the warmup with
`ORBBEC_CAMERA_WARMUP_FRAMES`.

To use the optional D405 overhead camera, disable the Orbbec `cam_high` mapping
and enable the D405, for example
`ORBBEC_ENABLED_CAMERAS=cam_left_wrist,cam_right_wrist D405_ENABLED_CAMERAS=cam_high`.
For the D405, every `run_hardware.sh` launch loads
`profiles/d405_workcell.json`. It enables automatic exposure, turns on the D405
exposure/gain limit toggles, caps exposure at 33 ms and gain at 64, and discards
90 startup frames while exposure converges. Override the profile with
`D405_CAMERA_PROFILE`; the individual
`D405_CAMERA_AUTO_EXPOSURE_LIMIT_US`, `D405_CAMERA_AUTO_GAIN_LIMIT`,
`D405_CAMERA_EXPOSURE_US`, and `D405_CAMERA_WARMUP_FRAMES` variables remain
available for temporary tuning.

The supplied dual-arm EVA deploy and collection configs enable and record all
three camera keys.

## Dual-leader collection and HIL

Leader collection/HIL is available only for `dual_yam`, with both teaching
handles enabled together by default. Running `run_hardware.sh` immediately
captures relative leader/follower anchors and starts direct dual-arm teleoperation:

```bash
LEFT_LEADER_CAN=can_leader_l \
RIGHT_LEADER_CAN=can_leader_r \
  bash examples/hardware/yam/run_hardware.sh

eva --config configs/02_collection/dual_yam.py --web-port 8080
```

The real node publishes leader positions as `action.qpos` while commanding the
followers. Collection and relative HIL snapshot the current leader/follower
poses and apply leader deltas on top of the follower pose, so different motor
zero offsets do not cause a jump at takeover. Grippers are the exception: each
teaching handle is mapped through its calibrated raw encoder endpoints and sent
in absolute `[0, 1]` space, so fully open and fully closed always reach the
follower endpoints. Absolute HIL remains available when both arms have been
calibrated into the same joint coordinate frame.
Both leader CAN interfaces must be present and distinct; left-only and
right-only leader control are not supported.
Each teaching-handle leader starts in the SDK's gravity-compensation mode using
the YAM model and the teaching handle's modeled 0.258 kg end-effector load.
Set `ENABLE_YAM_LEADERS=0` only for policy deployment without teaching
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
`ENABLE_YAM_LEADERS=0` is used.

An optional bounded outer-loop integral trim can remove gravity, friction, and
small encoder-zero steady-state errors without increasing the SDK's inner-loop
stiffness. It is disabled by default; enable it with `YAM_TRACKING_KI=2.0`.
The correction limit, deadband, settle delay, and startup learning duration have
matching `YAM_TRACKING_*` / `YAM_STARTUP_TRIM_DURATION` variables.

For the currently verified workstation mapping (can1 left follower, can2 right
follower, can0 left leader, can3 right leader, and the three D405 serials listed
above), use the sole hardware launcher directly:

```bash
bash examples/hardware/yam/run_hardware.sh
```

This defaults to the physical `linear_4310` gripper and starts the left and right
leaders as a required pair. The CAN mappings and camera serial remain
overridable through the environment. CAN interface numbers can change after
replugging or rebooting, so inspect the devices again before using the fallbacks
if the USB layout changes. The launcher first matches the four verified
CANable USB serials and only then falls back to `can0` through `can3`; override
`YAM_CAN_SERIAL_LEFT_FOLLOWER`, `YAM_CAN_SERIAL_RIGHT_FOLLOWER`,
`YAM_CAN_SERIAL_LEFT_LEADER`, and `YAM_CAN_SERIAL_RIGHT_LEADER` when adapters
are replaced or moved to another workstation.
The verified teaching-handle encoder endpoints are passed as raw radians in
`OPEN,CLOSED` order. Override them after recalibration with
`LEFT_LEADER_GRIPPER_ENDPOINTS` and `RIGHT_LEADER_GRIPPER_ENDPOINTS`.
Do not reuse raw `YAM_GRIPPER_LIMITS` values across process restarts unless
they have been converted into the SDK's post-wrap motor coordinate frame;
an unverified override is unsafe.
Leave this terminal running; both leaders connect when the hardware node starts.

Both followers and both leaders move smoothly to their all-zero arm position by
default over `YAM_STARTUP_DURATION` seconds. The leaders then return to gravity
compensation before direct relative control starts. Set
`YAM_STARTUP_POSITION=current` to skip the startup motion. `YAM_JOINT4_KP`
remains available for a calibrated fourth-joint gain override.

## Safety behavior

- Commands are clipped to the official YAM joint limits and gripper `[0, 1]`.
- If commands stop for 0.5 seconds, connected followers return to the configured
  idle mode. The default is `gravity_comp`; override with `YAM_IDLE_MODE` only
  after the gravity model and end-effector load have been calibrated.
- Closing the node calls the SDK's safe `close()` path.
- The pinned SDK fork closes partially initialized CAN connections and waits
  for the motor-control loop before closing its socket. It also filters CAN
  replies by motor ID, so stale frames cannot be decoded as another motor.
- Linear grippers calibrate by default when no verified limit override is
  supplied. Follow YAM's requirement to start them fully closed.
