[← Back to README](../README.md)

# Core concepts

## Device processes and transports

EVA-CLIENT runs policy inference, operator controls, recording, and the console.
Hardware SDKs and camera capture run in separate processes under
[`examples/hardware/`](../examples/hardware/) and
[`examples/input_sources/`](../examples/input_sources/). Device definitions in
their `config.yaml` files supply hardware defaults; Python presets select the
model and workflow. The console's DEVICE workspace manages device selection,
settings, startup, and health.

The transport reads observations and publishes robot commands. Policy EEF
targets are converted to joint commands before reaching the execution layer.

| `transport.type` | Purpose | Requirement |
| --- | --- | --- |
| `ros1` | ROS 1 robot topics | ROS 1 environment |
| `ros2` | ROS 2 robot topics, including compressed camera streams | ROS 2 environment |
| `zmq` | Execution node using EVA's ZeroMQ observation/action protocol | Matching execution node |
| `dataset` | Offline LeRobot v2.1 episode replay | Local dataset with mapped state, action, and video columns |

For ZMQ, `transport.image_mode="stream"` consumes streaming images.
`"on_demand"` sends an image request before policy inference and waits for a new
image revision, bounded by `image_request_timeout_s` (default 2 seconds). The
execution node must implement image requests; a timeout returns no policy frame.

## Policy backends

An EVA observation contains `state` (flat float32 vector), `images` (camera name
to uint8 array), and `prompt` (string). Image size, RGB conversion, padding, and
HWC/CHW layout are configured before the policy call. They must match the model's
input contract. The policy returns `actions` shaped `[T, D]`: one row per action
step, with `D` determined by the configured action space and robot.

| `policy.type` | Connection | Contract |
| --- | --- | --- |
| `openpi` | WebSocket + OpenPI msgpack-numpy | Sends the observation directly; remote reset is used when metadata advertises `supports_reset` |
| `openpi_rtc` | WebSocket + OpenPI msgpack-numpy | Feeds back the previous response's `origin_actions` or `actions`, shifted by `backend_options.latency_k` |
| `starvla` | WebSocket + typed msgpack envelope | Sends all camera images in observation order, language, and state; optional `unnorm_key` |
| `xpolicylab` | WebSocket + typed msgpack-numpy frames | `hello`, `infer`, and `reset` exchanges; explicit joint/EEF arm and gripper dimensions |
| `gr00t` | ZeroMQ REQ/REP + msgpack-numpy | Embodiment-specific `video_keys`, `state_key`, `language_key`, and `action_keys` |
| `mock` | Local | Synthetic action chunks for integration debugging |
| `replay` | Local | Recorded dataset actions; requires a dataset transport |

[DreamLabs](../examples/policy/dreamlabs/README.md) uses a separate API bridge and
the `openpi` backend. It is not a separate `policy.type`. Its dual YAM preset
selects HWC RGB images at 320 × 176 and a 20 Hz action rate.

XPolicyLab sends instruction, named camera color arrays, and named state fields.
For two joint-space arms, wire state/action order is left joints, left gripper,
right joints, right gripper. EEF arms use seven values (`xyz + wxyz`) plus the
configured gripper width. Reset creates a new trial identity and restarts the
inference-step counter. See [configuration](configuration.md) for presets.

## Inference strategies

The backend controls the wire protocol; the strategy controls how action chunks
are scheduled. DEBUG allows switching among the configured strategies.

| Key | Behavior |
| --- | --- |
| `sync` | Blocking inference, then executes a cropped chunk; the default horizon is 5 steps |
| `async` | Background inference with linear smoothing across overlapping chunks |
| `naive` | Background inference; each new chunk replaces the buffered trajectory |
| `act` | Background inference with temporal ensembling of overlapping predictions |
| `rtc` | Background inference with latency compensation and linear overlap; pair with `openpi_rtc` for RTC server conditioning |

`execute_horizon` crops chunks. `inference_cfg.inference_rate` controls background
request frequency (default 3 Hz), while `publish_rate` controls action publication
(default 30 Hz). `latency_k` controls front trimming for background strategies;
`exp_weight_m` controls ACT ensembling decay. The `openpi_rtc` backend's
`backend_options.latency_k` is separate from the strategy's `args.latency_k`.

## Robots and action spaces

Robot registry names are `agilex_piper`, `arx_r5`, `arx_x5`, `dual_franka`,
`dual_yam`, `r1lite`, `ur5e`, and `agibot_g2`. Hardware presets define joint groups,
grippers, cameras, and EEF links. Joint vector order follows the robot's
`state_composition`.

`inference_cfg.obs_space` and `action_space` independently select `JointState`
or `EEFPose`. EEF layout is per arm `xyz + rotation [+ gripper]`, concatenated
across `n_arms`. Supported rotations are `quat`, `euler`, `rotvec`, and `rot6d`;
quaternions default to `wxyz`, Euler angles to `xyz` in radians. The default
quaternion layout with a gripper is 8 values per arm. Other representations or
`include_gripper=False` change that width.

EEF policy outputs are normalized to canonical quaternion poses and converted
through PyRoki IK. `robot.eef_links` selects the target link, so a TCP preset and
a flange preset can use different physical targets despite equal vector widths.
Use the matching preset for the model's training convention.

## Evaluation identity

An evaluation config adds `eval_cfg` with checkpoint choices, tasks, trial count,
and storage settings. EVAL can also load scene plans from task-set directories.
Scene-aware attempts bind `task_set`, `task_id`, and `scene_id` together and save
a scene-map snapshot with the trial. This keeps RESULT and dashboard views tied
to the attempted layout when the source plan later changes. See
[configuration](configuration.md#evaluation-and-scene-plans) and the
[scene-plan format](collection_scene_plan.md).
