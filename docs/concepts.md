[← Back to README](../README.md)

# 📚 Core concepts

**Transports** — EVA speaks only the middleware protocol; the SDK that drives
the hardware runs in a separate execution-layer node (see `../examples/hardware/`).
Every backend obeys the same contract: pull one observation, send a joint
command (EEF poses are converted upstream via IK), report link health.

| `transport.type` | Use when                                                          | Prerequisite      |
|------------------|-------------------------------------------------------------------|-------------------|
| `ros1`           | real robot on a ROS 1 stack                                       | ROS        |
| `ros2`           | real robot on a ROS 2 stack; decodes `/compressed` camera streams | ROS 2             |
| `zmq`            | real robot fronted by a ZeroMQ execution node                     | — (in-tree)       |
| `dataset`        | offline LeRobot v2.x (v2.1 layout) replay                         | — (fully offline) |

If a ROS backend is selected without ROS installed, EVA prints a warning and
runs a silent no-op link so the web console still opens.

**Policy backends**

| `policy.type` | Protocol                       | Server? | Notes                                                                                        |
|---------------|--------------------------------|---------|----------------------------------------------------------------------------------------------|
| `openpi`      | WebSocket + msgpack            | yes     | OpenPI-compatible server, stateless                                                          |
| `openpi_rtc`  | WebSocket + msgpack            | yes     | Real-Time Chunking variant; feeds `prev_action` back for alignment (`latency_k`, start at 4) |
| `starvla`     | WebSocket + msgpack            | yes     | typed envelope, configurable `camera_key` / `unnorm_key`                                     |
| `gr00t`       | ZeroMQ REQ/REP + msgpack-numpy | yes     | Isaac-GR00T; payloads keyed by modality (`video_keys`, `state_key`, `language_key`, …)       |
| `mock`        | local                          | no      | smooth random actions, offline integration testing                                           |
| `replay`      | local                          | no      | replays a dataset's recorded action trajectory                                               |

`openpi_rtc` (backend, wire protocol) pairs with but is separate from the `rtc`
inference strategy (chunk scheduling); each has its own `latency_k`.

**Inference strategies** (preset in config, switchable live in DEBUG)

| Key       | Behavior                                                                                                                              | Choose when                                            |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `sync`    | blocking: fetch chunk, execute first few steps, fetch again; no background thread, no smoothing — robot pauses on each prediction     | simplest, most predictable; brief pauses acceptable    |
| `async`   | background prediction, linear crossfade over the overlap between old and new chunks — **default**                                     | continuous smooth motion — the usual choice            |
| `naive`   | background prediction, each new chunk fully replaces the buffer                                                                       | freshest prediction immediately; small jumps tolerable |
| `act`     | background prediction, ACT-style temporal ensembling averages overlapping chunks per step (`exp_weight_m`)                            | smoothest, most averaged motion; noisy model output    |
| `rtc`     | background scheduler with Real-Time Chunking: new chunk aligned to action already in flight for seamless handovers at high rates      | fast policy server; latency-aware handovers required   |

Tunable knobs: `execute_horizon` (all), `inference_rate` (default 3 Hz for
async/naive/act/rtc), `latency_k` (async/naive/rtc — front-trims new chunks),
`exp_weight_m` (act only).

**Robots** — `agilex_piper`, `arx_r5`, `dual_franka`, `r1lite`, `ur5e`,
`agibot_g2`. All kinematics go through one PyRoki (JAX + jaxls) backend:
per-frame Levenberg–Marquardt IK chained for continuity (a velocity cost ties
each frame to the previous solve; a rest cost biases toward home to fight
null-space drift). The first frame anchors to the measured current state, so
execution starts without a jump.

**Action spaces** — `inference_cfg.obs_space` and `inference_cfg.action_space`
are each `JointState` or `EEFPose`. When the policy outputs `EEFPose` but the
robot consumes joints, EVA runs PyRoki IK to convert. An `EEFPose` vector is
laid out as `n_arms × (xyz3 + quat4 + grip1)` (8D per arm).
