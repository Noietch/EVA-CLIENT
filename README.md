# EVA Client

> v0.1 — Unified robot VLA inference debugging and evaluation client

EVA connects to a VLA (Vision-Language-Action) policy server, streams live
observations, and dispatches actions in real time — covering the full loop of
**single-machine debugging → data collection → blind evaluation → result
visualization**. Everything is driven from a browser-based **web console** (control
commands flow through an internal HTTP command queue), including a self-rendered 3D
scene (yourdfpy parses URDF, Three.js loads geometry once and streams per-frame 4×4
world transforms).

It supports real / simulation / offline runtimes, 6 robots, 4 transports, 6 policy
backends, and 5 inference strategies.

## Architecture

```
eva-client/
├── configs/              # .py configs with mmengine-style `_base_` inheritance + deep merge
│   ├── 00_base/          #   defaults.py — single source of truth for omitted fields
│   ├── 00_openloop/      #   dataset open-loop (no policy server), one per robot
│   ├── 01_deploy/        #   per-robot deployment (qpos / eef presets)
│   ├── 02_collection/    #   per-robot teleop data-collection configs
│   └── 03_evaluation/    #   blind eval (eval_cfg.checkpoints[] + optional ssh forward)
├── examples/             # execution-layer nodes, teleop publishers, sample dataset
├── tests/                # unit tests (IK / config merge / action conversion / smoothing)
└── src/
    ├── main.py           # entry: parse CLI, launch the web app
    ├── core/
    │   ├── cfg/          #   .py config loading (`_base_`) + type registry
    │   ├── app/          #   main loop, state, command handlers, console server
    │   ├── recorder/     #   LeRobot v2.1 episode + collection recording
    │   └── utils/        #   math, paths, lerobot meta, ssh forwarding
    ├── policy_client/    # backends: openpi / openpi_rtc / starvla / gr00t / mock / replay
    ├── robots/           # robot zoo (6 robots) + PyRoki (JAX/jaxls) kinematics
    ├── strategy/         # inference strategies (sync / async / naive / act / rtc)
    └── transport/        # transports (ros1 / ros2 / zmq / dataset)
```

## Requirements

- Python ≥ 3.10, < 3.12 for the EVA client environment
- ROS Noetic — only for the `ros1` transport
- ROS 2 — only for the `ros2` transport. Its Python packages (`rclpy`, `cv_bridge`,
  message packages) must be built for the same Python minor version as `.venv`; use
  Python 3.10 for Ubuntu 22.04 / ROS 2 Humble apt packages.
- `zmq` / `dataset` transports have no ROS dependency
- [uv](https://github.com/astral-sh/uv) for dependency management

## Installation

```bash
source /opt/ros/humble/setup.bash  # required before running the ros2 transport
uv venv --python /usr/bin/python3.10 --system-site-packages .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras: `franka`, `arx`, `ur5e`, `camera` (Orbbec SDK), `viz`.

## Quick start

The entry point is `eva` (`src/main.py`). `--config` is required and must point to a
`.py` config; operate everything from the browser.

```bash
# deploy: openpi policy, joint-space action
eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py

# deploy: EEF-pose action (policy outputs EEF, PyRoki IK converts to joint commands)
eva --config configs/01_deploy/dual_agilex_piper/openpi_eef.py

# offline: dataset open-loop replay (no ROS or policy server)
eva --config configs/00_openloop/dual_agilex_piper_openloop.py

# blind multi-model evaluation
eva --config configs/03_evaluation/dual_agilex_piper_eval.py
```

Startup logs print the web port (default `8080`); open `http://<host>:8080`. For
remote work, forward the port over SSH:

```bash
ssh -L 8080:localhost:8080 <user>@<host>   # then open http://localhost:8080
```

| Argument | Description | Default |
|---|---|---|
| `--config` | path to a `.py` config (required) | — |
| `--web-port` | web server port | `8080` |

## Web console

A single console app with six tabs. A config carrying an `eval_cfg.checkpoints[]`
block auto-activates the EVAL/RESULT tabs on startup; otherwise they are greyed out.

| Tab | Purpose |
|---|---|
| **DEBUG** | run control (connect, setup, run, stop, reset), live camera + qpos feed, gripper control, breakpoint single-step: `SIM ↻` previews one chunk in 3D → `REAL ▶` dispatches after review |
| **REPLAY** | open-loop replay of a recorded dataset |
| **COLLECT** | teleoperation recording into a LeRobot v2.1 dataset |
| **MANUAL** | drag per-joint qpos sliders and dispatch (`SEND` to stage, `SYNC` to mirror the real robot) |
| **EVAL** | blind multi-model eval — anonymized model labels (A/B/…) + reproducible shuffle, flat prompt list, per-prompt trials with milestone scoring, optional SSH port-forward + result rsync |
| **RESULT** | per-trial replay synced across three views: camera mp4 (HTTP Range seek), 3D URDF (backend FK over recorded qpos), and a per-dimension state chart |

## Core concepts

**Transports** — EVA speaks only the middleware protocol; the SDK that drives the
hardware runs in a separate execution-layer node (see `examples/hardware/`).

| Type | Description | Dependency |
|---|---|---|
| `ros1` | ROS Noetic bridge (joint / EEF / camera topics) | ROS Noetic |
| `ros2` | ROS 2 bridge | ROS 2 |
| `zmq` | ZeroMQ pub/sub to the execution-layer node | — |
| `dataset` | offline LeRobot v2 dataset replay | — |

**Policy backends**

| Backend | Protocol | Notes |
|---|---|---|
| `openpi` / `openpi_rtc` | WebSocket + msgpack | OpenPI server; `_rtc` variant feeds prev_action back |
| `starvla` | WebSocket + msgpack | typed envelope, configurable camera / unnorm key |
| `gr00t` | ZeroMQ REQ/REP + msgpack-numpy | Isaac-GR00T, nested by modality key |
| `mock` | local | smooth random actions, offline integration testing |
| `replay` | local | replays a dataset's recorded action trajectory |

**Inference strategies** (preset in config, switchable from the console)

| Type | Description |
|---|---|
| `BaseInferStrategy` | synchronous — execute one chunk per response |
| `AsyncLinearOverlapInferStrategy` | background fixed-rate thread, linear-overlap smoothing (`latency_k` clips the first k steps for latency) |
| `NaiveAsyncInferStrategy` | background thread, switches to the new chunk with no smoothing |
| `ActEnsembleInferStrategy` | background thread, ACT-style temporal aggregation (`exp_weight_m`) |
| `RtcInferStrategy` | Real-Time Chunking — prev_action-guided diffusion alignment + latency-compensated overlap |

**Robots** — `agilex_piper`, `arx_r5`, `dual_franka`, `r1_lite`, `ur5e`, `agibot_g2`.
All kinematics go through one PyRoki (JAX + jaxls) backend: per-frame Levenberg–
Marquardt IK chained for continuity (velocity cost ties each frame to the previous
solve, rest cost biases toward the home pose to fight null-space drift). The first
frame anchors to the measured current state, so execution starts without a jump.

**Action spaces** — `inference_cfg.obs_space` and `inference_cfg.action_space` are
each `JointState` or `EEFPose`. When the policy outputs `EEFPose` but the robot
consumes joints, EVA runs PyRoki IK to convert. An `EEFPose` vector is laid out as
`n_arms × (xyz3 + rot + grip1)` (`rotation: quat` → 4D).

## Configuration

Configs are `.py` files using mmengine-style `_base_` inheritance with field-wise
deep merge — a preset overrides only what it changes. `configs/00_base/defaults.py`
is the single source of truth for omitted fields.

```python
# configs/01_deploy/dual_agilex_piper/openpi_qpos.py
_base_ = ['_base.py']                          # → ../../00_base/defaults.py

policy = dict(
    type='openpi_rtc',                         # openpi / openpi_rtc / starvla / gr00t / mock / replay
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    obs_space=dict(type='JointState'),
    action_space=dict(type='JointState'),
    publish_rate=30,
    debug_tasks=['pick up the yellow spoon and place it on the green plate'],
)

inference_strategies = {
    'sync':  dict(type='BaseInferStrategy', args=dict(execute_horizon=5)),
    'async': dict(type='AsyncLinearOverlapInferStrategy', args=dict(latency_k=4)),
}
```

## Data collection

With logging enabled, each inference run is recorded as one LeRobot v2.1 episode
(one config → one dataset), using a two-table layout: a main table (strictly 1 row
per step, SFT-ready) plus a debug long-table sidecar (multiple raw predictions per
step for overlapping async/RTC chunks), joined on `absolute_step`. Cameras are
encoded as streaming mp4. The COLLECT tab records teleoperation via transport-level
`CollectionFrame` snapshots; QC marks each episode green/red (red ones are still
saved, with issues recorded in `quality_issues`).

## Development

```bash
pytest            # tests
ruff check .      # lint
pyright           # type check
```
