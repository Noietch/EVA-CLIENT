<p align="center">
  <img src="assets/eva-logo.svg" alt="EVA-Client Logo" width="34%">
</p>

<h1 align="center">EVA-Client: A Unified Framework for Deployment, Evaluation, and Data Collection on Real Robots</h1>

<p align="center">One policy, any robot — the smooth all-in-one real-robot stack. Debug, record, evaluate, visualize, all in the browser.</p>

<p align="center">
<a href="https://colalab.net/projects/eva-client/"><img src="https://img.shields.io/badge/Project%20Page-colalab.net-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
<a href="https://colalab.net/projects/eva-client/paper/EVA_Client_Report.pdf"><img src="https://img.shields.io/badge/Technical%20Report-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Technical Report"></a>
<a href="https://colalab.net/projects/eva-client/docs/introduction.html"><img src="https://img.shields.io/badge/Docs-English-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation (English)"></a>
<a href="https://colalab.net/projects/eva-client/docs/introduction.zh.html"><img src="https://img.shields.io/badge/文档-中文-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation (中文)"></a>
<a href="https://github.com/Noietch/EVA-CLIENT/stargazers"><img src="https://img.shields.io/github/stars/Noietch/EVA-CLIENT?style=for-the-badge&logo=github&logoColor=white&color=0a0a0a&cacheSeconds=60" alt="GitHub Stars"></a>
<p align="center">
  <video src="https://github.com/user-attachments/assets/09cf8c98-396d-45d0-bd38-8603412ec3c2" controls muted></video>
</p>

<p align="center"><em>EVA-Client driving an AgileX bimanual arm end-to-end from the browser — teleop → record → π₀ checkpoint → smooth async deploy. Real hardware, not a rendering.</em></p>

<p align="center">
  <b>Jump to:</b>&nbsp;
  <a href="#-install">Install</a> ·
  <a href="#-quick-start">Quick start</a> ·
  <a href="#-web-console">Web console</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-core-concepts">Concepts</a> ·
  <a href="#%EF%B8%8F-configuration">Config</a> ·
  <a href="#-recording">Recording</a> ·
  <a href="#-development">Dev</a> ·
  <a href="#-citation">Cite</a>
</p>

---

## ✨ What you get

* **🚀 Deployment.** One command brings up a real-robot closed loop:
  `.py` config → transport ([ROS1](https://github.com/ros/ros) / [ROS2](https://github.com/ros2/ros2) / [ZeroMQ](https://github.com/zeromq/pyzmq) / offline dataset) → policy
  backend ([OpenPI](https://github.com/Physical-Intelligence/openpi), [OpenPI-RTC](https://www.pi.website/research/real_time_chunking), [StarVLA](https://github.com/starVLA/starVLA), [GR00T](https://github.com/Nvidia/Isaac-GR00T), mock, replay) → inference
  strategy (sync / [async](https://github.com/OpenDriveLab/kai0#train-deploy-alignment) / naive / [ACT-ensemble](https://github.com/tonyzhaozh/act) / RTC) with live latency
  compensation. **6 robots already** — joint-space or EEF-space (PyRoki IK),
  all live-switchable from the DEBUG tab.
* **📊 Evaluation.** Multi-checkpoint sweeps with per-trial records: every
  rollout captures camera video, 3D URDF scene, per-dimension state charts,
  and per-prompt milestone scores into dataset metadata, then replays
  synchronously in the RESULT tab. Prompt shuffling, and remote
  policy servers over SSH port-forward are first-class.
* **🎥 Data collection.** Teleop capture straight into [LeRobot](https://github.com/huggingface/lerobot) v2.1 episodes
  from the COLLECT tab — background saver, in-tab QC PASS/FAIL replay, camera
  streams encoded to mp4, per-frame green/red quality flags. Teleop demos and
  model rollouts share one on-disk layout.

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/workflow.png" width="100%" alt="EVA-Client workflow">
</p>

---

## 📦 Install

Requirements:

- Python ≥ 3.10 (any 3.10+ works; `uv` will fetch on demand)
- [uv](https://github.com/astral-sh/uv) for dependency management
- ROS Noetic — **only** for the `ros1` transport; `ros2` / `zmq` / `dataset`
  need no ROS

```bash
git clone https://github.com/Noietch/EVA-CLIENT.git
cd EVA-CLIENT
curl -LsSf https://astral.sh/uv/install.sh | sh

# Standard install (non-ROS backends)
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

<details>
<summary><b>ROS1 install</b> — expose the system ROS install to the venv</summary>

```bash
uv venv --python 3.10 --system-site-packages .venv
```

</details>

Optional hardware extras:

| Extra    | Use it for                                                                    |
|----------|-------------------------------------------------------------------------------|
| `dev`    | test / lint / type-check tooling — always recommended                         |
| `franka` | Franka arms (`dual_franka`)                                                   |
| `arx`    | ARX R5 arm (`arx_r5`); vendored driver in-tree                                |
| `ur5e`   | UR5e arm with DH gripper; vendored driver in-tree                             |
| `camera` | Orbbec depth cameras (Linux x86_64/aarch64 + macOS arm64 wheels; Python 3.11) |
| `viz`    | Faster 3D robot-model loading in the web view                                 |

Extras stack: `uv pip install -e ".[dev,ur5e]"`.

Verify the install with no ROS and no policy server:

```bash
eva --config configs/00_openloop/dual_agilex_piper_openloop.py
```

---

## 🚀 Quick start

The entry point is `eva` (`src/main.py`). `--config` is required and points to a
`.py` config; everything else is operated from the browser.

**Two processes on real hardware.** EVA is the browser-side client; the robot
SDK runs in a separate execution-layer node under `examples/hardware/<robot>/`.
Start the hardware node first, then start `eva` — order matters because the
`ros1` / `ros2` / `zmq` transport connects on the EVA side.

```bash
# Terminal 1 — bring up the hardware node (per robot; see examples/hardware/<name>/README.md)
bash examples/hardware/agilex_piper/run_agilex_ros.sh    # ROS1: AgileX Piper
bash examples/hardware/r1_lite/run_hardware.sh           # ROS2: Galaxea R1-Lite
bash examples/hardware/arx/run_hardware.sh               # ZMQ:  ARX R5
bash examples/hardware/franka/run_hardware.sh            # ZMQ:  Dual Franka
bash examples/hardware/ur5e/run_hardware.sh              # ZMQ:  UR5e
bash examples/hardware/agibot_g2/node.py                 # ZMQ:  AgiBot G2

# No robot handy? Every folder ships a software-only fake node over ZMQ:
bash examples/hardware/agilex_piper/run_fake_node.sh
```

```bash
# Terminal 2 — EVA
eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py
```

The four supported transports are:

| Robot           | Transport | Hardware launcher                              |
|-----------------|-----------|------------------------------------------------|
| AgileX Piper    | `ros1`    | `examples/hardware/agilex_piper/run_agilex_ros.sh` |
| Galaxea R1-Lite | `ros2`    | `examples/hardware/r1_lite/run_hardware.sh`    |
| ARX R5          | `zmq`     | `examples/hardware/arx/run_hardware.sh`        |
| Dual Franka     | `zmq`     | `examples/hardware/franka/run_hardware.sh`     |
| UR5e            | `zmq`     | `examples/hardware/ur5e/run_hardware.sh`       |
| AgiBot G2       | `zmq`     | `examples/hardware/agibot_g2/node.py`          |

Then, from the EVA side, pick a preset:

```bash
# Deploy — opens on DEBUG
eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py

# Deploy with EEF-space policy (EVA runs PyRoki IK to joint commands)
eva --config configs/01_deploy/ur5e/openpi_eef.py

# Offline replay — no policy server, no live robot, no hardware node
eva --config configs/00_openloop/dual_agilex_piper_openloop.py

# Multi-checkpoint evaluation — opens on EVAL
eva --config configs/03_evaluation/dual_agilex_piper_eval.py
```

Startup logs print the web URL (default `http://127.0.0.1:8080`); open it in a
browser. The server binds `0.0.0.0`.

<details>
<summary><b>Remote policy server?</b> Forward the port over SSH</summary>

```bash
ssh -L 8080:localhost:8080 <user>@<host>   # then open http://localhost:8080
```

</details>

| Argument     | Description                        | Default |
|--------------|------------------------------------|---------|
| `--config`   | path to a `.py` config (required)  | —       |
| `--web-port` | web server port                    | `8080`  |

---

## 🧭 Web console

A single console app with six tabs, organized into three flows. Tab
availability is config-driven — a deploy config lands on **DEBUG**, an eval
config (`eval_cfg.checkpoints[]`) lands on **EVAL**; unrelated tabs are greyed
out or read-only.

**Collection flow**

| Tab         | Purpose                                                                                                            |
|-------------|--------------------------------------------------------------------------------------------------------------------|
| **MANUAL**  | hand-drive a real robot with per-joint qpos sliders (no policy in loop); stage target pose then `SEND` or `HOME` — for hardware bring-up |
| **COLLECT** | teleop recording into a LeRobot v2.1 episode; background saver + in-tab QC PASS/FAIL replay                        |
| **REPLAY**  | open-loop playback of a recorded episode via the `dataset` transport, same live view as DEBUG                      |

**Deployment flow**

| Tab       | Purpose                                                                                                            |
|-----------|--------------------------------------------------------------------------------------------------------------------|
| **DEBUG** | default tab — prompt → config → setup → control; runs one closed-loop policy run in `REAL` / `SIM` / `STEP` (single-step breakpoint) modes with live 3D arm + camera + action/state charts |

**Evaluation flow**

| Tab        | Purpose                                                                                                            |
|------------|--------------------------------------------------------------------------------------------------------------------|
| **EVAL**   | multi-checkpoint sweep — each checkpoint shown as Model A/B/… + its real name, per-prompt trials with milestone scoring, optional shuffling and SSH port-forward to a remote policy server |
| **RESULT** | read-only browser over eval recordings: `model → task → trial` tree with score/success gauges, synced camera + 3D URDF + per-dimension state chart replay |

---

## 📚 Core concepts

**Transports** — EVA speaks only the middleware protocol; the SDK that drives
the hardware runs in a separate execution-layer node (see `examples/hardware/`).
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

---

## ⚙️ Configuration

Configs are `.py` files using mmengine-style `_base_` inheritance with
field-wise recursive deep merge — a preset overrides only what it changes,
`_delete_=True` drops an inherited subtree. `configs/00_base/defaults.py` is
the single source of truth for omitted fields; scalars and whole lists
**replace** the parent value (lists are never concatenated). Deploy presets
inherit a sibling `_base.py`, which in turn inherits `../../00_base/defaults.py`.

```python
# configs/01_deploy/dual_agilex_piper/openpi_qpos.py
_base_ = ['_base.py']                          # → sibling _base.py → ../../00_base/defaults.py

policy = dict(
    type='openpi_rtc',                         # openpi / openpi_rtc / starvla / gr00t / mock / replay
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    debug_tasks=['pick up the yellow spoon and place it on the green plate'],
)

# deep merge: override only `args` for the inherited 'async' / 'rtc' strategies
inference_strategies = {
    'async': dict(args=dict(latency_k=4)),
    'rtc':   dict(args=dict(latency_k=4)),
}
```

Startup pipeline: merge the `_base_` chain → build control spaces (joint / EEF)
→ fill derived paths from `work_dir` → validate recording configs → resolve
eval checkpoints (each with its own config, host, port). Failures print a
Python traceback naming the section/file at fault.

---

## 🎞️ Recording

Every run — teleop capture from **COLLECT** or model rollout from **DEBUG** /
**EVAL** — is written to the same LeRobot v2.1 dataset. The on-disk layout is:

```
<log_dir>/<task name>/
├── data/chunk-000/episode_000000.parquet     # 1 row per step, SFT-ready
├── videos/chunk-000/<camera>/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl                         # per-episode length, task, quality flag
    ├── episodes_stats.jsonl
    ├── tasks.jsonl
    └── stats.json                             # per-feature min/max/mean/std
```

Inference runs additionally carry a debug long-table sidecar (multiple raw
predictions per step for overlapping async / RTC chunks), joined to the main
table on `absolute_step`. Cameras are encoded as streaming mp4. Per-frame QC
marks each episode `green` or `red` with a reason (`episode_too_short`,
`non_monotonic_timestamp`, `missing_camera`, …); red episodes are still saved,
issues recorded in `episodes.jsonl`. Eval trials save on STOP; milestone scores
live in dataset metadata. Re-running the same eval prompt/trial overwrites that
episode in place — teleop always appends. A sample dataset lives at
`examples/agilex_dataset/`.

---

## 🔬 Development

```bash
uv pip install -e ".[dev]"       # pulls pytest, ruff, pyright
pytest                           # tests
pytest -m "not slow"             # skip slow IK tests
ruff check . && ruff format .    # lint + format
pyright                          # type check
pre-commit install               # optional git hooks
```

Tests are grouped under `tests/config/`, `tests/comms/`, `tests/ik/` (slow),
`tests/strategy/`, `tests/transport/`, `tests/recorder/`, and
`tests/integration/web/`. No ROS install or hardware required — the robot side
is faked. Manual verification scripts under `tests/manual/` (run as
`PYTHONPATH=src python tests/manual/<script>.py`) exercise end-to-end eval
flow, rollout, LeRobot meta, and viser chunk tracking.

---

## 🗺️ Roadmap

- [ ] **More robots.** Extend the robot zoo to more embodiments — dual-arm
      manipulators (YAM, Tianji, …), humanoids
      (Unitree H1/G1, Fourier GR-1, Booster T1, …) and mobile / wheeled
      platforms (mobile ALOHA, Galaxea R1 base, quadruped + arm).
- [ ] **Human-in-the-loop data collection for RL.** Interventions during
      policy rollout captured as preference / correction data, DAgger-style
      relabeling, and reward-model signals piped back through the LeRobot
      episode format for online RL fine-tuning.
- [ ] **Embodied agent.** Wrap the deployment + evaluation loop with a
      language-driven planner (VLM / VLA + tool use) so long-horizon tasks
      can be decomposed, executed, verified, and re-planned end-to-end from
      the same console.
- [ ] **Data annotation.** Extend Collect mode with fine-grained task and
      sub-task annotation, segmenting long-horizon episodes into labeled
      sub-task units and milestones within the same LeRobot dataset. This
      makes a collection reusable at the level of individual manipulation
      phases.

---

## 📝 Citation

If EVA-Client is useful for your research or product, please cite:

```bibtex
@misc{evaclient2026,
  title        = {EVA-Client: A Unified Framework for Deployment, Evaluation,
                  and Data Collection on Real Robots},
  author       = {EVA-Client Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/Noietch/EVA-CLIENT}},
}
```

---

<details>
<summary><b>License &amp; Acknowledgements</b></summary>

This project is licensed under the Apache-2.0 License. See [LICENSE](./LICENSE)
for more information.

Builds upon several excellent open-source efforts, including
[PyRoki](https://github.com/chungmin99/pyroki) and
[jaxls](https://github.com/brentyi/jaxls) for kinematics,
[OpenPI](https://github.com/Physical-Intelligence/openpi) for policy serving,
the [LeRobot](https://github.com/huggingface/lerobot) dataset format, and a
vendored mmengine-style config system from
[MMEngine](https://github.com/open-mmlab/mmengine).

</details>
