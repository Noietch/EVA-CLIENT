[← Back to README](../README.md)

# Quick start

Run commands from the repository root using the existing application environment.
The `eva` entry point is implemented in [`src/main.py`](../src/main.py).

## Start with DEVICE

```bash
.venv/bin/eva --language en
# Chinese console:
# .venv/bin/eva --language zh
```

`--config` is optional. Without it, EVA loads the shared defaults and opens
**DEVICE**. Select a robot, teleoperation source, camera combination, and real
or fake hardware mode; apply the selection before starting the required device
components. Enable motion after inspecting their readiness.

Device choices and parameter overrides persist in
`~/.local/eva-client/hardware_config.yaml`; `EVA_WORKSTATION_PATH` selects another
file. A saved selection also applies when loading a business preset, so check
DEVICE when the active robot differs from the preset's name. Dataset transport
presets retain their configured robot and dataset settings.

The same device workflow is available through the CLI:

```bash
.venv/bin/eva device list
.venv/bin/eva device config
.venv/bin/eva device status
.venv/bin/eva device logs
```

To try a software robot, select ARX R5 with no cameras in DEVICE, choose fake
mode, apply, and start the robot component. The equivalent commands are:

```bash
.venv/bin/eva device select --robot arx_r5 --camera none
.venv/bin/eva device set robot.mode fake
.venv/bin/eva device start robot
```

Fake mode uses the shared ZMQ fake node and does not require the physical robot's
SDK environment. It provides synthetic observations; a real policy server is a
separate component. For local inference exercises, set `policy.type="mock"` in
a local workflow preset.
Stop the owned fake process when finished:

```bash
.venv/bin/eva device stop robot
```

Device processes are managed by a detached service. Closing the browser or
exiting EVA does not stop them; use DEVICE or `eva device stop` explicitly.

## Choose a workflow preset

```bash
# Joint-space deployment: DEBUG
.venv/bin/eva --config configs/01_deploy/dual_agilex_piper/openpi_qpos.py

# EEF-space deployment with IK to joint commands
.venv/bin/eva --config configs/01_deploy/ur5e/openpi_eef.py

# Teleoperation collection: COLLECT
.venv/bin/eva --config configs/02_collection/dual_yam.py

# Offline dataset transport with the bundled sample
.venv/bin/eva --config configs/00_openloop/dual_agilex_piper_openloop.py

# Multi-checkpoint evaluation: EVAL
.venv/bin/eva --config configs/03_evaluation/dual_agilex_piper_eval.py

# EVA-SIM task-set collection (requires a running compatible simulator)
.venv/bin/eva --config configs/02_collection/dual_yam_eva_sim.py

# DreamLabs deployment through the local policy bridge
.venv/bin/eva --config configs/01_deploy/dual_yam/dream.py
```

Inspect preset paths, endpoints, task-set mounts, and model/checkpoint settings
before use. A preset does not launch its policy server or simulator. The DreamLabs
bridge has a separate [setup guide](../examples/policy/dreamlabs/README.md);
SIM recording and scene plans are described in [Recording](recording.md) and
[Collection scene plans](collection_scene_plan.md).

## Bring up hardware manually

The robot execution node and EVA run in separate processes. If starting a node
manually, bring it up before connecting EVA, and avoid starting the same component
again through DEVICE. Use each robot's isolated SDK environment and launcher:

| Robot | Default transport | Launcher / instructions |
|-------|-------------------|-------------------------|
| AgileX Piper | `ros1` | [`run_agilex_ros.sh`](../examples/hardware/agilex_piper/run_agilex_ros.sh), [guide](../examples/hardware/agilex_piper/README.md) |
| Galaxea R1 Lite | `ros2` | [`run_hardware.sh`](../examples/hardware/r1_lite/run_hardware.sh), [guide](../examples/hardware/r1_lite/README.md) |
| ARX R5 | `zmq` | [`run_hardware.sh`](../examples/hardware/arx_r5/run_hardware.sh), [guide](../examples/hardware/arx_r5/README.md) |
| ARX X5 | `zmq` | [`run_hardware.sh`](../examples/hardware/arx_x5/run_hardware.sh), [guide](../examples/hardware/arx_x5/README.md) |
| Dual Franka | `zmq` | [`run_hardware.sh`](../examples/hardware/franka/run_hardware.sh), [guide](../examples/hardware/franka/README.md) |
| UR5e | `zmq` | [`run_hardware.sh`](../examples/hardware/ur5e/run_hardware.sh), [guide](../examples/hardware/ur5e/README.md) |
| AgiBot G2 | `zmq` | [`node.py`](../examples/hardware/agibot_g2/node.py), [guide](../examples/hardware/agibot_g2/README.md) |
| Dual YAM | `zmq` | [`run_hardware.sh`](../examples/hardware/yam/run_hardware.sh), [guide](../examples/hardware/yam/README.md) |

Fake mode uses `zmq` for all registered robots. Its cameras and dynamics are
synthetic and do not establish physical hardware readiness.

## Browser and remote access

Startup logs show the web address; the default is `http://127.0.0.1:8080` and
the server binds to `0.0.0.0`. To access a remote EVA console on a different
local port:

```bash
ssh -L 18080:127.0.0.1:8080 '<user>@<eva-host>'
# Open http://127.0.0.1:18080 on your workstation.
```

Policy traffic uses its own configured port. For example, if EVA is local and
the policy server listens on port 9000 on another host:

```bash
ssh -N -L 9000:127.0.0.1:9000 '<user>@<policy-host>'
```

Set `policy.host="127.0.0.1"` and `policy.port=9000` for that tunnel. Evaluation
can also manage SSH forwarding through `eval_cfg`.

## Headless and dataset tools

```bash
.venv/bin/eva --config configs/01_deploy/arx_r5/openpi_qpos.py --headless
```

Headless mode disables the HTTP console and browser visualization. On a TTY,
the inference CLI provides `task`, `setup`, `run`, `stop`, `reset`, and `status`.
Headless mode enables the [ZMQ control channel](control-channel.md) automatically;
its configured host and port also serve external automation. With the browser
enabled, set `control_channel.enabled=True` to expose the channel.

The standalone dataset browser does not need a running robot or EVA console:

```bash
bash tools/datasets/start.sh --config configs/dataset_browser.py --read-only
```

It defaults to `http://127.0.0.1:8412` and requires explicitly configured data sources in
[`configs/dataset_browser.py`](../configs/dataset_browser.py). Read-only mode disables editing and transfer actions;
the service can still create catalog directories and preview caches. See
[Recording](recording.md) for the data layout and QC/export workflow.

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Business `.py` preset | Shared defaults, DEVICE |
| `--web-port` | HTTP console port | `8080` |
| `--headless` | Disable the web console | Off |
| `--language`, `--locale` | Console language, `en` or `zh` | `en` |

Device subcommands have their own `--workspace` and `--url` options, placed
before the action: `eva device --url http://127.0.0.1:8081 status`.
