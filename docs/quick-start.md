[← Back to README](../README.md)

# 🚀 Quick start

The entry point is `eva` (`../src/main.py`). `--config` is required and points to a
`.py` config; everything else is operated from the browser.

**Two processes on real hardware.** EVA is the browser-side client; the robot
SDK runs in a separate execution-layer node under `../examples/hardware/<robot>/`.
Start the hardware node first, then start `eva` — order matters because the
`ros1` / `ros2` / `zmq` transport connects on the EVA side.

```bash
# Terminal 1 — bring up the hardware node (per robot; see examples/hardware/<name>/README.md)
bash examples/hardware/agilex_piper/run_agilex_ros.sh    # ROS1: AgileX Piper
bash examples/hardware/r1_lite/run_hardware.sh           # ROS2: Galaxea R1-Lite
bash examples/hardware/arx/run_hardware.sh               # ZMQ:  ARX R5
bash examples/hardware/franka/run_hardware.sh            # ZMQ:  Dual Franka
bash examples/hardware/ur5e/run_hardware.sh              # ZMQ:  UR5e
I2RT_ROBOT=i2rt_dual_yam bash examples/hardware/i2rt/run_hardware.sh  # ZMQ: I2RT YAM
python examples/hardware/agibot_g2/node.py               # ZMQ:  AgiBot G2

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
| I2RT YAM / dual YAM | `zmq`  | `examples/hardware/i2rt/run_hardware.sh`       |
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
