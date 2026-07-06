[← Back to README](../README.md)

# 📦 Install

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
