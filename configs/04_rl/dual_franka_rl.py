"""Dual Franka RL workspace with independent policy and critic selection."""

_base_ = ["../02_collection/dual_franka.py", "../00_base/rl.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    policies=[
        dict(
            name="dual_franka_openpi_qpos",
            config="../01_deploy/dual_franka/openpi_qpos.py",
            host="127.0.0.1",
            port=9000,
        ),
    ],
    critics=[
        dict(
            name="dual_franka_critic",
            type="websocket",
            host="127.0.0.1",
            port=9100,
            backend_options=dict(),
        ),
    ],
    data=dict(storage=dict(log_dir="work_dirs/rl/dual_franka")),
)
