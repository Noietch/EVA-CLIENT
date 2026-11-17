"""ARX X5 RL workspace with independent policy/critic selection."""

_base_ = ["../02_collection/arx_x5.py", "../00_base/rl.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    policies=[
        dict(
            name="arx_x5_openpi_qpos",
            config="../01_deploy/arx_x5/openpi_qpos.py",
            host="127.0.0.1",
            port=9000,
        ),
    ],
    critics=[
        dict(
            name="arx_x5_critic",
            type="websocket",
            host="127.0.0.1",
            port=9100,
            backend_options=dict(),
        ),
    ],
    data=dict(storage=dict(log_dir="work_dirs/rl/arx_x5")),
)
