"""ARX R5 RL workspace with independent policy and critic selection."""

_base_ = ["../01_deploy/arx_r5/openpi_qpos.py", "../00_base/rl.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    policies=[
        dict(
            name="arx_r5_openpi_qpos",
            config="../01_deploy/arx_r5/openpi_qpos.py",
            host="127.0.0.1",
            port=9000,
        ),
    ],
    critics=[
        dict(
            name="arx_r5_critic",
            type="websocket",
            host="127.0.0.1",
            port=9100,
            backend_options=dict(),
        ),
    ],
    data=dict(storage=dict(log_dir="work_dirs/rl/arx_r5")),
)
