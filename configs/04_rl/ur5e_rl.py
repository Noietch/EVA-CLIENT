"""UR5e RL workspace with independent policy and critic selection."""

_base_ = ["../01_deploy/ur5e/openpi_qpos.py", "../00_base/rl.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    policies=[
        dict(
            name="ur5e_openpi_qpos",
            config="../01_deploy/ur5e/openpi_qpos.py",
            host="127.0.0.1",
            port=9000,
        ),
    ],
    critics=[
        dict(
            name="ur5e_critic",
            type="websocket",
            host="127.0.0.1",
            port=9100,
            backend_options=dict(),
        ),
    ],
    data=dict(storage=dict(log_dir="work_dirs/rl/ur5e", fps=20)),
)
