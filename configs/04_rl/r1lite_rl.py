"""R1 Lite RL workspace with independent policy and critic selection."""

_base_ = ["../01_deploy/r1lite/openpi_qpos.py", "../00_base/rl.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    policies=[
        dict(
            name="r1lite_openpi_qpos",
            config="../01_deploy/r1lite/openpi_qpos.py",
            host="127.0.0.1",
            port=9000,
        ),
    ],
    critics=[
        dict(
            name="r1lite_critic",
            type="websocket",
            host="127.0.0.1",
            port=9100,
            backend_options=dict(),
        ),
    ],
    data=dict(
        storage=dict(log_dir="work_dirs/rl/r1lite", fps=15, image_height=360, image_width=640)
    ),
)
