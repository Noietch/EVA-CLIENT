"""AgiBot G2 RL workspace with independent policy and critic selection."""

_base_ = ["../02_collection/agibot_g2.py", "../00_base/rl.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    policies=[
        dict(
            name="agibot_g2_openpi_qpos",
            config="../01_deploy/agibot_g2/openpi_qpos.py",
            host="127.0.0.1",
            port=9000,
        ),
    ],
    critics=[
        dict(
            name="agibot_g2_critic",
            type="websocket",
            host="127.0.0.1",
            port=9100,
            backend_options=dict(),
        ),
    ],
    data=dict(storage=dict(log_dir="work_dirs/rl/agibot_g2")),
)
