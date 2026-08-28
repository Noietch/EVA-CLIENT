"""ARX X5 RL workspace with VR intervention and independent policy/critic selection."""

_base_ = ["../02_collection/arx_x5_vr.py"]

console = dict(
    initial_tab="rl",
)

rl_cfg = dict(
    cli_mode="real",
    inference_strategy="async",
    tasks=["placeholder task — replace with the real RL task prompt"],
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
            backend_options={},
        ),
    ],
    data=dict(
        format="lerobot",
        storage=dict(
            log_dir="work_dirs/rl/arx_x5",
            fps=30,
            save_queue_max=15,
            async_save=True,
        ),
    ),
    intervention=dict(control_mode="relative", source="teleop_client"),
)
