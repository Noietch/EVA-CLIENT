"""UR5e RL workspace with independent policy and critic selection."""

_base_ = ["../01_deploy/ur5e/openpi_qpos.py"]

console = dict(
    initial_tab="rl",
)

collection = dict(
    teleop=dict(
        _delete_=True,
        control_source="client",
        safety=dict(
            max_qpos_step=0.08,
            max_position_error_m=0.08,
            max_orientation_error_rad=0.35,
        ),
        client=dict(
            type="vr_webxr",
            endpoint="tcp://127.0.0.1:8765",
            ack_endpoint="tcp://127.0.0.1:8766",
            input_timeout_s=0.25,
            heartbeat_timeout_s=2.0,
            position_scale=1.0,
            base_from_xr_rotation=[
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            gripper=dict(
                mode="linear",
                threshold=0.6,
                open_value=1.0,
                close_value=0.0,
            ),
            arms=dict(
                arm=dict(controller="right"),
            ),
        ),
    ),
)

rl_cfg = dict(
    cli_mode="real",
    inference_strategy="async",
    tasks=["placeholder task — replace with the real RL task prompt"],
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
            backend_options={},
        ),
    ],
    data=dict(
        format="lerobot",
        storage=dict(
            log_dir="work_dirs/rl/ur5e",
            fps=20,
            save_queue_max=15,
            async_save=True,
        ),
    ),
    intervention=dict(control_mode="relative", source="teleop_client"),
)
