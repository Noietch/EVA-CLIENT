"""Dual Piper collection driven by the EVA Client WebXR teleoperation node."""

_base_ = ["dual_agilex_piper.py"]

collection = dict(
    teleop=dict(
        _delete_=True,
        control_source="client",
        safety=dict(
            max_qpos_step=0.03,
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
            squeeze_threshold=0.5,
            clutch_mode="toggle",
            base_from_xr_rotation=[
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            gripper=dict(
                mode="toggle",
                threshold=0.5,
                open_value=0.1,
                close_value=0.0,
            ),
            arms=dict(
                left_arm=dict(
                    controller="left",
                    workspace=dict(min=[0.0, -0.55, -0.10], max=[0.75, 0.55, 0.65]),
                ),
                right_arm=dict(
                    controller="right",
                    workspace=dict(min=[0.0, -0.55, -0.10], max=[0.75, 0.55, 0.65]),
                ),
            ),
        ),
    ),
)
