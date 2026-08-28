"""Dual Franka collection driven by the EVA Client WebXR teleoperation node."""

_base_ = ["dual_franka.py"]

console = dict(initial_tab="collect")

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
                left_arm=dict(controller="left"),
                right_arm=dict(controller="right"),
            ),
        ),
    ),
)
