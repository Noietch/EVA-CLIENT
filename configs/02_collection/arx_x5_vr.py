"""Dual ARX X5 real-hardware collection driven by WebXR teleoperation."""

_base_ = ["../01_deploy/arx_x5/_base.py"]

console = dict(
    initial_tab="collect",
)

collection = dict(
    storage=dict(
        log_dir="work_dirs/collection/arx_x5_vr",
        image_skew_tolerance_sec=0.035,
    ),
    schema=dict(
        robot_type="arx_x5",
        arms=dict(
            left_arm="left",
            right_arm="right",
        ),
        cameras=dict(
            cam_high="observation.images.cam_high",
            cam_left_wrist="observation.images.cam_left_wrist",
            cam_right_wrist="observation.images.cam_right_wrist",
        ),
        columns=dict(
            qpos="observations.state.qpos",
            eef="observations.state.eef",
            action_qpos="action.qpos",
            action_eef="action.eef",
        ),
    ),
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
            # EMA smoothing for small VR controller pose jitter (1.0 disables it).
            eef_filter_alpha=0.1,
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
    tasks=dict(
        pick_up_the_mango_and_place_it_in_the_plate=[
            ("pick up the mango and place it in the plate", 100),
        ],
        clean=[
            ("clean the whiteboard", 100),
        ],
    ),
)
