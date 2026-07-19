"""ARX X5: teleop data collection (schema + tasks)."""

_base_ = ['../01_deploy/arx_x5/_base.py']

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/arx_x5/',
        fps=30,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type='arx_x5',
        arms=dict(
            left_arm='left',
            right_arm='right',
        ),
        cameras=dict(
            cam_high='observation.images.cam_high',
            cam_left_wrist='observation.images.cam_left_wrist',
            cam_right_wrist='observation.images.cam_right_wrist',
        ),
        columns=dict(
            qpos='observations.state.qpos',
            eef='observations.state.eef',
            action_qpos='action.qpos',
            action_eef='action.eef',
        ),
    ),
    tasks=['pick up the apple','pick up the orange'],
)
