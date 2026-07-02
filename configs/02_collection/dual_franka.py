"""Dual Franka: teleop data collection (schema)."""

_base_ = ['../01_deploy/dual_franka/_base.py']

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/dual_franka',
        fps=30,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type='dual_franka',
        min_episode_frames=10,
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
    tasks=['pick up the apple'],
)
