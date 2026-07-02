"""AgiBot G2: teleop data collection (schema)."""

_base_ = ['../01_deploy/agibot_g2/_base.py']

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/agibot_g2',
        fps=30,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type='agibot_g2',
        min_episode_frames=10,
        arms=dict(
            left_arm='left',
            right_arm='right',
        ),
        cameras=dict(
            top_head='observation.images.cam_high',
            hand_left='observation.images.cam_left_wrist',
            hand_right='observation.images.cam_right_wrist',
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
