"""Dual Piper: teleop data collection (schema + collection topics)."""

_base_ = ['../01_deploy/dual_agilex_piper/_base.py']

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/dual_agilex_piper',
        fps=30,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type='agilex_piper',
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
    transport=dict(
        ros1=dict(
            groups=dict(
                left_arm=dict(
                    qpos_topic='/puppet/joint_left',
                    eef_topic='/puppet/end_pose_left',
                    action_qpos_topic='/puppet/master_joint_left',
                    action_eef_topic='/puppet/end_pose_left',
                ),
                right_arm=dict(
                    qpos_topic='/puppet/joint_right',
                    eef_topic='/puppet/end_pose_right',
                    action_qpos_topic='/puppet/master_joint_right',
                    action_eef_topic='/puppet/end_pose_right',
                ),
            ),
        ),
    ),
    tasks=['pick up the apple'],
)
