"""UR5e: teleop data collection (schema + teleop)."""

_base_ = ['../01_deploy/ur5e/_base.py']

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/ur5e',
        fps=20,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type='ur5e',
        min_episode_frames=10,
        arms=dict(
            arm='arm',
        ),
        cameras=dict(
            cam_high='observation.images.cam_high',
            cam_wrist='observation.images.cam_wrist',
        ),
        columns=dict(
            qpos='observations.state.qpos',
            eef='observations.state.eef',
            action_qpos='action.qpos',
            action_eef='action.eef',
        ),
    ),
    teleop=dict(
        type='ur5e_master_arm',
        port='/dev/ttyACM0',
        joint_coef=[1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        gripper=dict(
            source='analog',
        ),
    ),
    tasks=['pick up the apple'],
)
