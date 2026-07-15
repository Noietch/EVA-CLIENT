"""UR5e: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    type='ur5e',
)

transport = dict(
    type='zmq',
    resize_pad=False,
    image_layout='hwc',
)

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
)

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir='work_dirs/rollout/ur5e',
        fps=20,
        save_queue_max=15,
        async_save=True,
    ),
    intervention=dict(control_mode='relative'),
)
