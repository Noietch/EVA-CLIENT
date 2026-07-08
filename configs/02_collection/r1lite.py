"""R1 Lite: teleop data collection over ROS2."""

_base_ = ['../01_deploy/r1lite/_base.py']

robot = dict(
    type='r1_lite',
    eef_reference_frame='torso_link3',
)

transport = dict(
    type='ros2',
    resize_pad=False,
    image_layout='hwc',
)

policy = dict(
    port=9000,
)

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/r1lite',
        fps=15,
        save_queue_max=15,
        image_height=360,
        image_width=640,
    ),
    schema=dict(
        robot_type='r1_lite',
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
        ros2=dict(
            max_frame_skew_sec=0.3,
            groups=dict(
                left_arm=dict(
                    qpos_topic='/hdas/feedback_arm_left',
                    qpos_gripper_topic='/hdas/feedback_gripper_left',
                    eef_topic='/relaxed_ik/motion_control/pose_ee_arm_left',
                    hil_qpos_topic='/eva/hil/input_joint_state_arm_left',
                    action_qpos_topic='/motion_target/target_joint_state_arm_left',
                    action_qpos_gripper_source='operator',
                    action_qpos_gripper_topic=None,
                    action_qpos_gripper_binary=True,
                    action_eef_topic='/relaxed_ik/motion_control/pose_ee_arm_left',
                ),
                right_arm=dict(
                    qpos_topic='/hdas/feedback_arm_right',
                    qpos_gripper_topic='/hdas/feedback_gripper_right',
                    eef_topic='/relaxed_ik/motion_control/pose_ee_arm_right',
                    hil_qpos_topic='/eva/hil/input_joint_state_arm_right',
                    action_qpos_topic='/motion_target/target_joint_state_arm_right',
                    action_qpos_gripper_source='operator',
                    action_qpos_gripper_topic=None,
                    action_qpos_gripper_binary=True,
                    action_eef_topic='/relaxed_ik/motion_control/pose_ee_arm_right',
                ),
            ),
        ),
    ),
    tasks=['pick up the apple'],
)

operator_control = dict(
    enabled=True,
    button_topic='/eva/operator_button',
)

inference_cfg = dict(
    publish_rate=15,
    setup_warmup_chunks=3,
)

supported_prompts = [
    'first scoop up the black foam and place it in the box, then scoop up the phone and place it in the box, and finally pick up the lid, put it on the box, and press it down firmly',  # noqa: E501
]

inference_strategies = dict(
    sync=dict(
        args=dict(
            execute_horizon=50,
        ),
    ),
)
