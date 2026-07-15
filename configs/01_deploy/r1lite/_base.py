"""R1 Lite: shared deploy settings across openpi / eva, eef / qpos."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    type='r1_lite',
    eef_reference_frame='torso_link3',
    gripper_open=100.0,
)

transport = dict(
    type='ros2',
    topics=dict(
        camera_topics=dict(
            head='/hdas/camera_head/right_raw/image_raw_color/compressed',
            left_wrist='/hdas/camera_wrist_left/color/image_raw/compressed',
            right_wrist='/hdas/camera_wrist_right/color/image_raw/compressed',
        ),
        group_topics=dict(
            left_arm=dict(
                state_topic='/hdas/feedback_arm_left',
                command_topic='/motion_target/target_joint_state_arm_left',
                eef_state_topic='/relaxed_ik/motion_control/pose_ee_arm_left',
                gripper_state_topic='/hdas/feedback_gripper_left',
                gripper_command_topic='/motion_target/target_position_gripper_left',
            ),
            right_arm=dict(
                state_topic='/hdas/feedback_arm_right',
                command_topic='/motion_target/target_joint_state_arm_right',
                eef_state_topic='/relaxed_ik/motion_control/pose_ee_arm_right',
                gripper_state_topic='/hdas/feedback_gripper_right',
                gripper_command_topic='/motion_target/target_position_gripper_right',
            ),
        ),
    ),
)

collection = dict(
    storage=dict(
        log_dir='work_dirs/collection/r1lite',
        fps=15,
        save_queue_max=15,
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
)

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir='work_dirs/rollout/r1lite',
        fps=15,
        save_queue_max=15,
        async_save=True,
        image_height=360,
        image_width=640,
    ),
    intervention=dict(control_mode='relative'),
)

operator_control = dict(
    enabled=True,
    button_topic='/eva/operator_button',
)
