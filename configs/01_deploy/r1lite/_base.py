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
