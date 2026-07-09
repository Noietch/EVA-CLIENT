"""Dual Piper: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    gripper_threshold=None,
    gripper_open=0.1,
)

transport = dict(
    type='ros1',
    image_height=224,
    image_width=224,
    resize_pad=False,
    image_layout='hwc',
    convert_bgr_to_rgb=False,
    ssh=dict(host='<remote-host>', user='<remote-user>', port=8000),
    topics=dict(
        camera_topics=dict(
            front='/camera_f/color/image_raw',
            left_wrist='/camera_l/color/image_raw',
            right_wrist='/camera_r/color/image_raw',
        ),
        group_topics=dict(
            left_arm=dict(
                state_topic='/puppet/joint_left',
                command_topic='/puppet/master_joint_left',
                eef_state_topic='/puppet/end_pose_left',
                sim_command_topic='/sim_joint_left',
            ),
            right_arm=dict(
                state_topic='/puppet/joint_right',
                command_topic='/puppet/master_joint_right',
                eef_state_topic='/puppet/end_pose_right',
                sim_command_topic='/sim_joint_right',
            ),
        ),
    ),
)

