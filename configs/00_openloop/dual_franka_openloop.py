"""Dual Franka: open-loop dataset replay."""

_base_ = ['../00_base/defaults.py']

robot = dict(
    type='dual_franka',
    gripper_threshold=None,
)

transport = dict(
    type='dataset',
    dataset_dir='datasets/sft/real_robot/tong_franka/push_black_block_gripper_open',
    resize_pad=False,
    image_layout='hwc',
    disabled_cameras=['cam_left_wrist'],
    disabled_groups=['left_arm'],
    dataset_keys=dict(
        state_key='observation.state',
        video_keys=dict(
            cam_high='observation.images.topcam',
            cam_right_wrist='observation.images.wristcam',
        ),
    ),
)

policy = dict(
    port=9000,
)

inference_cfg = dict(
    setup_warmup_chunks=0,
    debug_tasks=['push black block'],
)
