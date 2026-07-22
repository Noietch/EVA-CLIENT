"""R1 Lite: open-loop dataset replay."""

_base_ = ['../00_base/defaults.py']

robot = dict(
    type='r1_lite',
    gripper_threshold=None,
    gripper_open=0.0,
    gripper_close=1.0,
)

transport = dict(
    type='dataset',
    dataset_dir='datasets/r1_lite',
    resize_pad=False,
    image_layout='hwc',
    dataset_keys=dict(
        state_key='observation.qpos',
        video_keys=dict(
            cam_high='observation.images.cam_high',
            cam_left_wrist='observation.images.cam_left_wrist',
            cam_right_wrist='observation.images.cam_right_wrist',
        ),
    ),
)

policy = dict(
    port=9000,
)

inference_cfg = dict(
    publish_rate=15,
    setup_warmup_chunks=0,
    debug_tasks=['placeholder task — replace with the real task prompt'],
)
