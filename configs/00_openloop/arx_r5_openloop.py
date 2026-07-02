"""ARX R5: open-loop dataset replay."""

_base_ = ['../00_base/defaults.py']

robot = dict(
    type='arx_r5',
    gripper_threshold=None,
)

transport = dict(
    type='dataset',
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
    debug_tasks=['pick up the block and place it on the plate'],
)
