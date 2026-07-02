"""UR5e: open-loop dataset replay."""

_base_ = ['../00_base/defaults.py']

robot = dict(
    type='ur5e',
)

transport = dict(
    type='dataset',
    resize_pad=False,
    image_layout='hwc',
    dataset_keys=dict(
        state_key='observation.qpos',
        video_keys=dict(
            cam_high='observation.images.cam_high',
            cam_wrist='observation.images.cam_wrist',
        ),
    ),
)

policy = dict(
    port=8000,
)

inference_cfg = dict(
    publish_rate=25,
    setup_warmup_chunks=0,
    debug_tasks=['Put the apples in the white basket'],
)
