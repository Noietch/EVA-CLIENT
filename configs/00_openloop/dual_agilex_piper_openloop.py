"""Dual Piper: open-loop dataset replay."""

_base_ = ['../00_base/defaults.py']

robot = dict(
    gripper_threshold=None,
    gripper_open=0.1,
)

transport = dict(
    type='dataset',
    dataset_dir='examples/agilex_dataset',
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
    port=8006,
)

inference_cfg = dict(
    setup_warmup_chunks=0,
    debug_tasks=['pick up the orange block and place it on the green plate'],
)
