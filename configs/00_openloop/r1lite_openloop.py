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
    dataset_dir='datasets/sft/real_robot/galaxea_r1_lite/r1lite_pack_phone_new_eva_vla_abs_gripper_open',
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
    debug_tasks=[
        'first scoop up the black foam and place it in the box, then scoop up the phone and place it in the box, and finally pick up the lid, put it on the box, and press it down firmly',  # noqa: E501
    ],
)
