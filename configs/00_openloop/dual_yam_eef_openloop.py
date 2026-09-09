"""Dual YAM: EEF open-loop dataset replay."""

_base_ = ["../01_deploy/dual_yam/openpi_eef.py"]

transport = dict(
    type="dataset",
    dataset_dir=(
        "datasets/data_collection/datasets/real_robot/dual_yam/"
        "pick_up_the_object_and_place_it_in_the_target_area"
    ),
    episode_id=0,
    convert_bgr_to_rgb=True,
    image_height=224,
    image_width=224,
    resize_pad=False,
    image_layout="hwc",
    dataset_keys=dict(
        state_key="observations.state.eef",
        action_key="action.eef",
        video_keys=dict(
            cam_high="observation.images.cam_high",
            cam_left_wrist="observation.images.cam_left_wrist",
            cam_right_wrist="observation.images.cam_right_wrist",
        ),
    ),
)

policy = dict(
    host="127.0.0.1",
    port=9000,
)

openloop = dict(
    output_dir=(
        "work_dirs/openloop/dual_yam_eef/"
        "pick_up_the_object_and_place_it_in_the_target_area/episode_000000"
    ),
    execute_horizon=None,
    max_steps=None,
    startup_timeout_s=30,
)
