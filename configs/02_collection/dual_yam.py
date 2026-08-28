"""YAM dual YAM leader-follower data collection."""

_base_ = ["../01_deploy/dual_yam/_base.py"]

console = dict(initial_tab="collect")

collection = dict(
    storage=dict(
        log_dir="work_dirs/collection/dual_yam/",
        fps=30,
        save_queue_max=15,
        image_skew_tolerance_sec=0.020,
    ),
    schema=dict(
        robot_type="dual_yam",
        arms=dict(
            left_arm="left",
            right_arm="right",
        ),
        cameras=dict(
            cam_high="observation.images.cam_high",
            cam_left_wrist="observation.images.cam_left_wrist",
            cam_right_wrist="observation.images.cam_right_wrist",
        ),
        columns=dict(
            qpos="observations.state.qpos",
            eef="observations.state.eef",
            action_qpos="action.qpos",
            action_eef="action.eef",
        ),
    ),
    tasks=dict(
        pick_up_the_object_and_place_it_in_the_target_area=[
            ("pick up the object and place it in the target area", -1),
        ],
        put_cup_on_the_plate=[
            ("put cup on the plate", -1),
        ],
        put_all_objects_into_the_box=[
            ("put all objects into the box", -1),
        ],
        put_blocks_on_corresponding_signs=[
            ("put blocks on corresponding signs", -1),
        ],
    ),
)
