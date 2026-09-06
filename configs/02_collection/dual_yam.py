"""YAM dual YAM leader-follower data collection."""

_base_ = ["../00_base/defaults.py", "../00_base/collection.py"]

console = dict(initial_tab="collect")

collection = dict(
    storage=dict(log_dir="work_dirs/collection/dual_yam/", image_skew_tolerance_sec=0.02),
    tasks=dict(
        pick_up_the_object_and_place_it_in_the_target_area=[
            ("pick up the object and place it in the target area", -1)
        ],
        put_cup_on_the_plate=[("put cup on the plate", -1)],
        put_all_objects_into_the_box=[("put all objects into the box", -1)],
        put_blocks_on_corresponding_signs=[("put blocks on corresponding signs", -1)],
    ),
)

robot = dict(type="dual_yam")
