"""Dual Piper: teleop data collection (schema + collection topics)."""

_base_ = ["../00_base/defaults.py", "../00_base/collection.py"]

console = dict(initial_tab="collect")

collection = dict(
    storage=dict(
        log_dir="datasets/data_collection/datasets/real_robot/agilex_piper",
    ),
    schema=dict(min_episode_frames=10),
    tasks=dict(pick_up_the_apple=[("pick up the apple", 10)]),
)

dashboard = dict(
    raw_roots=["datasets/data_collection/datasets/real_robot/agilex_piper"],
    raw_roots_only=True,
)

robot = dict(type="agilex_piper")
