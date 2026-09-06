"""UR5e: teleop data collection (schema + teleop)."""

_base_ = ["../00_base/defaults.py", "../00_base/collection.py"]

console = dict(initial_tab="collect")

collection = dict(
    storage=dict(log_dir="work_dirs/collection/ur5e", fps=20),
    schema=dict(min_episode_frames=10),
    tasks=dict(pick_up_the_apple=[("pick up the apple", -1)]),
)

robot = dict(type="ur5e")
