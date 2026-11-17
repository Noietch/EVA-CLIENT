"""R1 Lite: teleop data collection over ROS2."""

_base_ = ["../00_base/defaults.py", "../00_base/collection.py"]

console = dict(initial_tab="collect")

transport = dict(resize_pad=False, image_layout="hwc")

collection = dict(
    storage=dict(log_dir="work_dirs/collection/r1lite", fps=15, image_height=360, image_width=640),
    schema=dict(min_episode_frames=10),
)

robot = dict(type="r1_lite")
