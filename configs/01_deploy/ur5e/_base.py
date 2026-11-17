"""UR5e: shared deploy settings for openpi (eef / qpos)."""

_base_ = ["../../00_base/defaults.py", "../../00_base/collection.py"]

console = dict(initial_tab="debug")

robot = dict(type="ur5e")

transport = dict(resize_pad=False, image_layout="hwc")

collection = dict(
    storage=dict(log_dir="work_dirs/collection/ur5e", fps=20), schema=dict(min_episode_frames=10)
)

rollout = dict(
    storage=dict(enabled=True, log_dir="work_dirs/rollout/ur5e", fps=20),
    intervention=dict(control_mode="relative"),
)
