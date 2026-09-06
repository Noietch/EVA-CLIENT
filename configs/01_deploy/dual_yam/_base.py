"""YAM dual YAM shared deploy settings."""

_base_ = ["../../00_base/defaults.py"]

console = dict(initial_tab="debug")

robot = dict(type="dual_yam")

transport = dict(resize_pad=False, image_layout="hwc")

rollout = dict(
    storage=dict(enabled=True, log_dir="work_dirs/rollout/dual_yam"),
    intervention=dict(control_mode="relative"),
)
