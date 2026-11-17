"""Dual Piper: shared deploy settings for openpi (eef / qpos)."""

_base_ = ["../../00_base/defaults.py"]

console = dict(initial_tab="debug")

robot = dict(type="agilex_piper")

# The Astra color topics publish truthfully-labelled `rgb8`; the transport
# normalizes them to OpenCV-native BGR, which is the convention every consumer
# below assumes. Convert back to RGB for the policy, the console preview and the
# dataset videos.
transport = dict(resize_pad=False, image_layout="hwc", convert_bgr_to_rgb=True)

rollout = dict(
    storage=dict(enabled=True, log_dir="work_dirs/rollout/dual_agilex_piper"),
    intervention=dict(control_mode="relative"),
)
