"""ARX X5: shared deploy settings for openpi (eef / qpos)."""

_base_ = ["../../00_base/defaults.py"]

console = dict(initial_tab="debug")

robot = dict(type="arx_x5")

# The ARX X5 RealSense pipeline streams BGR8, while the policy, console preview
# and recordings expect RGB; convert at the transport boundary.
transport = dict(resize_pad=False, image_layout="hwc", convert_bgr_to_rgb=True)

rollout = dict(
    storage=dict(enabled=True, log_dir="work_dirs/rollout/arx_x5"),
    intervention=dict(control_mode="relative"),
)

inference_strategies = {
    "async": dict(
        args=dict(
            latency_k=8,
        ),
    ),
    "rtc": dict(
        args=dict(
            latency_k=10,
        ),
    ),
}
