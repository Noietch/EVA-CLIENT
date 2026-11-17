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

# Sync publishes each chunk step at the 30 Hz loop rate, as in the
# demonstrations, without waiting for the robot to reach every step. The arrival
# wait (up to 3.3 s per step) never ended once a gripper closed on an object,
# because the gripper cannot reach its commanded closed value.
inference_strategies = {
    "sync": dict(args=dict(sync_wait_threshold=9999.0)),
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
