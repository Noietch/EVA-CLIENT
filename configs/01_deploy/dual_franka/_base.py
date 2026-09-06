"""Dual Franka: shared deploy settings for openpi (eef / qpos)."""

_base_ = ["../../00_base/defaults.py"]

console = dict(initial_tab="debug")

robot = dict(type="dual_franka")

transport = dict(resize_pad=False, image_layout="hwc")

policy = dict(type="openpi_rtc", backend_options=dict(latency_k=10))

inference_strategies = {
    "sync": dict(
        args=dict(
            execute_horizon=50,
        ),
    ),
    "async": dict(
        args=dict(
            latency_k=4,
        ),
    ),
    "rtc": dict(
        args=dict(
            latency_k=10,
        ),
    ),
}
