"""Dual Franka: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    type='dual_franka',
    gripper_threshold=None,
)

transport = dict(
    type='zmq',
    resize_pad=False,
    image_layout='hwc',
    disabled_cameras=['cam_left_wrist'],
)

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=10),
)

inference_strategies = {
    'sync': dict(
        args=dict(
            execute_horizon=50,
        ),
    ),
    'async': dict(
        args=dict(
            latency_k=4,
        ),
    ),
    'rtc': dict(
        args=dict(
            latency_k=10,
        ),
    ),
}
