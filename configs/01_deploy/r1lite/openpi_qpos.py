"""R1 Lite: openpi policy deploy, qpos action space."""

_base_ = ['openpi_base.py']

policy = dict(
    type='openpi',
    backend_options=dict(latency_k=4),
)

transport = dict(
    resize_pad=False,
    image_layout='hwc',
)

robot = dict(
    gripper_threshold=80.0,
)

inference_cfg = dict(
    publish_rate=30,
    debug_tasks=['placeholder task — replace with the real task prompt'],
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
            latency_k=4,
        ),
    ),
}
