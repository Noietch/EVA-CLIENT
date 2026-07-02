"""R1 Lite: openpi policy deploy, qpos action space."""

_base_ = ['openpi_base.py']

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    debug_tasks=[
        'first scoop up the black foam and place it in the box, then scoop up the phone and place it in the box, and finally pick up the lid, put it on the box, and press it down firmly',  # noqa: E501
    ],
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
