"""Dual Piper: openpi policy deploy, qpos action space."""

_base_ = ['_base.py']

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    debug_tasks=[
        'pick up the yellow spoon and place it on the green plate',
        'pick up the mango and place it on the green plate',
        'pick up the light blue fork and place it on the green plate',
        'pick up the green knife and place it on the green plate',
        'pick up the green block and place it on the green plate',
        'pick up the orange block and place it on the green plate',
        'pick up the green block and place it on the green plate',
        'pick up the yellow spoon and place it on the green plate',
        'pick up the gray cup and place it on the green plate',
        'pick up the light blue fork and place it on the green plate',
        'pick up the orange block and place it on the green plate',
        'pick up the yellow cup and place it on the green plate',
        'pick up the light blue spoon and place it on the green plate',
        'pick up the green cup and place it on the green plate',
        'pick up the red block and place it on the green plate',
        'pick up the lemon and place it on the green plate',
        'pick up the gray knife and place it on the green plate',
        'pick up the blue cup and place it on the green plate',
        'pick up the green fork and place it on the green plate',
        'pick up the banana and place it on the green plate',
        'pick up the blue block and place it on the green plate',
        'pick up the yellow cup and place it on the green plate',
        'pick up the green mango and place it on the green plate',
        'pick up the green knife and place it on the green plate',
    ],
)

inference_strategies = {
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
