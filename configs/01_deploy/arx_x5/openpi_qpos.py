"""ARX X5: openpi policy deploy, qpos action space."""

_base_ = ['_base.py']

inference_cfg = dict(
    inference_rate=15,
    publish_rate=40,
    debug_tasks=[
        'pick up the mango and place it on the plate',
        'pick up the block and place it on the plate',
        'pick up the orange',
    ],
)
