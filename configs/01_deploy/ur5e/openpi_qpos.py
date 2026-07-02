"""UR5e: openpi policy deploy, qpos action space."""

_base_ = ['_base.py']

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    publish_rate=20,
    debug_tasks=['Put the apples in the white basket'],
)

inference_strategies = {
    'sync': dict(
        args=dict(
            sync_wait_ignore_gripper=True,
        ),
    ),
    'async': dict(
        args=dict(
            latency_k=4,
        ),
    ),
}
