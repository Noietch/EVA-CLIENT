"""ARX R5: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    type='arx_r5',
    gripper_threshold=None,
)

transport = dict(
    type='zmq',
    resize_pad=False,
    image_layout='hwc',
)

policy = dict(type='openpi')

rollout = dict(
    storage=dict(
        enabled=True,
        log_dir='work_dirs/rollout/arx_r5',
        fps=30,
        save_queue_max=15,
        async_save=True,
    ),
    intervention=dict(control_mode='relative'),
)

inference_strategies = {
    'sync': dict(
        args=dict(
            execute_horizon=30,
        ),
    ),
    'async': dict(
        args=dict(
            latency_k=8,
        ),
    ),
    'rtc': dict(
        args=dict(
            latency_k=10,
        ),
    ),
}
