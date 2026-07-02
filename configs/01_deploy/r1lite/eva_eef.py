"""R1 Lite: eva policy deploy, EEF-pose action space."""

_base_ = ['eva_base.py']

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=8),
)

robot = dict(
    gripper_threshold=0.9,
)

inference_cfg = dict(
    obs_space=dict(
        type='EEFPose',
        n_arms=2,
        rotation='quat',
        include_gripper=True,
    ),
    action_space=dict(
        type='EEFPose',
        n_arms=2,
        rotation='quat',
        include_gripper=True,
    ),
    publish_rate=15,
    debug_tasks=['placeholder task — replace with the real EEF task prompt'],
)

inference_strategies = {
    'sync': dict(
        args=dict(
            execute_horizon=50,
        ),
    ),
    'async': dict(
        args=dict(
            latency_k=8,
        ),
    ),
    'rtc': dict(
        args=dict(
            latency_k=8,
        ),
    ),
}
