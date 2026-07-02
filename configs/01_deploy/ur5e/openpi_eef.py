"""UR5e: openpi policy deploy, EEF-pose action space."""

_base_ = ['_base.py']

policy = dict(
    type='openpi_rtc',
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    obs_space=dict(
        type='EEFPose',
        n_arms=1,
        rotation='quat',
        include_gripper=True,
    ),
    action_space=dict(
        type='EEFPose',
        n_arms=1,
        rotation='quat',
        include_gripper=True,
    ),
    publish_rate=25,
    debug_tasks=['placeholder task — replace with the real EEF task prompt'],
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
