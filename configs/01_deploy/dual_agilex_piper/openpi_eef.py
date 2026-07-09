"""Dual Piper: openpi policy deploy, EEF-pose action space."""

_base_ = ['_base.py']

policy = dict(
    type='openpi',
    backend_options=dict(latency_k=8),
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
    debug_tasks=[
        'pick up letter blocks and place them to spell MEI, the order is: first M, then E, then I',
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
            latency_k=8,
        ),
    ),
    'rtc': dict(
        args=dict(
            latency_k=8,
        ),
    ),
}
