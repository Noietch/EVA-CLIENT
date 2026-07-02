"""R1 Lite: openpi policy deploy, EEF-pose action space."""

_base_ = ['openpi_base.py']

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
    debug_tasks=[
        'move the white box from the left to the center, then pick up the yellow-red mango on the right and place it inside the box',  # noqa: E501
    ],
)

inference_strategies = {
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
