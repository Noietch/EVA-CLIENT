"""ARX R5: openpi policy deploy, EEF-pose action space."""

_base_ = ['_base.py']

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
        'pick up the mango and place it on the plate',
        'pick up the orange',
        'pick up the block and place it on the plate',
    ],
)
