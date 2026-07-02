"""Dual Franka: openpi policy deploy, EEF-pose action space."""

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
    debug_tasks=['placeholder task — replace with the real EEF task prompt'],
)
