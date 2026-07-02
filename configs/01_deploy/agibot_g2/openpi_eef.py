"""AgiBot G2: openpi policy deploy, EEF-pose action space."""

_base_ = ['_base.py']

inference_cfg = dict(
    action_space=dict(
        type='EEFPose',
    ),
)
