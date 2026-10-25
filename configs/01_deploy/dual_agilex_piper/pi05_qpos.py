"""Dual Piper: Pi05 through XPolicyLab with absolute joint-QPos output."""

_base_ = ["xpolicylab.py"]

policy = dict(
    backend_options=dict(
        action_type="joint",
    ),
)

inference_cfg = dict(
    obs_space=dict(type="JointState"),
    action_space=dict(type="JointState"),
)
