"""Dual Franka: openpi policy deploy, qpos action space."""

_base_ = ['_base.py']

inference_cfg = dict(
    debug_tasks=['push black block'],
)
