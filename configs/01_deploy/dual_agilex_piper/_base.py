"""Dual Piper: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    gripper_threshold=None,
    gripper_open=0.1,
)

