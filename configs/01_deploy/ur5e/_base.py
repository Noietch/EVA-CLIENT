"""UR5e: shared deploy settings for openpi (eef / qpos)."""

_base_ = ['../../00_base/defaults.py']

robot = dict(
    type='ur5e',
)

transport = dict(
    type='zmq',
    resize_pad=False,
    image_layout='hwc',
)

