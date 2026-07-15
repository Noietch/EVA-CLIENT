"""R1 Lite: teleop data collection over ROS2."""

_base_ = ['../01_deploy/r1lite/_base.py']

transport = dict(
    resize_pad=False,
    image_layout='hwc',
)

collection = dict(
    storage=dict(
        image_height=360,
        image_width=640,
    ),
)
