"""R1 Lite: shared eva deploy settings (eef / qpos)."""

_base_ = ['_base.py']

transport = dict(
    resize_pad=False,
    image_layout='hwc',
)
