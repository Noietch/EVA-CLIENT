"""Dual Piper: openpi policy over EVA Sim ZMQ transport, qpos action space."""

_base_ = ["openpi_qpos.py"]

transport = dict(
    type="zmq",
    resize_pad=False,
    image_layout="hwc",
    convert_bgr_to_rgb=False,
    topics={},
    ssh=dict(host="", user="", port=0),
)

# control_channel = dict(
#     enabled=True,
#     host="127.0.0.1",
#     port=5757,
# )
