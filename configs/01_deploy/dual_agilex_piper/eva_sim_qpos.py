"""Dual Piper: EVA Sim ZMQ integration with a local mock policy."""

_base_ = ["_base.py"]

robot = dict(
    extra_cameras=dict(third_person="cam_third_person"),
)

transport = dict(
    type="zmq",
    resize_pad=False,
    image_layout="hwc",
    convert_bgr_to_rgb=False,
    topics={},
    ssh=dict(host="", user="", port=0),
)

policy = dict(
    type="mock",
    backend_options=dict(chunk_size=8),
)

inference_cfg = dict(
    setup_warmup_chunks=0,
    debug_tasks=["pick up the orange block and place it on the green plate."],
)

control_channel = dict(
    enabled=True,
    host="127.0.0.1",
    port=5757,
)
