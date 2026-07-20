"""Dual Piper EVA Sim teleoperation collection."""

_base_ = ["dual_agilex_piper.py"]

transport = dict(
    type="zmq",
    resize_pad=False,
    image_layout="hwc",
    convert_bgr_to_rgb=False,
    topics={},
    ssh=dict(host="", user="", port=0),
)

collection = dict(
    storage=dict(
        log_dir="work_dirs/collection/dual_agilex_piper_eva_sim",
    ),
    tasks=["pick and place"],
)

control_channel = dict(
    enabled=True,
    host="127.0.0.1",
    port=5757,
)
