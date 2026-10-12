"""Dual YAM: end-effector policy evaluation through EVA Client and EVA-SIM."""

_base_ = ["../01_deploy/dual_yam/openpi_eef.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

inference_cfg = dict(max_debug_time_s=1800.0)

transport = dict(
    type="zmq",
    image_mode="on_demand",
    convert_bgr_to_rgb=False,
)

control_channel = dict(enabled=True)
