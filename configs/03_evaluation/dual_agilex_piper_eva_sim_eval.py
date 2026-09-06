"""Dual Piper: automated EVA-SIM evaluation through EVA Client and EVA-SIM."""

_base_ = ["dual_agilex_piper_eval.py"]


transport = dict(
    type="zmq",
    image_mode="on_demand",
)


control_channel = dict(
    enabled=True,
)
