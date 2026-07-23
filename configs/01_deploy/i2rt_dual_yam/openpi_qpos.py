"""I2RT dual YAM OpenPI deploy in joint space."""

_base_ = ["_base.py"]

inference_cfg = dict(
    inference_rate=15,
    publish_rate=30,
    debug_tasks=["pick up the object and place it in the target area"],
)
