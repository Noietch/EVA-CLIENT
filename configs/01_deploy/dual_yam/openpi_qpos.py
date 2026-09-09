"""YAM dual YAM OpenPI deploy in joint space."""

_base_ = ["_base.py"]

inference_cfg = dict(
    inference_rate=3,
    publish_rate=30,
    debug_tasks=[
        "pick up the object and place it in the target area",
        "put cup on the plate",
        "put all objects into the box",
        "put blocks on corresponding signs",
    ],
)
