"""YAM dual YAM OpenPI deploy in end-effector pose space."""

_base_ = ["_base.py"]

inference_cfg = dict(
    obs_space=dict(
        type="EEFPose",
        n_arms=2,
        rotation="quat",
        rotation_order="wxyz",
        include_gripper=True,
    ),
    action_space=dict(
        type="EEFPose",
        n_arms=2,
        rotation="quat",
        rotation_order="wxyz",
        include_gripper=True,
    ),
    inference_rate=3,
    publish_rate=30,
    debug_tasks=[
        "pick up the object and place it in the target area",
        "put cup on the plate",
        "put all objects into the box",
        "put blocks on corresponding signs",
    ],
)

inference_strategies = {
    "sync": dict(
        args=dict(
            execute_horizon=50,
        ),
    ),
    "async": dict(
        args=dict(
            latency_k=0,
        ),
    ),
}
