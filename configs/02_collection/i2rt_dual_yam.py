"""I2RT dual YAM leader-follower data collection."""

_base_ = ["../01_deploy/i2rt_dual_yam/_base.py"]

collection = dict(
    storage=dict(
        log_dir="work_dirs/collection/i2rt_dual_yam/",
        fps=30,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type="i2rt_dual_yam",
        arms=dict(
            left_arm="left",
            right_arm="right",
        ),
        cameras=dict(
            cam_high="observation.images.cam_high",
        ),
        columns=dict(
            qpos="observations.state.qpos",
            eef="observations.state.eef",
            action_qpos="action.qpos",
            action_eef="action.eef",
        ),
    ),
    tasks=["pick up the object and place it in the target area"],
)
