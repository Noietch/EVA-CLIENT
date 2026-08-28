"""ARX R5: teleop data collection (schema + tasks)."""

_base_ = ["../01_deploy/arx_r5/_base.py"]

console = dict(initial_tab="collect")

collection = dict(
    storage=dict(
        log_dir="work_dirs/collection/arx_r5/",
        fps=30,
        save_queue_max=15,
    ),
    schema=dict(
        robot_type="arx_r5",
        arms=dict(
            left_arm="left",
            right_arm="right",
        ),
        cameras=dict(
            cam_high="observation.images.cam_high",
            cam_left_wrist="observation.images.cam_left_wrist",
            cam_right_wrist="observation.images.cam_right_wrist",
        ),
        columns=dict(
            qpos="observations.state.qpos",
            eef="observations.state.eef",
            action_qpos="action.qpos",
            action_eef="action.eef",
        ),
    ),
    teleop=dict(
        control_source="transport",
    ),
    tasks=dict(
        pick_up_the_apple=[("pick up the apple", -1)],
        pick_up_the_orange=[("pick up the orange", -1)],
    ),
)
