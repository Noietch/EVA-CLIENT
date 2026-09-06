"""UR5e: eval checkpoint sweep."""

_base_ = ["../01_deploy/ur5e/openpi_qpos.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    storage=dict(fps=20),
    checkpoints=[
        dict(
            name="ur5e_openpi_qpos_baseline", config="../01_deploy/ur5e/openpi_qpos.py", port=9000
        ),
    ],
    tasks=[
        dict(
            prompt_en="pick the apple",
            milestones=(
                ("approach", "approach apple"),
                ("grasp", "grasp apple"),
                ("lift", "lift off table"),
            ),
        ),
    ],
)
