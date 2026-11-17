"""ARX R5: eval checkpoint sweep."""

_base_ = ["../01_deploy/arx_r5/_base.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    checkpoints=[
        dict(name="arx_r5_openpi_modified", config="../01_deploy/arx_r5/openpi_qpos.py", port=9000),
        dict(name="arx_r5_openpi_baseline", config="../01_deploy/arx_r5/openpi_qpos.py", port=9000),
    ],
    tasks=[
        dict(
            prompt_en="pick up the apple",
            milestones=(
                ("approach", "approach the apple"),
                ("pick", "pick up the apple"),
                ("place", "place the apple into the plate"),
            ),
        ),
        dict(
            prompt_en="pick up the orange",
            milestones=(
                ("approach", "approach the orange"),
                ("pick", "pick up the orange"),
                ("place", "place the orange into the plate"),
            ),
        ),
    ],
)
