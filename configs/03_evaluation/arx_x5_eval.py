"""ARX X5: mango pick-and-place eval checkpoint sweep."""

_base_ = ["../01_deploy/arx_x5/_base.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    checkpoints=[
        dict(name="arx_x5_openpi_modified", config="../01_deploy/arx_x5/openpi_eef.py", port=9000),
    ],
    tasks=[
        dict(
            prompt_en="pick up the mango and place it in the plate",
            milestones=(
                ("approach", "approach the mango"),
                ("pick", "pick up the mango"),
                ("place", "place the mango into the plate"),
            ),
        ),
    ],
)
