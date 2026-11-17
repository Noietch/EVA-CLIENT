"""Dual YAM: EEF policy evaluation configuration."""

_base_ = ["../01_deploy/dual_yam/openpi_eef.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    checkpoints=[
        dict(
            name="dual_yam_eef",
            config="../01_deploy/dual_yam/openpi_eef.py",
            port=9000,
        ),
    ],
    tasks=[
        dict(
            prompt_en="pick up the object and place it in the target area",
            milestones=(
                ("approach", "approach the object"),
                ("pick", "pick up the object"),
                ("place", "place it in the target area"),
            ),
        ),
    ],
)
