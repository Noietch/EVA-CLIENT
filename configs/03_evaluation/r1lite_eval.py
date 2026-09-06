"""R1 Lite: openpi qpos eval."""

_base_ = ["../01_deploy/r1lite/openpi_qpos.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    storage=dict(fps=15),
    inference_strategy="rtc",
    checkpoints=[
        dict(
            name="r1lite_openpi_qpos_baseline",
            config="../01_deploy/r1lite/openpi_qpos.py",
            port=9000,
        ),
    ],
    tasks=[
        dict(
            prompt_en="placeholder task — replace with the real task prompt",
            milestones=(("complete", "complete the task"),),
        ),
    ],
)
