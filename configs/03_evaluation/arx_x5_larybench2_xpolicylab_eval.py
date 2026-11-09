"""ARX X5: local real-robot eval against a cloud XPolicyLab policy (joint/qpos)."""

_base_ = ["../01_deploy/arx_x5/xpolicylab.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    task_set_dir="work_dirs/tasksets/larybench2_20261102_real",
    # One row in this task set is one episode-level eval case, so a single trial
    # per task covers it. Raise this only to repeat tasks intentionally.
    trials_per_prompt=1,
    cli_mode="real",
    inference_strategy="async",
    checkpoints=[
        dict(
            name="arx_x5_larybench2_xpolicylab_qpos",
            config="../01_deploy/arx_x5/xpolicylab.py",
            host="33.29.36.208",
            port=8416,
        ),
    ],
)
