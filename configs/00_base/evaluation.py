"""Shared evaluation workflow settings."""

eval_cfg = dict(
    task_set_dir="",
    checkpoints=[],
    tasks=[],
    storage=dict(enabled=True, fps=30, save_queue_max=15),
    trials_per_prompt=5,
    cli_mode="real",
    inference_strategy="async",
    reset_after_each_trial=False,
    skip_warmup_after_first=True,
    shuffle_ckpts=False,
    shuffle_seed=42,
    enable_ssh_forward=False,
)
