"""Shared RL workflow settings."""

rl_cfg = dict(
    cli_mode="real",
    inference_strategy="async",
    tasks=["placeholder task — replace with the real RL task prompt"],
    data=dict(format="lerobot", storage=dict(fps=30, save_queue_max=15, async_save=True)),
    intervention=dict(control_mode="relative"),
)
