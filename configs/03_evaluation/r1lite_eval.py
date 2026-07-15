"""R1 Lite: openpi qpos eval."""

_base_ = ['../01_deploy/r1lite/openpi_qpos.py']

eval_cfg = dict(
    storage=dict(
        fps=15,
        save_queue_max=15,
    ),
    trials_per_prompt=5,
    cli_mode='real',
    inference_strategy='rtc',
    reset_after_each_trial=False,
    skip_warmup_after_first=True,
    checkpoints=[
        dict(
            name='r1lite_openpi_qpos_baseline',
            config='../01_deploy/r1lite/openpi_qpos.py',
            port=9000,
        ),
    ],
    shuffle_ckpts=False,
    shuffle_seed=42,
    enable_ssh_forward=False,
    tasks=[
        dict(
            prompt_en='placeholder task — replace with the real task prompt',
            milestones=(('complete', 'complete the task'),),
        ),
    ],
)
