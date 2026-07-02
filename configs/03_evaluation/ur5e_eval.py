"""UR5e: eval checkpoint sweep."""

_base_ = ['../01_deploy/ur5e/openpi_qpos.py']

eval_cfg = dict(
    storage=dict(
        fps=20,
        save_queue_max=15,
    ),
    trials_per_prompt=5,
    cli_mode='real',
    inference_strategy='async',
    reset_after_each_trial=False,
    skip_warmup_after_first=True,
    checkpoints=[
        dict(
            name='ur5e_openpi_qpos_baseline',
            config='../01_deploy/ur5e/openpi_qpos.py',
            port=9000,
        ),
    ],
    enable_ssh_forward=False,
    tasks=[
        dict(
            prompt_en='pick the apple',
            milestones=(
                ('approach', 'approach apple'),
                ('grasp', 'grasp apple'),
                ('lift', 'lift off table'),
            ),
        ),
    ],
)
