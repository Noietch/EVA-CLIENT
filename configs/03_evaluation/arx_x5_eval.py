"""ARX X5: eval checkpoint sweep."""

_base_ = ['../01_deploy/arx_x5/_base.py']

eval_cfg = dict(
    storage=dict(
        fps=30,
        save_queue_max=15,
    ),
    trials_per_prompt=5,
    cli_mode='real',
    inference_strategy='async',
    reset_after_each_trial=False,
    skip_warmup_after_first=True,
    checkpoints=[
        dict(
            name='arx_x5_openpi_modified',
            config='../01_deploy/arx_x5/openpi_qpos.py',
            port=9000,
        ),
        dict(
            name='arx_x5_openpi_baseline',
            config='../01_deploy/arx_x5/openpi_qpos.py',
            port=9000,
        ),
    ],
    enable_ssh_forward=False,
    # To forward each checkpoint port to a remote inference server, fill in your
    # own endpoint below and set enable_ssh_forward=True.
    # ssh=dict(
    #     host='<remote-host>',
    #     user='<remote-user>',
    #     port=8000,
    #     remote_sync_dir='<remote-sync-dir>',
    # ),
    tasks=[
        dict(
            prompt_en='pick up the apple',
            milestones=(
                ('approach', 'approach the apple'),
                ('pick', 'pick up the apple'),
                ('place', 'place the apple into the plate'),
            ),
        ),
        dict(
            prompt_en='pick up the orange',
            milestones=(
                ('approach', 'approach the orange'),
                ('pick', 'pick up the orange'),
                ('place', 'place the orange into the plate'),
            ),
        ),
    ],
)
