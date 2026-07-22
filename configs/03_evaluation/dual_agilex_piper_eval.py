"""Dual Piper: openpi qpos eval (checkpoints + SSH-forwarded remote endpoint)."""

_base_ = ['../01_deploy/dual_agilex_piper/openpi_qpos.py']

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
            name='qwen3vl_mlp_oft_xpred_qpos-agilex-ood-eval_qpos-agilex-midtrain_from_mid_train_539_step30000',
            config='../01_deploy/dual_agilex_piper/openpi_qpos.py',
            port=9000,
        ),
    ],
    shuffle_ckpts=False,
    shuffle_seed=42,
    enable_ssh_forward=False,
    # To forward each checkpoint port to a remote inference server, fill in your
    # own endpoint below and set enable_ssh_forward=True.
    ssh=dict(
        host='<remote-host>',
        user='<remote-user>',
        port=8000,
        remote_sync_dir='<remote-sync-dir>',
    ),
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
