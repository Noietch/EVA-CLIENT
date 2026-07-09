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
            name='r1lite_baseline_A',
            config='../01_deploy/r1lite/openpi_qpos.py',
            port=9000,
        ),
        # dict(
        #     name='pi05_huawei_ww_lora_2606_task1_0621_100000',
        #     config='../01_deploy/r1lite/openpi_qpos.py',
        #     port=9001,
        # ),
    ],
    shuffle_ckpts=False,
    shuffle_seed=42,
    enable_ssh_forward=False,
    tasks=[
        dict(
            prompt_en='pack up a smart phone',
            milestones=(
                ('half', 'about half of the objects are stored correctly'),
                ('all', 'all objects are stored correctly'),
            ),
        ),
        # dict(
        #     prompt_en='Sort the tabletop items: put trash into the black bin and useful items into the white bin.',  # noqa: E501
        #     milestones=(
        #         ('half', 'about half of the tabletop items are sorted correctly'),
        #         ('all', 'all tabletop items are sorted correctly'),
        #     ),
        # ),
    ],
)
