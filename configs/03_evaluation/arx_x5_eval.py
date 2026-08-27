"""ARX X5: mango pick-and-place eval checkpoint sweep."""

_base_ = ["../01_deploy/arx_x5/_base.py"]

eval_cfg = dict(
    storage=dict(
        fps=30,
        save_queue_max=15,
    ),
    trials_per_prompt=5,
    cli_mode="real",
    inference_strategy="async",
    reset_after_each_trial=False,
    skip_warmup_after_first=True,
    checkpoints=[
        dict(
            name="arx_x5_openpi_modified",
            config="../01_deploy/arx_x5/openpi_eef.py",
            port=9000,
        ),
    ],
    shuffle_ckpts=False,
    shuffle_seed=42,
    enable_ssh_forward=False,
    # To forward each checkpoint port to a remote inference server, fill in your
    # own endpoint below and set enable_ssh_forward=True.
    # ssh=dict(
    #     host="<remote-host>",
    #     user="<remote-user>",
    #     port=8000,
    #     remote_sync_dir="<remote-sync-dir>",
    # ),
    tasks=[
        dict(
            prompt_en="pick up the mango and place it in the plate",
            milestones=(
                ("approach", "approach the mango"),
                ("pick", "pick up the mango"),
                ("place", "place the mango into the plate"),
            ),
        ),
    ],
)
