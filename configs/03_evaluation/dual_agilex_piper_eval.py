"""Dual Piper: openpi qpos eval (checkpoints + SSH-forwarded remote endpoint)."""

_base_ = ["../01_deploy/dual_agilex_piper/openpi_qpos.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")

eval_cfg = dict(
    checkpoints=[
        dict(
            name="qwen3vl_mlp_oft_xpred_qpos-agilex-ood-eval_qpos-agilex-midtrain_from_mid_train_539_step30000",
            config="../01_deploy/dual_agilex_piper/openpi_qpos.py",
            port=9000,
        ),
    ],
    ssh=dict(
        host="<remote-host>", user="<remote-user>", port=8000, remote_sync_dir="<remote-sync-dir>"
    ),
    tasks=[
        dict(
            prompt_en="pick the apple",
            milestones=(
                ("approach", "approach apple"),
                ("grasp", "grasp apple"),
                ("lift", "lift off table"),
            ),
        ),
    ],
)
