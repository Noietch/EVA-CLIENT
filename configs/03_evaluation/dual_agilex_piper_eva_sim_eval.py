"""Dual Piper: automated EVA-VLA evaluation through EVA Client and EVA Sim."""

_base_ = ["../01_deploy/dual_agilex_piper/eva_sim_openpi_qpos_zmq.py"]

transport = dict(
    render_on_demand=True,
    record_images=False,
)

eval_cfg = dict(
    storage=dict(fps=30, save_queue_max=15),
    trials_per_prompt=5,
    cli_mode="real",
    inference_strategy="sync",
    reset_after_each_trial=False,
    skip_warmup_after_first=False,
    shuffle_ckpts=False,
    shuffle_seed=42,
    checkpoints=[
        dict(
            name="eva_vla_replay",
            config="../01_deploy/dual_agilex_piper/eva_sim_openpi_qpos_zmq.py",
            port=9000,
        ),
    ],
    enable_ssh_forward=False,
    ssh=dict(host="", user="", port=0, remote_sync_dir=""),
    tasks=[
        dict(
            prompt_en="pick up the orange block and place it on the green plate.",
            milestones=(),
        ),
    ],
)

control_channel = dict(
    enabled=True,
    host="127.0.0.1",
    port=5757,
)
