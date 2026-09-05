"""Offline LingBot-VLA open-loop evaluation through the XPolicyLab client."""

_base_ = ["../01_deploy/dual_agilex_piper/xpolicylab.py"]

transport = dict(
    type="dataset",
    dataset_dir=(
        "../EVA-XPolicyLab/datasets/test/dual_agilex_piper_eva_sim_vr_0806_sim_example/LingBot_VLA"
    ),
    episode_id=0,
    convert_bgr_to_rgb=True,
    image_height=224,
    image_width=224,
    resize_pad=False,
    image_layout="hwc",
    dataset_keys=dict(
        state_key="observation.state",
        action_key="action",
        video_keys=dict(
            cam_high="observation.images.cam_high",
            cam_left_wrist="observation.images.cam_left_wrist",
            cam_right_wrist="observation.images.cam_right_wrist",
        ),
    ),
)

openloop = dict(
    output_dir=(
        "../EVA-XPolicyLab/policy/LingBot_VLA/work_dirs/"
        "dual_agilex_piper_vr0806_3ep_bs256_s2073_r2/open_loop/"
        "dual_agilex_piper_eva_sim_vr_0806_sim_example/episode_000000/eva_client"
    ),
    execute_horizon=None,
    max_steps=None,
    startup_timeout_s=1200,
    server=dict(
        cwd="../EVA-XPolicyLab",
        command=[
            "bash",
            "policy/LingBot_VLA/sft/serve.sh",
            (
                "policy/LingBot_VLA/work_dirs/"
                "dual_agilex_piper_vr0806_3ep_bs256_s2073_r2/checkpoints/"
                "global_step_2073/hf_ckpt"
            ),
            "{prompt}",
            "{port}",
            "0",
            (
                "policy/LingBot_VLA/work_dirs/"
                "dual_agilex_piper_vr0806_3ep_bs256_s2073_r2/norm_stats.json"
            ),
        ],
    ),
)
