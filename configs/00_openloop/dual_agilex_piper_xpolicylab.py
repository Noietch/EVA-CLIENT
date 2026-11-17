"""Offline open-loop evaluation through the official XPolicyLab server.

Expects a sibling checkout of https://github.com/XPolicyLab/XPolicyLab with the
policy installed and its deploy.yml pointing at your checkpoint; replace the
dataset and output paths with your own.
"""

_base_ = ["../01_deploy/dual_agilex_piper/xpolicylab.py"]

transport = dict(
    type="dataset",
    dataset_dir="datasets/<your_lerobot_dataset>",
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
    output_dir="work_dirs/open_loop/dual_agilex_piper_xpolicylab/episode_000000",
    execute_horizon=None,
    max_steps=None,
    startup_timeout_s=1200,
    server=dict(
        cwd="../XPolicyLab",
        command=[
            "python",
            "setup_policy_server.py",
            "--config_path",
            "policy/<POLICY>/deploy.yml",
            "--port",
            "{port}",
        ],
    ),
)
