"""Dual ARX X5 collection tasks and storage settings."""

_base_ = ["../00_base/defaults.py", "../00_base/collection.py"]

console = dict(
    initial_tab="collect",
)

collection = dict(
    storage=dict(log_dir="work_dirs/collection/arx_x5_vr", image_skew_tolerance_sec=0.035),
    tasks=dict(
        ArxKine_PnP_DivObj_Norm_Sngl_Base_v1_scene_1_20260828=[
            ("pick up the yellow cup and place it on the green plate with left hand.", 1),
            ("pick up the yellow spoon and place it on the green plate with left hand.", 1),
            ("pick up the yellow block and place it on the green plate with left hand.", 1),
            ("pick up the mango and place it on the green plate with left hand.", 1),
            ("pick up the small tape roll and place it on the green plate with left hand.", 1),
            ("pick up the gray cup and place it on the green plate with left hand.", 1),
        ],
    ),
)

dashboard = dict(
    raw_roots=["datasets/data_collection/datasets/real_robot/arx_x5"],
    raw_roots_only=True,
)

robot = dict(type="arx_x5")
