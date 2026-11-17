"""Dual Agilex Piper collection task-set defaults.

Configure Hugging Face for task-set/QC sync and optionally add one delivery
backend (SFTP, S3 or loopback) in the matching ``*.local.py`` override.
"""

_base_ = ["dual_agilex_piper.py"]

collection = dict(
    task_set_dir="datasets/data_collection/task_sets",
    storage=dict(
        # Three independent 30 FPS Astra streams need a small scheduling margin.
        image_skew_tolerance_sec=0.025,
    ),
)
