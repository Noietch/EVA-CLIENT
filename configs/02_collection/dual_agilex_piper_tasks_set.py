"""Dual Agilex Piper collection task-set defaults.

Choose exactly one machine-local sync backend in the matching ``*.local.py``
override: Hugging Face or SFTP.
"""

_base_ = ["dual_agilex_piper.py"]

collection = dict(
    task_set_dir="datasets/data_collection/task_sets",
    storage=dict(
        # Three independent 30 FPS Astra streams need a small scheduling margin.
        image_skew_tolerance_sec=0.025,
    ),
)
