"""Dual YAM collection with selectable task-set datasets."""

_base_ = ["dual_yam.py"]

collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/larybench2_20260908",
    ],
)
