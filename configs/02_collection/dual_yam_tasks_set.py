"""Dual YAM collection with selectable task-set datasets."""

_base_ = ["dual_yam.py"]

collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_cut",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_flip",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_handover",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_insert_withdraw",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_open_close",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_pickplace",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_pour",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_press",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_pull",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_push",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_rotate",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_scoop",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_sweep",
        "datasets/data_collection/task_sets/larybench2_20260908_dual_yam_wipe",
    ],
)
