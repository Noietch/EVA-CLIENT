"""Collect v1.2 eval layouts against EVA-SIM."""

_base_ = ["dual_yam_eva_sim.py"]

collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.supplement",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_01_logistics",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_02_logistics",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_03_retail",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_04_home_dining",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_05_home_kitchen",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_06_office",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_07_hotel_service",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_08_industrial_repair",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_09_public_cleaning",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.eval.composite_10_teaching_meeting",
    ],
    task_set_name="",
)
