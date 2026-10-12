"""Dual YAM VR collection for the ten EVA-SIM composite task sets."""

_base_ = ["dual_yam_eva_sim.py"]

collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_02_logistics",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_01_logistics",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_05_home_kitchen",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_07_hotel_service",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_04_home_dining",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_08_industrial_repair",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_03_retail",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_10_teaching_meeting",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_09_public_cleaning",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_06_office",
    ],
)
