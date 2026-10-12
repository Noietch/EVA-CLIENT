"""Dual YAM VR collection for the ten EVA-SIM composite task sets."""

_base_ = ["dual_yam_eva_sim.py"]

collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/composite_02_logistics",
        "datasets/data_collection/task_sets/composite_01_logistics",
        "datasets/data_collection/task_sets/composite_05_home_kitchen",
        "datasets/data_collection/task_sets/composite_07_hotel_service",
        "datasets/data_collection/task_sets/composite_04_home_dining",
        "datasets/data_collection/task_sets/composite_08_industrial_repair",
        "datasets/data_collection/task_sets/composite_03_retail",
        "datasets/data_collection/task_sets/composite_10_teaching_meeting",
        "datasets/data_collection/task_sets/composite_09_public_cleaning",
        "datasets/data_collection/task_sets/composite_06_office",
    ],
)
