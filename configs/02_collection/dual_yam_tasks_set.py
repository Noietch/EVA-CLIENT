"""ARX X5 collection using the dataset service's task-plan directory."""

_base_ = ["dual_yam.py"]

collection = dict(
    task_set_dir="datasets/data_collection/task_tests/larybench2_20260908_dual_yam",
    task_set_name="larybench2.0-20260908-dual_yam",
)
