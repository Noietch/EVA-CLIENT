"""ARX X5 collection using the dataset service's task-plan directory."""

_base_ = ["arx_x5.py"]

collection = dict(
    task_set_dir="datasets/data_collection/task_sets/larybench2_20260901_arx_x5",
    task_set_name="larybench2.0-20260901-arx_x5",
    storage=dict(
        s3=dict(
            endpoint="DATASET_ENDPOINT.example.com",
            bucket="omni-video",
            prefix="embodied_raw_files",
            sign_service="https://SIGN_SERVICE.example.com",
            secure=False,
        ),
    ),
)
