"""ARX X5 collection with optional Hugging Face sync and one delivery remote.

Hugging Face sync may be combined with one of ``sftp`` or ``s3`` for export
delivery. Credentials live in the matching ``*.local.py``.
"""

_base_ = ["arx_x5.py"]

collection = dict(
    task_set_dir="datasets/data_collection/task_sets/larybench2_20260901_arx_x5",
    task_set_name="larybench2.0-20260901-arx_x5",
    storage=dict(
        huggingface=dict(
            repo_id="example-org/example-dataset",
            token="",
            proxy="http://proxy.example.com:8080",
            revision="main",
        ),
        # sftp=dict(
        #     host="sftp.example.com",
        #     port=22,
        #     user="collector",
        #     identity_file="~/.ssh/id_ed25519",
        #     remote_dir="/srv/datasets",
        # ),
        # s3=dict(
        #     endpoint="DATASET_ENDPOINT.example.com",
        #     bucket="omni-video",
        #     prefix="embodied_raw_files",
        #     sign_service="https://SIGN_SERVICE.example.com",
        #     secure=False,
        # ),
    ),
)
