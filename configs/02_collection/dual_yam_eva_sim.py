"""Dual YAM VR collection against EVA-SIM without rendered images."""

_base_ = ["dual_yam.py"]


collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/larybenchv2.v1.2/train_v1.2",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.supplement",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_01_logistics",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_02_logistics",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_03_retail",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_04_home_dining",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_05_home_kitchen",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_06_office",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_07_hotel_service",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_08_industrial_repair",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_09_public_cleaning",
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.composite_10_teaching_meeting",
    ],
    storage=dict(
        log_dir="work_dirs/collection/eva_sim_dual_yam/",
    ),
    sim_recording_url="http://127.0.0.1:8093",
    schema=dict(cameras={}),
    # Preserve raw recordings for offline rendering on a dedicated machine.
    render_service=dict(url=""),
)


dashboard = dict(
    raw_roots=["datasets/data_collection/datasets/eva_sim/dual_yam"],
    raw_roots_only=True,
)


transport = dict(
    type="zmq",
    sub_endpoint="tcp://127.0.0.1:5555",
    pub_endpoint="tcp://127.0.0.1:5556",
    disabled_cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
)
