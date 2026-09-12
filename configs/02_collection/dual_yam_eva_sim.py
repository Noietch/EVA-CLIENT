"""Dual YAM VR collection against EVA-SIM without rendered images."""

_base_ = ["dual_yam.py"]


collection = dict(
    task_set_dir=(
        "datasets/data_collection/task_sets/"
        "larybench2_20260908_dual_yam_pickplace"
    ),
    storage=dict(
        log_dir="work_dirs/collection/eva_sim_dual_yam/",
    ),
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
