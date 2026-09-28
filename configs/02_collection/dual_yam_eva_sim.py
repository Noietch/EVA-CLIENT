"""Dual YAM VR collection against EVA-SIM without rendered images."""

_base_ = ["dual_yam.py"]


collection = dict(
    task_set_dir=[
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_flip",
        "datasets/data_collection/task_sets/larybench2_20260916_eva_sim_dual_yam_fold",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_handover",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_insert_withdraw",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_open_close",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_pickplace",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_pickplace_last200",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_pour",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_press",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_pull",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_push",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_rotate",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_scoop",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_sweep",
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_wipe",
    ],
    storage=dict(
        log_dir="work_dirs/collection/eva_sim_dual_yam/",
    ),
    render_service=dict(url="http://127.0.0.1:8430"),
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
