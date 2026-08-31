"""ARX X5 WebXR collection using the normalized local task-set directory."""

_base_ = ["arx_x5_vr.py"]

collection = dict(
    task_set_dir="work_dirs/tasks_set",
    task_set_name="ArxKine_PnP_DivObj_Norm_Sngl_Base_v1_scene_1_20260828",
)
