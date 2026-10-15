"""ARX X5 policy observations and actions at the URDF TCP, in each arm base."""

_base_ = ["openpi_eef.py"]

robot = dict(eef_links=["tcp", "tcp"], ik_reg_weight=0.1)

inference_cfg = dict(
    obs_space=dict(rotation_order="wxyz"),
    action_space=dict(rotation_order="wxyz"),
)
