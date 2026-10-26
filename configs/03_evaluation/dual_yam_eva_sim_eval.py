"""Dual YAM TCP evaluation through the XPolicyLab websocket backend."""

_base_ = ["../01_deploy/dual_yam/openpi_tcp.py", "../00_base/evaluation.py"]

console = dict(initial_tab="eval")
policy = dict(
    _delete_=True,
    type="xpolicylab",
    host="127.0.0.1",
    port=19000,
    backend_options=dict(
        action_type="ee",
        arm_dims=[7, 7],
        gripper_dims=[1, 1],
        request_timeout_s=120.0,
        connect_timeout_s=10.0,
        handshake_timeout_s=30.0,
    ),
)
transport = dict(type="zmq", image_mode="on_demand", convert_bgr_to_rgb=False, resize=False)
control_channel = dict(enabled=True)
inference_cfg = dict(max_debug_time_s=1800.0, setup_warmup_chunks=1)
inference_strategies = {
    "async": dict(args=dict(latency_k=0)),
    "sync": dict(args=dict(execute_horizon=12)),
}
