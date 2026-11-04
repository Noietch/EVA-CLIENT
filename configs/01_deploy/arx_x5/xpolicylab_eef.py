"""ARX X5: XPolicyLab websocket backend with EEF-pose actions.

Inherits the EEFPose obs/action spaces from ``openpi_eef.py``; only the policy
backend changes. The endpoint here is a default — an eval checkpoint overrides
host/port per model.
"""

_base_ = ["openpi_eef.py"]

transport = dict(type="zmq", image_mode="on_demand")

policy = dict(
    type="xpolicylab",
    host="33.29.36.208",
    port=8416,
    backend_options=dict(
        action_type="ee",
        arm_dims=[7, 7],
        gripper_dims=[1, 1],
        request_timeout_s=120.0,
        connect_timeout_s=10.0,
        handshake_timeout_s=30.0,
    ),
)

inference_cfg = dict(
    inference_rate=3,
    publish_rate=40,
    setup_warmup_chunks=1,
)

inference_strategies = {
    "async": dict(args=dict(latency_k=0)),
}
