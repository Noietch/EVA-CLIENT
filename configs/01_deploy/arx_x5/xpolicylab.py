"""ARX X5: XPolicyLab websocket backend with joint-space (qpos) actions.

The fallback endpoint below points at the cloud XPolicyLab server used for
LaryBench2 evaluation; an eval preset can still override host/port per
checkpoint.
"""

_base_ = ["_base.py"]

transport = dict(type="zmq", image_mode="on_demand")

policy = dict(
    type="xpolicylab",
    host="33.29.36.208",
    port=8416,
    backend_options=dict(
        action_type="joint",
        arm_dims=[6, 6],
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
