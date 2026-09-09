"""Dual Piper: XPolicyLab websocket backend with joint-space actions."""

_base_ = ["_base.py"]

transport = dict(type="zmq", image_mode="on_demand")

policy = dict(
    type="xpolicylab",
    host="127.0.0.1",
    port=19000,
    backend_options=dict(
        xpolicylab_root="../EVA-XPolicyLab",
        env_cfg_type="arx_x5",
        action_type="joint",
        request_timeout_s=120.0,
        connect_timeout_s=10.0,
        handshake_timeout_s=30.0,
    ),
)

inference_cfg = dict(
    inference_rate=3,
    publish_rate=30,
    setup_warmup_chunks=1,
)

inference_strategies = {
    "async": dict(args=dict(latency_k=0)),
    "sync": dict(args=dict(execute_horizon=12)),
}
