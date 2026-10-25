"""Dual YAM: connect to examples/policy/dreamlabs over the OpenPI WebSocket backend."""

_base_ = ["_base.py"]

transport = dict(image_height=176, image_width=320, image_layout="hwc", resize_pad=False)

policy = dict(
    type="openpi",
    host="127.0.0.1",
    port=5555,
    backend_options=dict(),
)

inference_cfg = dict(publish_rate=20, setup_warmup_chunks=1)

inference_strategies = {
    "sync": dict(args=dict(execute_horizon=48)),
    "async": dict(args=dict(latency_k=0, execute_horizon=30)),
}
