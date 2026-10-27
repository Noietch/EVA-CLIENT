"""Dual YAM: connect to examples/policy/dreamlabs over the OpenPI WebSocket backend."""

_base_ = ["_base.py"]

transport = dict(image_height=176, image_width=320, image_layout="hwc", resize_pad=False)

policy = [
    dict(
        name="dream-1-beta",
        type="openpi",
        host="127.0.0.1",
        port=5558,
        backend_options=dict(),
    ),
]

inference_cfg = dict(
    publish_rate=10, inference_rate=1.0, default_strategy="async", setup_warmup_chunks=1
)

inference_strategies = {
    "sync": dict(
        args=dict(
            sync_wait_ignore_gripper=True,
            sync_wait_threshold=9999.0,
            sync_wait_max_ticks=5,  # 0.5 s at the configured 10 Hz action rate.
        )
    ),
    "async": dict(args=dict(latency_k=8, blend_gripper=False)),
}
