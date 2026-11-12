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

# Actions are published at 30 Hz, the frame rate of the v1.4 demonstrations the
# served checkpoints were fine-tuned on, so each chunk step lasts as long as in
# training (40 Hz replayed every chunk 1.33x faster). An eval checkpoint adopts
# this preset as its whole config, so the rate has to be set here.
inference_cfg = dict(
    inference_rate=3,
    publish_rate=10,
    setup_warmup_chunks=1,
)

# Async drops the first 8 steps (0.27 s at 30 Hz) of each new chunk, the part
# that already elapsed while the policy was inferring; with 0 every chunk
# replayed from its stale first step and re-opened grippers that had closed.
inference_strategies = {
    "async": dict(args=dict(latency_k=8)),
}
