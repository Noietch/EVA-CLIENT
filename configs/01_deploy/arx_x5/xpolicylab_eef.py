"""ARX X5: XPolicyLab websocket backend with EEF-pose actions.

Inherits the EEFPose obs/action spaces from ``openpi_eef.py``; only the policy
backend changes. The endpoint here is a default — an eval checkpoint overrides
host/port per model.
"""

_base_ = ["openpi_eef.py"]

transport = dict(type="zmq", image_mode="on_demand")

# The rest-pose weight is only passed to the IK solver together with explicit EEF
# links. A 6-DoF arm has no null space for it to act in, so the default of 1.0
# trades pose accuracy for staying near home: replayed v1.4 demonstrations miss
# their fingertip target by 2.5 mm (median) to 9 mm when the gripper closes. 0.1,
# as in the simulation EEF presets, keeps that under 0.2 mm.
robot = dict(eef_links=["link6", "link6"], ik_reg_weight=0.1)

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

# Actions are published at 30 Hz, the frame rate of the v1.4 demonstrations the
# served checkpoints were fine-tuned on, so each chunk step lasts as long as in
# training (40 Hz replayed every chunk 1.33x faster). An eval checkpoint adopts
# this preset as its whole config, so the rate has to be set here.
inference_cfg = dict(
    inference_rate=3,
    publish_rate=30,
    setup_warmup_chunks=1,
)

# Async drops the first 8 steps (0.27 s at 30 Hz) of each new chunk, the part
# that already elapsed while the policy was inferring; with 0 every chunk
# replayed from its stale first step and re-opened grippers that had closed.
inference_strategies = {
    "async": dict(args=dict(latency_k=8)),
}
