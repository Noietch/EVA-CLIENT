"""ARX X5: synchronous-mode eval against the served v1.4 XPolicyLab fleet.

The blocking sync strategy executes a whole policy chunk before acquiring the
next observation, so the control schedule is deterministic instead of
overlapping inference with execution. Task set, trials and checkpoint endpoints
are inherited unchanged from the async preset.
"""

_base_ = ["arx_x5_larybench2_xpolicylab_eval.py"]

eval_cfg = dict(inference_strategy="sync")

# Empty args keep the entire action chunk returned by the policy.
inference_strategies = {
    "sync": dict(type="BaseInferStrategy", args={}),
}
