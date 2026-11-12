"""ARX X5: asynchronous-mode eval against the served v1.4 XPolicyLab fleet.

Inference overlaps execution: a background loop requests a new chunk at
inference_rate, drops its first latency_k steps (set in the deploy presets) and
blends it into the remaining actions. Task set, trials and checkpoint endpoints
are inherited unchanged from the sync preset.
"""

_base_ = ["arx_x5_larybench2_xpolicylab_eval.py"]

eval_cfg = dict(inference_strategy="async")
