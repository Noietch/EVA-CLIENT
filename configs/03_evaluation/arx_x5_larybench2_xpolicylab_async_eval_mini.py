"""ARX X5: asynchronous-mode eval on the 85-case LaryBench2 miniset.

Same as ``arx_x5_larybench2_xpolicylab_async_eval.py`` on the miniset task set;
latency_k comes from the deploy presets.
"""

_base_ = ["arx_x5_larybench2_xpolicylab_eval_mini.py"]

eval_cfg = dict(inference_strategy="async")
