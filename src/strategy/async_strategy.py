"""Async linear-overlap inference strategy — runs a background inference thread
at a fixed Hz rate, pushing action chunks into a StreamActionBuffer (linear
overlap smoothing). The main loop pops individual actions from the buffer for
continuous robot control.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

from core.registry import STRATEGY_REGISTRY
from strategy.base_strategy import BackgroundLoopInferStrategy

logger = logging.getLogger(__name__)


@STRATEGY_REGISTRY.register("AsyncLinearOverlapInferStrategy")
@dataclasses.dataclass
class AsyncLinearOverlapInferStrategy(BackgroundLoopInferStrategy):
    """Async inference strategy with fixed-Hz continuous inference loop and
    StreamActionBuffer linear-overlap smoothing.

    YAML keys (in addition to BaseInferStrategy + BackgroundLoopInferStrategy):
    exp_weight_m (>= 0, decay for the temporal-ensembling variant).

    Reference: arXiv:2602.09021 (https://arxiv.org/abs/2602.09021).
    """

    exp_weight_m: float = 0.01
    _thread_name: str = dataclasses.field(default="eva-async-loop", init=False)
    _log_label: str = dataclasses.field(default="async", init=False)

    tune_fields: ClassVar[list[dict]] = [
        {"key": "execute_horizon", "label": "exec steps", "min": 1, "step": 1},
        {"key": "latency_k", "label": "latency k", "min": 0, "step": 1},
    ]

    def __post_init__(self) -> None:
        self.exp_weight_m = max(0.0, float(self.exp_weight_m))
        super().__post_init__()
        logger.info(
            "Using async inference strategy: execute_horizon=%s latency_k=%s inference_rate=%.1f",
            self.execute_horizon if self.execute_horizon is not None else "all",
            self.latency_k if self.latency_k is not None else "unset",
            self.inference_rate,
        )
