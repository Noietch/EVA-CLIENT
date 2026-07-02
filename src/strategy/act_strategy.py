"""ACT temporal-ensembling inference strategy — fixed-Hz background loop with
EnsembleActionBuffer averaging overlapping chunks via exp(-exp_weight_m * i)
weights (ACT-style temporal aggregation).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

from core.registry import STRATEGY_REGISTRY
from strategy.action_buffer import EnsembleActionBuffer
from strategy.base_strategy import BackgroundLoopInferStrategy

logger = logging.getLogger(__name__)


@STRATEGY_REGISTRY.register("ActEnsembleInferStrategy")
@dataclasses.dataclass
class ActEnsembleInferStrategy(BackgroundLoopInferStrategy):
    """ACT temporal-ensembling strategy: background loop + EnsembleActionBuffer.

    Stores every overlapping chunk and blends per-step predictions with
    exp(-exp_weight_m * i) weights. Larger exp_weight_m -> oldest prediction
    dominates; smaller -> flatter average.

    Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
    Low-Cost Hardware" (ACT), arXiv:2304.13705 (https://arxiv.org/abs/2304.13705).
    """

    exp_weight_m: float = 0.01
    _thread_name: str = dataclasses.field(default="eva-act-loop", init=False)
    _log_label: str = dataclasses.field(default="act", init=False)

    tune_fields: ClassVar[list[dict]] = [
        {"key": "execute_horizon", "label": "exec steps", "min": 1, "step": 1},
        {"key": "exp_weight_m", "label": "exp m", "min": 0, "step": 0.01},
    ]

    def _make_buffer(self) -> EnsembleActionBuffer:
        """Use temporal-ensembling buffer averaging overlapping per-step predictions."""
        return EnsembleActionBuffer(exp_weight_m=self.exp_weight_m)

    def __post_init__(self) -> None:
        self.exp_weight_m = max(0.0, float(self.exp_weight_m))
        super().__post_init__()
        logger.info(
            "Using act ensembling inference strategy: execute_horizon=%s "
            "exp_weight_m=%.3f inference_rate=%.1f",
            self.execute_horizon if self.execute_horizon is not None else "all",
            self.exp_weight_m,
            self.inference_rate,
        )
