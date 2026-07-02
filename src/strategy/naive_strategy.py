"""Naive async inference strategy — fixed-Hz background loop with a
NaiveActionBuffer doing latency-aligned whole-chunk replacement (no overlap
smoothing).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

from core.registry import STRATEGY_REGISTRY
from strategy.action_buffer import NaiveActionBuffer
from strategy.base_strategy import BackgroundLoopInferStrategy

logger = logging.getLogger(__name__)


@STRATEGY_REGISTRY.register("NaiveAsyncInferStrategy")
@dataclasses.dataclass
class NaiveAsyncInferStrategy(BackgroundLoopInferStrategy):
    """Naive async strategy: background loop + whole-chunk replacement buffer.

    Same fields as BackgroundLoopInferStrategy. Each new chunk wholly replaces
    whatever remained in the buffer after a latency-aligned front trim.
    """

    _thread_name: str = dataclasses.field(default="eva-naive-loop", init=False)
    _log_label: str = dataclasses.field(default="naive", init=False)

    tune_fields: ClassVar[list[dict]] = [
        {"key": "execute_horizon", "label": "exec steps", "min": 1, "step": 1},
        {"key": "latency_k", "label": "latency k", "min": 0, "step": 1},
    ]

    def _make_buffer(self) -> NaiveActionBuffer:
        """Use whole-chunk-replacement buffer instead of linear-overlap smoothing."""
        return NaiveActionBuffer()

    def __post_init__(self) -> None:
        super().__post_init__()
        logger.info(
            "Using naive async inference strategy: "
            "execute_horizon=%s latency_k=%s inference_rate=%.1f",
            self.execute_horizon if self.execute_horizon is not None else "all",
            self.latency_k if self.latency_k is not None else "unset",
            self.inference_rate,
        )
