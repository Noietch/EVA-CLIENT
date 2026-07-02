"""RTC (Real-Time Chunking) inference strategy — runs a background inference thread
at a fixed Hz rate with the RTC protocol (prev_action feedback for guided diffusion
alignment). Uses StreamActionBuffer for latency-compensated linear overlap smoothing.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

from core.registry import STRATEGY_REGISTRY
from strategy.base_strategy import BackgroundLoopInferStrategy

logger = logging.getLogger(__name__)


@STRATEGY_REGISTRY.register("RtcInferStrategy")
@dataclasses.dataclass
class RtcInferStrategy(BackgroundLoopInferStrategy):
    """RTC (Real-Time Chunking) strategy: fixed-Hz background loop with the RTC
    protocol (handled by RtcOpenPiPolicyClient, not here) and StreamActionBuffer
    linear-overlap smoothing.

    YAML keys are inherited unchanged from BackgroundLoopInferStrategy
    (execute_horizon, sync_wait_ignore_gripper, inference_rate, latency_k).

    Reference: Black et al., "Real-Time Execution of Action Chunking Flow
    Policies" (RTC), arXiv:2506.07339 (https://arxiv.org/abs/2506.07339).
    """

    _thread_name: str = dataclasses.field(default="eva-rtc-loop", init=False)
    _log_label: str = dataclasses.field(default="rtc", init=False)

    tune_fields: ClassVar[list[dict]] = [
        {"key": "execute_horizon", "label": "exec steps", "min": 1, "step": 1},
        {"key": "latency_k", "label": "latency k", "min": 0, "step": 1},
    ]

    def __post_init__(self) -> None:
        super().__post_init__()
        logger.info(
            "Using rtc inference strategy: execute_horizon=%s latency_k=%s",
            self.execute_horizon if self.execute_horizon is not None else "all",
            self.latency_k if self.latency_k is not None else "unset",
        )
