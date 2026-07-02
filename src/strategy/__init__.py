"""Inference scheduling strategies for robot action chunking.

Provides the strategies that decide how policy-produced action chunks are
fetched and executed: synchronous step-by-step (SYNC), background-thread
prefetch with smoothing (ASYNC_PREFETCH), and Real-Time Chunking (RTC).
Also exposes the action buffers used by the async/RTC loops to smooth and
serve individual actions. Submodules are auto-imported on package load so
their strategy builders register with STRATEGY_REGISTRY.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
for _importer, _modname, _ispkg in pkgutil.iter_modules([_PACKAGE_DIR]):
    if not _ispkg:
        importlib.import_module(f"strategy.{_modname}")
