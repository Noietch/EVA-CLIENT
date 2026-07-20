"""Critic client package and backend registration."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from critic_client.base import (
    CriticBuildContext,
    CriticClient,
    CriticConnectionError,
    CriticRequestError,
    MockCriticClient,
)

__all__ = [
    "CriticBuildContext",
    "CriticClient",
    "CriticConnectionError",
    "CriticRequestError",
    "MockCriticClient",
]

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
for _importer, _modname, _ispkg in pkgutil.iter_modules([_PACKAGE_DIR]):
    if not _ispkg and _modname != "base":
        importlib.import_module(f"critic_client.{_modname}")
