"""policy_client package — policy backends and the abstract client interface.

Re-exports the core types from ``policy_client.base`` and, on import, eagerly
loads every sibling backend module so each one runs its ``POLICY_REGISTRY``
registration side effect (the ``base`` module is already imported above).
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from policy_client.base import (
    DatasetReplayPolicyClient,
    PolicyBuildContext,
    PolicyClient,
    PolicyConnectionError,
    PolicyRequestError,
    RandomPolicyClient,
)

__all__ = [
    "DatasetReplayPolicyClient",
    "PolicyBuildContext",
    "PolicyClient",
    "PolicyConnectionError",
    "PolicyRequestError",
    "RandomPolicyClient",
]

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
for _importer, _modname, _ispkg in pkgutil.iter_modules([_PACKAGE_DIR]):
    if not _ispkg and _modname != "base":
        importlib.import_module(f"policy_client.{_modname}")
