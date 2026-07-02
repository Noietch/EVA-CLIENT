"""Transport package — the layer that exchanges observations and actions with a
robot source.

A transport abstracts where observations come from and where actions go: a live
robot (zmq middleware, ros1/ros2 topics), a recorded LeRobot dataset (replay),
or a degraded stand-in (offline). All backends implement the
TransportBridge contract in base.py. Importing this package eagerly imports every
sibling module so each one registers its builder in TRANSPORT_REGISTRY.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from transport.base import TransportBridge

__all__ = ["TransportBridge"]

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
for _importer, _modname, _ispkg in pkgutil.iter_modules([_PACKAGE_DIR]):
    if not _ispkg and _modname != "base":
        importlib.import_module(f"transport.{_modname}")
