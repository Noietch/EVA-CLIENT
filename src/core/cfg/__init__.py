"""Vendored mmengine-style Config + Registry, pruned for the EVA-CLIENT robot
inference client (no model build, dataset, distributed, or checkpoint loading).

Source: eva/engine/config + registry (https://github.com/open-mmlab/mmengine).
"""

from .config import Config, ConfigDict, DictAction, read_base
from .registry import Registry, build_from_cfg

__all__ = [
    "Config",
    "ConfigDict",
    "DictAction",
    "Registry",
    "build_from_cfg",
    "read_base",
]
