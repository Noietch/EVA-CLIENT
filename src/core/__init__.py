"""Core package — configuration, application loop, and recording.

Houses the client's runtime building blocks: config schema and YAML loading
(``core.config``), the application loop, command handlers, and web console
(``core.app``), and episode recording (``core.recorder``).
Top-level symbols are re-exported lazily (PEP 562) to avoid import cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.app.run import run
    from core.config import ConfigDict, load_config
    from core.types import ControlMode, Observation

__all__ = [
    "ConfigDict",
    "ControlMode",
    "Observation",
    "load_config",
    "run",
]

# Lazy re-export (PEP 562): importing a core submodule (e.g. core.registry) must
# not pull in the whole app stack, which would close an import cycle.
_LAZY = {
    "ConfigDict": "core.config",
    "load_config": "core.config",
    "run": "core.app.run",
    "ControlMode": "core.types",
    "Observation": "core.types",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'core' has no attribute '{name}'")
    import importlib

    return getattr(importlib.import_module(module_path), name)
