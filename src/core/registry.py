"""Domain Registry instances.

The Registry class lives in core.cfg (vendored from eva/engine, mmengine
style). This module just declares the per-domain instances and keeps the import
surface stable for the rest of the project.

Backends self-register with ``@<REGISTRY>.register("type")`` (positional-args
builder style used across this project) and callers build with
``<REGISTRY>.build("type", *args, **kwargs)``. Use ``.build_from_cfg(cfg)`` for
the engine-native dict(type=...) call form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from core.cfg import Registry as _BaseRegistry

if TYPE_CHECKING:
    from core.app.handlers.space import ActionSpace
    from policy_client.base import PolicyClient
    from robots.base import Robot
    from strategy.base_strategy import BaseInferStrategy
    from transport.base import TransportBridge

T = TypeVar("T")
C = TypeVar("C")


class Registry(_BaseRegistry, Generic[T]):
    """Thin generic-typed subclass of the vendored mmengine Registry.

    Adds nothing beyond a TypeVar parameter so static type checkers can verify
    the element type of each domain registry (T = Robot, PolicyClient,
    BaseInferStrategy, etc.).
    """

    def register(self, key: str) -> Callable[[C], C]:
        """Type-preserving override of the vendored decorator.

        The decorator returns the registered class unchanged, so static
        checkers should see the original type rather than the base class's
        erased ``Type | Callable``.
        """
        return super().register(key)  # type: ignore[return-value]

    def register_client(self, key: str) -> Callable[[C], C]:
        """Register a class under ``key`` via its ``from_config`` classmethod.

        The decorator sits on the class itself (``type`` and class read together),
        but what ``build(key, config, ctx)`` invokes is ``cls.from_config(config,
        ctx)`` — so each client keeps a clean, directly-constructible ``__init__``
        while the registry build path stays uniform. The class is returned
        unchanged.
        """

        def decorator(cls: C) -> C:
            self._register_module(cls.from_config, key, force=False)  # type: ignore[attr-defined]
            return cls

        return decorator



ROBOT_REGISTRY: Registry[Robot] = Registry("robot")
TRANSPORT_REGISTRY: Registry[TransportBridge] = Registry("transport")
STRATEGY_REGISTRY: Registry[BaseInferStrategy] = Registry("strategy")
POLICY_REGISTRY: Registry[PolicyClient] = Registry("policy")
SPACE_REGISTRY: Registry[ActionSpace] = Registry("space")

# Generic registry for standalone callables (no shared base class). Side-effect
# helpers like the eval SSH forwarder register here and are built by name.
FUNCTIONS: Registry[Callable[..., object]] = Registry("function")
