"""Robots package — one self-contained ``Robot`` subclass per bundled robot.

Each robot lives in its own directory under ``zoo/`` as a single ``__init__.py``
holding a ``Robot`` subclass (declarative structure + ``build_kinematics``) decorated
with ``@ROBOT_REGISTRY.register(name)``, alongside its ``assets/``. Importing each
subpackage below triggers its registration.
"""

from __future__ import annotations

from robots.zoo import (
    agibot_g2,  # noqa: F401
    agilex_piper,  # noqa: F401
    arx_r5,  # noqa: F401
    arx_x5,  # noqa: F401
    dual_franka,  # noqa: F401
    r1_lite,  # noqa: F401
    ur5e,  # noqa: F401
)
