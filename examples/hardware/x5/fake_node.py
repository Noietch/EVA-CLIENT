#!/usr/bin/env python3
"""Configurable-response ZMQ fake node for the dual ARX X5."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.hardware.fake_common import FakeRobotNode, main  # noqa: E402


class X5FakeRobotNode(FakeRobotNode):
    """ARX X5 binding for the shared ZMQ fake plant."""


if __name__ == "__main__":
    main("arx_x5", node_type=X5FakeRobotNode)
