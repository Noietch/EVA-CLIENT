#!/usr/bin/env python3
"""Fake ur5e hardware node — thin wrapper over examples/hardware/fake_common.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_common import main  # noqa: E402

if __name__ == "__main__":
    main("ur5e")
