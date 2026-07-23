#!/usr/bin/env python3
"""Fake I2RT YAM node for EVA development without SDK, CAN, or D405 hardware."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_common import main  # noqa: E402

if __name__ == "__main__":
    main(os.environ.get("I2RT_ROBOT", "i2rt_dual_yam"))
