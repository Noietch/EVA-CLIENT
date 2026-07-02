from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def ag95_to_raw(value: float, raw_open: int, raw_close: int) -> int:
    clipped = float(np.clip(value, 0.0, 1.0))
    return int(round(raw_close + (raw_open - raw_close) * clipped))


def ag95_raw_to_eva(raw_value: float | None, raw_open: int, raw_close: int) -> float:
    if raw_value is None or raw_open == raw_close:
        return 0.0
    value = (float(raw_close) - float(raw_value)) / (float(raw_close) - float(raw_open))
    return float(np.clip(value, 0.0, 1.0))


class DahuanAG95Controller:
    def __init__(self, port: str, raw_open: int = 1000, raw_close: int = 0) -> None:
        self.port = port
        self.raw_open = raw_open
        self.raw_close = raw_close
        self._gripper: Any | None = None

    def connect(self) -> None:
        from pyDHgripper import AG95

        gripper = AG95(self.port)
        gripper.init_feedback()
        gripper.set_force(100)
        self._gripper = gripper

    def set_normalized(self, value: float) -> None:
        gripper = self._gripper
        if gripper is None:
            raise RuntimeError("AG95 gripper is not connected")
        gripper.set_pos(val=ag95_to_raw(value, self.raw_open, self.raw_close))

    def read_normalized(self) -> float:
        gripper = self._gripper
        if gripper is None:
            raise RuntimeError("AG95 gripper is not connected")
        return ag95_raw_to_eva(gripper.read_pos(), self.raw_open, self.raw_close)

    def close(self) -> None:
        gripper = self._gripper
        if gripper is None:
            return
        close_gripper = getattr(gripper, "close", None)
        if close_gripper is not None:
            close_gripper()
        else:
            serial_connection = getattr(gripper, "ser", None)
            close_serial = getattr(serial_connection, "close", None)
            if close_serial is not None:
                close_serial()
        self._gripper = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a Dahuan AG95 gripper.")
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--raw-open", type=int, default=1000)
    parser.add_argument("--raw-close", type=int, default=0)
    parser.add_argument("--command", type=float, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    gripper = DahuanAG95Controller(args.port, args.raw_open, args.raw_close)
    gripper.connect()
    try:
        if args.command is not None:
            gripper.set_normalized(args.command)
        logger.info("AG95 normalized position: %.3f", gripper.read_normalized())
    finally:
        gripper.close()


if __name__ == "__main__":
    main()
