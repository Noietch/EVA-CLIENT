#!/usr/bin/env python3
"""Continuously print events from the USB foot pedal.

The default device is the stable udev symlink for the 0483:5750 pedal, so the
script keeps working when the USB port changes and the event number changes.
"""

from __future__ import annotations

import argparse
import array
import fcntl
import os
import select
import struct
import sys
import time
from datetime import datetime

DEFAULT_DEVICE = "/dev/input/by-id/usb-0483_5750-if01-event-kbd"
EV_KEY = 0x01
KEY_BITMAP_BYTES = 96
EVIOCGKEY = 0x80000000 | (KEY_BITMAP_BYTES << 16) | (ord("E") << 8) | 0x18
EVENT = struct.Struct("@llHHi")

KEY_NAMES = {
    28: "KEY_ENTER",
}
VALUE_NAMES = {
    0: "RELEASED",
    1: "PRESSED",
    2: "REPEATED",
}


class CollectionController:
    """Translate pedal presses into collection actions.

    A save is held for ``double_press_window`` seconds before being emitted, so
    a second press can turn it into a cancel action instead.
    """

    def __init__(self, double_press_window: float) -> None:
        self.collecting = False
        self.save_deadline: float | None = None
        self.double_press_window = double_press_window

    def on_press(self, now: float) -> str | None:
        if not self.collecting:
            self.collecting = True
            return "开始采集"

        if self.save_deadline is not None and now <= self.save_deadline:
            self.save_deadline = None
            self.collecting = False
            return "取消"

        self.save_deadline = now + self.double_press_window
        return None

    def on_timeout(self, now: float) -> str | None:
        if self.save_deadline is None or now < self.save_deadline:
            return None

        self.save_deadline = None
        self.collecting = False
        return "保存"

    def timeout_seconds(self, now: float) -> float | None:
        if self.save_deadline is None:
            return None
        return max(0.0, self.save_deadline - now)


def pressed_key_codes(fd: int) -> list[int]:
    """Return all key codes that are currently held down."""
    bitmap = array.array("B", [0]) * KEY_BITMAP_BYTES
    fcntl.ioctl(fd, EVIOCGKEY, bitmap, True)
    return [
        byte_index * 8 + bit
        for byte_index, byte in enumerate(bitmap)
        for bit in range(8)
        if byte & (1 << bit)
    ]


def key_name(code: int) -> str:
    return KEY_NAMES.get(code, f"KEY_{code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Linux input event device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--double-press-window",
        type=float,
        default=1.0,
        help="Seconds allowed between two presses to cancel (default: 1.0)",
    )
    args = parser.parse_args()
    if args.double_press_window <= 0:
        parser.error("--double-press-window must be greater than zero")

    try:
        fd = os.open(args.device, os.O_RDONLY)
    except OSError as exc:
        print(f"Unable to open {args.device}: {exc}", file=sys.stderr)
        return 1

    try:
        initial_pressed = pressed_key_codes(fd)
        initial_state = "PRESSED" if initial_pressed else "RELEASED"
        print(f"监控设备：{args.device}", flush=True)
        print(
            f"硬件初始状态：{initial_state}；pressed_codes={initial_pressed}",
            flush=True,
        )
        print("采集状态：未采集", flush=True)
        controller = CollectionController(args.double_press_window)

        while True:
            now = time.monotonic()
            timeout = controller.timeout_seconds(now)
            ready, _, _ = select.select([fd], [], [], timeout)

            if not ready:
                action = controller.on_timeout(time.monotonic())
                if action is not None:
                    print(action, flush=True)
                    print("采集状态：未采集", flush=True)
                continue

            raw = os.read(fd, EVENT.size)
            if len(raw) != EVENT.size:
                raise OSError(f"short input event read: {len(raw)} bytes")

            seconds, microseconds, event_type, code, value = EVENT.unpack(raw)
            if event_type != EV_KEY:
                continue

            timestamp = datetime.fromtimestamp(
                seconds + microseconds / 1_000_000
            ).isoformat(timespec="milliseconds")
            state = VALUE_NAMES.get(value, str(value))
            print(
                f"{timestamp} type=EV_KEY(1) code={code} "
                f"name={key_name(code)} value={value} state={state}",
                flush=True,
            )

            if value != 1:
                continue

            action = controller.on_press(time.monotonic())
            if action is not None:
                print(action, flush=True)
                if action == "取消":
                    print("采集状态：未采集", flush=True)
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    except OSError as exc:
        print(f"Input device error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
