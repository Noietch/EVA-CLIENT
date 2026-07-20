#!/usr/bin/env python3
"""Use a USB foot pedal to control EVA collection and its simulation task.

The first pedal press starts collection. Once armed, one press sends ``RESET``
(cancel) after one second; two presses inside that second send ``NEXT`` (save)
instead. Each action reaches EVA through its public ``/api/operator_action``
endpoint at ``http://localhost:8081``, then sends the matching scene command to
the simulation control API at ``http://localhost:8093``. The latter avoids
routing pedal commands through Viser's 3D WebSocket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import select
import struct
import sys
import time
from collections.abc import Callable
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

DEFAULT_DEVICE = "/dev/input/by-id/usb-0483_5750-if01-event-kbd"
DEFAULT_CONTROL_URL = "http://localhost:8093"
DEFAULT_EVA_URL = "http://localhost:8081"
EV_KEY = 0x01
KEY_PRESSED = 1
KEY_ENTER = 28
EVENT = struct.Struct("@llHHi")


class WebCollectionController:
    """Map pedal presses to EVA operator intents and simulator actions."""

    def __init__(self, double_press_window: float, emit_action: Callable[[str], None]) -> None:
        if double_press_window <= 0:
            raise ValueError("double_press_window must be greater than zero")
        self._double_press_window = double_press_window
        self._emit_action = emit_action
        self._armed = False
        self._next_deadline: float | None = None

    def on_press(self, now: float) -> str | None:
        if not self._armed:
            self._armed = True
            self._emit_action("start")
            return "start"
        if self._next_deadline is not None and now <= self._next_deadline:
            self._next_deadline = None
            self._armed = False
            self._emit_action("accept")
            return "accept"
        self._next_deadline = now + self._double_press_window
        return None

    def on_timeout(self, now: float) -> str | None:
        if self._next_deadline is None or now < self._next_deadline:
            return None
        self._next_deadline = None
        self._armed = False
        self._emit_action("cancel")
        return "cancel"

    def timeout_seconds(self, now: float) -> float | None:
        if self._next_deadline is None:
            return None
        return max(0.0, self._next_deadline - now)


def api_endpoint(base_url: str, path: str) -> str:
    """Return one API endpoint under an HTTP(S) base URL."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("control URL must begin with http:// or https://")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def request_api(
    base_url: str,
    path: str,
    timeout: float,
    body: dict[str, object] | None = None,
) -> int:
    """Perform an API request and return its HTTP status code."""
    method = "GET" if path in {"/api/health", "/api/status"} else "POST"
    payload = None if body is None else json.dumps(body).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    request = Request(api_endpoint(base_url, path), data=payload, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        response.read()
        return response.status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE, help=f"Input device (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--key-code", type=int, default=KEY_ENTER, help="Linux input key code (default: 28)"
    )
    parser.add_argument(
        "--control-url",
        default=DEFAULT_CONTROL_URL,
        help=f"Simulation control API URL (default: {DEFAULT_CONTROL_URL})",
    )
    parser.add_argument(
        "--eva-url",
        default=DEFAULT_EVA_URL,
        help=f"EVA collection console URL (default: {DEFAULT_EVA_URL})",
    )
    parser.add_argument(
        "--double-press-window",
        type=float,
        default=1.0,
        help="Seconds allowed between two presses for save/NEXT (default: 1.0)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=2.0, help="Control API request timeout"
    )
    args = parser.parse_args()
    if args.double_press_window <= 0:
        parser.error("--double-press-window must be greater than zero")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than zero")
    return args


async def run(args: argparse.Namespace) -> int:
    print(f"正在连接控制 API：{args.control_url}", flush=True)
    try:
        status = await asyncio.to_thread(
            request_api, args.control_url, "/api/health", args.request_timeout
        )
        if status != 200:
            raise RuntimeError(f"unexpected health status: {status}")
        status = await asyncio.to_thread(
            request_api, args.eva_url, "/api/status", args.request_timeout
        )
        if status != 200:
            raise RuntimeError(f"unexpected EVA status: {status}")
    except (OSError, RuntimeError, URLError) as error:
        print(f"Unable to connect to control API: {error}", file=sys.stderr)
        return 1

    try:
        fd = os.open(args.device, os.O_RDONLY)
    except OSError as error:
        print(f"Unable to open {args.device}: {error}", file=sys.stderr)
        return 1

    actions: asyncio.Queue[str] = asyncio.Queue()
    controller = WebCollectionController(args.double_press_window, actions.put_nowait)
    print(f"监控设备：{args.device}；键码={args.key_code}", flush=True)
    print("控制 API 已就绪：NEXT=/api/next，RESET=/api/reset", flush=True)
    print(f"EVA operator-action API 已就绪：{args.eva_url}/api/operator_action", flush=True)
    print("状态：未开始；首次按下开始采集", flush=True)
    print("采集中：单次按下 1 秒后取消/RESET；1 秒内双按保存/NEXT", flush=True)
    try:
        while True:
            ready, _, _ = await asyncio.to_thread(
                select.select, [fd], [], [], controller.timeout_seconds(time.monotonic())
            )
            if not ready:
                controller.on_timeout(time.monotonic())
            else:
                raw = os.read(fd, EVENT.size)
                if len(raw) != EVENT.size:
                    raise OSError(f"short input event read: {len(raw)} bytes")
                _, _, event_type, code, value = EVENT.unpack(raw)
                if event_type == EV_KEY and code == args.key_code and value == KEY_PRESSED:
                    action = controller.on_press(time.monotonic())
            while not actions.empty():
                action = actions.get_nowait()
                status = await asyncio.to_thread(
                    request_api,
                    args.eva_url,
                    "/api/operator_action",
                    args.request_timeout,
                    {"intent": action},
                )
                if status != 200:
                    raise RuntimeError(f"unexpected operator_action status: {status}")
                if action == "start":
                    print("开始采集：EVA 已接收", flush=True)
                    print("状态：采集中", flush=True)
                    continue

                simulator_action = "next" if action == "accept" else "reset"
                status = await asyncio.to_thread(
                    request_api,
                    args.control_url,
                    f"/api/{simulator_action}",
                    args.request_timeout,
                )
                if status != 202:
                    raise RuntimeError(f"unexpected {simulator_action} status: {status}")
                label = "保存/NEXT" if action == "accept" else "取消/RESET"
                print(f"{label}：EVA 与远端仿真均已接收", flush=True)
                print("状态：未开始", flush=True)
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    except (OSError, RuntimeError, URLError) as error:
        print(f"Control error: {error}", file=sys.stderr)
        return 1
    finally:
        os.close(fd)


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
