"""Minimal Chrome DevTools driver for human-like browser E2E input."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


class Cdp:
    """Synchronous Chrome DevTools Protocol client over a raw WebSocket."""

    def __init__(self, websocket_url: str) -> None:
        rest = websocket_url.removeprefix("ws://")
        host_port, path = rest.split("/", 1)
        host, port_text = host_port.split(":", 1)
        self._socket = socket.create_connection((host, int(port_text)), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode())
        response = self._socket.recv(4096)
        if b" 101 " not in response:
            raise RuntimeError(response.decode(errors="replace"))
        self._next_id = 1

    def close(self) -> None:
        self._socket.close()

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 20.0,
    ) -> Any:
        message_id = self._next_id
        self._next_id += 1
        self._send({"id": message_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = self._recv()
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result")
        raise TimeoutError(method)

    def eval(self, expression: str, timeout: float = 20.0) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
                "userGesture": True,
            },
            timeout=timeout,
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def _send(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        header = bytearray([0x81])
        if len(raw) < 126:
            header.append(0x80 | len(raw))
        elif len(raw) < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", len(raw)))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", len(raw)))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(raw[index] ^ mask[index % 4] for index in range(len(raw)))
        self._socket.sendall(bytes(header) + masked)

    def _recv(self) -> dict[str, Any]:
        while True:
            first = self._read_exact(2)
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            masked = bool(first[1] & 0x80)
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length)
            if masked:
                payload = bytes(payload[index] ^ mask[index % 4] for index in range(len(payload)))
            if opcode == 1:
                return json.loads(payload.decode())
            if opcode == 8:
                raise RuntimeError("Chrome DevTools WebSocket closed")
            if opcode == 9:
                self._socket.sendall(bytes([0x8A, len(payload)]) + payload)

    def _read_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise RuntimeError("Chrome DevTools socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def start_chrome(headless: bool = False) -> tuple[subprocess.Popen[Any], Cdp, Path]:
    """Start an isolated Chrome and return its process, CDP connection, and profile."""
    profile = Path(tempfile.mkdtemp(prefix="eva-rl-e2e-chrome-"))
    command = [
        "google-chrome",
        "--no-sandbox",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1600,1000",
        f"--user-data-dir={profile}",
        "--remote-debugging-port=0",
        "about:blank",
    ]
    if headless:
        command.insert(1, "--headless=new")
    chrome = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    websocket_url = None
    assert chrome.stderr is not None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        line = chrome.stderr.readline()
        if "DevTools listening on " in line:
            websocket_url = line.strip().split("DevTools listening on ", 1)[1]
            break
    if websocket_url is None:
        chrome.terminate()
        raise RuntimeError("Chrome did not expose a DevTools WebSocket")
    host_port = websocket_url.removeprefix("ws://").split("/", 1)[0]
    request = urllib.request.Request(f"http://{host_port}/json/new?about:blank", method="PUT")
    with urllib.request.urlopen(request, timeout=10) as response:
        page_url = json.loads(response.read().decode())["webSocketDebuggerUrl"]
    return chrome, Cdp(page_url), profile


def wait_for(cdp: Cdp, expression: str, timeout: float = 10.0) -> Any:
    """Poll a JavaScript expression until it returns a truthy value."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cdp.eval(expression)
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError(expression)


def element_center(cdp: Cdp, selector: str) -> dict[str, float]:
    """Return the hit-tested center of one visible DOM element."""
    return cdp.eval(
        f"""
        (() => {{
          const element = document.querySelector({json.dumps(selector)});
          if (!element) throw new Error({json.dumps(f'missing selector: {selector}')});
          element.scrollIntoView({{ block: "center", inline: "center" }});
          const rect = element.getBoundingClientRect();
          const x = rect.left + rect.width / 2;
          const y = rect.top + rect.height / 2;
          const hit = document.elementFromPoint(x, y);
          if (!hit || (hit !== element && !element.contains(hit))) {{
            throw new Error(`selector center is covered: ${{hit && hit.outerHTML}}`);
          }}
          return {{ x, y, width: rect.width, height: rect.height }};
        }})()
        """
    )


def dispatch_mouse_click(cdp: Cdp, x: float, y: float) -> None:
    """Dispatch the same mouse move/down/up sequence as a physical click."""
    for event_type in ("mouseMoved", "mousePressed", "mouseReleased"):
        cdp.call(
            "Input.dispatchMouseEvent",
            {
                "type": event_type,
                "x": x,
                "y": y,
                "button": "left",
                "buttons": 1 if event_type == "mousePressed" else 0,
                "clickCount": 1,
            },
        )


def human_click(cdp: Cdp, selector: str) -> None:
    """Hit-test and click one element through CDP mouse input."""
    target = element_center(cdp, selector)
    dispatch_mouse_click(cdp, target["x"], target["y"])


def human_select_first(cdp: Cdp, selector: str) -> None:
    """Choose the first enabled option through mouse focus and keyboard input."""
    human_click(cdp, selector)
    for key, code in (("ArrowDown", "ArrowDown"), ("Enter", "Enter")):
        for event_type in ("keyDown", "keyUp"):
            cdp.call(
                "Input.dispatchKeyEvent",
                {"type": event_type, "key": key, "code": code},
            )


def human_scrub(cdp: Cdp, selector: str, fraction: float) -> None:
    """Click a range input at the requested fractional position."""
    target = element_center(cdp, selector)
    x = target["x"] - target["width"] / 2 + target["width"] * fraction
    dispatch_mouse_click(cdp, x, target["y"])
