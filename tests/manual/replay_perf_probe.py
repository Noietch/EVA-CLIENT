from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class Cdp:
    def __init__(self, ws_url: str) -> None:
        rest = ws_url.removeprefix("ws://")
        host_port, path = rest.split("/", 1)
        host, port_text = host_port.split(":", 1)
        self._sock = socket.create_connection((host, int(port_text)), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._sock.sendall(request.encode())
        response = self._sock.recv(4096)
        if b" 101 " not in response:
            raise RuntimeError(response.decode(errors="replace"))
        self._next_id = 1

    def close(self) -> None:
        self._sock.close()

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 20) -> Any:
        msg_id = self._next_id
        self._next_id += 1
        self._send({"id": msg_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._recv()
            if msg.get("id") != msg_id:
                continue
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg.get("result")
        raise TimeoutError(method)

    def eval(self, expression: str, timeout: float = 20) -> Any:
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
        masked = bytes(raw[i] ^ mask[i % 4] for i in range(len(raw)))
        self._sock.sendall(bytes(header) + masked)

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
                payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
            if opcode == 1:
                return json.loads(payload.decode())
            if opcode == 8:
                raise RuntimeError("websocket closed")
            if opcode == 9:
                self._send_pong(payload)

    def _send_pong(self, payload: bytes) -> None:
        self._sock.sendall(bytes([0x8A, len(payload)]) + payload)

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise RuntimeError("socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


PROBE_JS = r"""
(() => {
  function createProbe(ignoredVideos) {
    const state = {
      start: 0,
      videos: {},
      events: [],
      ignoredVideos,
      rafGaps: [],
      syncSamples: [],
      urdfFrames: [],
      longTasks: [],
      originalRaf: null,
      originalSceneApply: null,
      originalResultSetFrame: null,
      currentNodes: {},
    };

    const installHooks = () => {
      if (!state.originalRaf) {
        state.originalRaf = window.requestAnimationFrame;
        const requestAnimationFrame = state.originalRaf.bind(window);
        window.requestAnimationFrame = (callback) => requestAnimationFrame((stamp) => {
          const now = performance.now();
          if (state.start > 0 && state.lastRaf != null) state.rafGaps.push(now - state.lastRaf);
          state.lastRaf = now;
          const range = Array.from(document.querySelectorAll("#scrub-range, #tp-seek"))
            .find((node) => getComputedStyle(node).display !== "none" && !node.disabled);
          const cursor = range ? Number(range.value) : null;
          const videos = Array.from(document.querySelectorAll(
            "#cam-strip video.cam, #tp-cam-strip video.tp-cam",
          ));
          const times = videos.map((video) => Number(video.currentTime)).filter(Number.isFinite);
          const urdf = Number(window.__evaReplayAppliedFrame ?? window.__evaResultUrdfAppliedFrame);
          state.syncSamples.push({
            t: now - state.start,
            cursor: Number.isFinite(cursor) ? cursor : null,
            urdf: Number.isFinite(urdf) ? urdf : null,
            times,
          });
          return callback(stamp);
        });
      }
      if (!state.originalSceneApply && window.Scene3D &&
          typeof window.Scene3D.applyTransformFrame === "function") {
        state.originalSceneApply = window.Scene3D.applyTransformFrame;
        window.Scene3D.applyTransformFrame = function(...args) {
          state.urdfFrames.push({ t: performance.now() - state.start, frame: Number(args[4]) });
          return state.originalSceneApply.apply(this, args);
        };
      }
      if (!state.originalResultSetFrame && window.ReplayScene &&
          typeof window.ReplayScene.setFrame === "function") {
        state.originalResultSetFrame = window.ReplayScene.setFrame;
        window.ReplayScene.setFrame = function(...args) {
          state.urdfFrames.push({ t: performance.now() - state.start, frame: Number(args[1]) });
          return state.originalResultSetFrame.apply(this, args);
        };
      }
    };
    const longTaskObserver = window.PerformanceObserver && new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => state.longTasks.push({
        t: entry.startTime - state.start, duration: entry.duration,
      }));
    });
    if (longTaskObserver) {
      try { longTaskObserver.observe({ type: "longtask", buffered: true }); } catch (e) {}
    }

    const markVisible = (node, rec) => {
      if (rec.visible !== null || rec.firstFrame === null) return;
      const cell = node.closest(".cam-cell, .tp-cam-cell");
      const rect = node.getBoundingClientRect();
      const overlay = cell ? cell.querySelector(".cam-loading") : null;
      const overlayRect = overlay ? overlay.getBoundingClientRect() : null;
      const overlayVisible = overlay && getComputedStyle(overlay).display !== "none" &&
        overlayRect && overlayRect.width > 0 && overlayRect.height > 0;
      const style = node ? getComputedStyle(node) : null;
      const hidden = !cell || overlayVisible || cell.style.display === "none" ||
        !style || style.display === "none" || style.visibility === "hidden" ||
        rect.width <= 0 || rect.height <= 0;
      if (hidden) {
        requestAnimationFrame(() => markVisible(node, rec));
        return;
      }
      rec.visible = performance.now() - state.start;
      state.events.push([
        "visible", rec.key, rec.visible, node.readyState || null, node.currentTime || null,
      ]);
    };

    const watch = (node) => {
      if (state.ignoredVideos.has(node)) return;
      const baseKey = node.dataset.key || `video-${Object.keys(state.videos).length}`;
      const key = node.tagName === "IMG" ? `poster:${baseKey}` : baseKey;
      const previous = state.videos[key];
      if (previous && state.currentNodes[key] !== node) delete state.videos[key];
      if (state.videos[key]) {
        markVisible(node, state.videos[key]);
        return;
      }
      const rec = {
        key,
        tag: node.tagName,
        inserted: performance.now() - state.start,
        firstFrame: null,
        visible: null,
        canplay: null,
        play: null,
        playing: null,
        waiting: [],
        stalled: [],
        seeking: [],
        seeked: [],
        timeupdate: [],
        videoFrames: [],
      };
      state.videos[key] = rec;
      state.currentNodes[key] = node;
      const mark = (name) => {
        rec[name] = performance.now() - state.start;
        state.events.push([
          name, key, rec[name], node.readyState || null, node.currentTime || null,
        ]);
      };
      const push = (name) => {
        rec[name].push([
          performance.now() - state.start, node.readyState || null, node.currentTime || null,
        ]);
        const at = rec[name][rec[name].length - 1][0];
        state.events.push([name, key, at, node.readyState || null, node.currentTime || null]);
      };
      if (node.tagName === "IMG") {
        const imageDone = () => {
          if (rec.firstFrame !== null || !node.complete || node.naturalWidth <= 0) return;
          rec.firstFrame = performance.now() - state.start;
          state.events.push(["firstFrame", key, rec.firstFrame, null, null]);
          requestAnimationFrame(() => markVisible(node, rec));
        };
        node.addEventListener("load", imageDone, { once: true });
        imageDone();
        return;
      }
      node.addEventListener("canplay", () => { if (rec.canplay === null) mark("canplay"); });
      node.addEventListener("play", () => { if (rec.play === null) mark("play"); });
      node.addEventListener("playing", () => { if (rec.playing === null) mark("playing"); });
      node.addEventListener("waiting", () => push("waiting"));
      node.addEventListener("stalled", () => push("stalled"));
      node.addEventListener("seeking", () => push("seeking"));
      node.addEventListener("seeked", () => push("seeked"));
      node.addEventListener("timeupdate", () => {
        if (rec.timeupdate.length < 80) push("timeupdate");
      });
      const frameDone = () => {
        if (rec.firstFrame !== null) return;
        rec.firstFrame = performance.now() - state.start;
        state.events.push(["firstFrame", key, rec.firstFrame, node.readyState, node.currentTime]);
        requestAnimationFrame(() => markVisible(node, rec));
      };
      const markReadyFrame = () => {
        if (rec.firstFrame !== null || node.readyState < 2) return;
        frameDone();
      };
      if (typeof node.requestVideoFrameCallback === "function") {
        const mediaFrame = (stamp) => {
          rec.videoFrames.push([stamp - state.start, node.currentTime || 0]);
          if (rec.videoFrames.length < 4000 && !state.stopped) {
            node.requestVideoFrameCallback(mediaFrame);
          }
        };
        node.requestVideoFrameCallback(mediaFrame);
        try {
          node.requestVideoFrameCallback(frameDone);
        } catch (e) {
          requestAnimationFrame(frameDone);
        }
      } else {
        node.addEventListener("loadeddata", frameDone, { once: true });
      }
      markReadyFrame();
    };

    const scan = () => {
    const mediaSelector = "#cam-strip video.cam, #cam-strip img.cam-poster, " +
      "#tp-cam-strip video.tp-cam";
      document.querySelectorAll(mediaSelector).forEach(watch);
      document.querySelectorAll(mediaSelector).forEach((node) => {
        const key = node.dataset.key || "";
        const rec = state.videos[key];
        if (rec) {
          if (rec.firstFrame === null && node.tagName === "IMG" &&
              node.complete && node.naturalWidth > 0) {
            rec.firstFrame = performance.now() - state.start;
            state.events.push(["firstFrame", key, rec.firstFrame, null, null]);
          }
          if (rec.firstFrame === null && node.readyState >= 2) {
            rec.firstFrame = performance.now() - state.start;
            state.events.push([
              "firstFrame", key, rec.firstFrame, node.readyState, node.currentTime,
            ]);
          }
          markVisible(node, rec);
        }
      });
    };

    const observer = new MutationObserver(scan);
    [document.getElementById("cam-strip"), document.getElementById("tp-cam-strip")]
      .filter(Boolean)
      .forEach((host) => observer.observe(host, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["class", "style"],
      }));
    state.stop = () => observer.disconnect();
    state.stop = () => {
      state.stopped = true;
      observer.disconnect();
      if (state.originalRaf) window.requestAnimationFrame = state.originalRaf;
      if (state.originalSceneApply && window.Scene3D) {
        window.Scene3D.applyTransformFrame = state.originalSceneApply;
      }
      if (state.originalResultSetFrame && window.ReplayScene) {
        window.ReplayScene.setFrame = state.originalResultSetFrame;
      }
      if (longTaskObserver) longTaskObserver.disconnect();
    };
    state.scan = () => { installHooks(); scan(); };
    return state;
  }

  function startProbeAtPoint(x, y, label) {
    const hit = document.elementFromPoint(x, y);
    if (!hit) {
      throw new Error(`click target ${label} hit none`);
    }
    const previous = window.__replayPerfProbe;
    if (previous && typeof previous.stop === "function") previous.stop();
    if (typeof performance.clearResourceTimings === "function") {
      performance.clearResourceTimings();
    }
    const ignoredVideos = new WeakSet(Array.from(document.querySelectorAll(
      "#cam-strip video.cam, #cam-strip img.cam-poster, #tp-cam-strip video.tp-cam",
    )));
    const state = createProbe(ignoredVideos);
    window.__replayPerfProbe = state;
    state.start = performance.now();
    state.lastRaf = state.start;
    state.scan();
    state.events.push(["mouseStart", label || "*", 0, null, null]);
    return { x, y };
  }

  function startProbeForSelector(selector) {
    const el = document.querySelector(selector);
    if (!el) throw new Error(`missing selector: ${selector}`);
    el.scrollIntoView({ block: "center", inline: "center" });
    const rect = el.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    if (!hit || (hit !== el && !el.contains(hit))) {
      const name = hit
        ? `${hit.tagName}#${hit.id || ""}.${hit.className || ""}`
        : "none";
      throw new Error(`selector ${selector} center hit ${name}`);
    }
    return {
      ...startProbeAtPoint(x, y, selector),
      width: rect.width,
      height: rect.height,
    };
  }

  window.__startReplayPerfProbe = startProbeForSelector;
  window.__startReplayPerfProbeAtPoint = startProbeAtPoint;
})();
"""


def dispatch_mouse_click(cdp: Cdp, x: float, y: float) -> None:
    for event_type in ("mouseMoved", "mousePressed", "mouseReleased"):
        params: dict[str, Any] = {
            "type": event_type,
            "x": x,
            "y": y,
            "button": "left",
            "buttons": 1 if event_type == "mousePressed" else 0,
            "clickCount": 1,
        }
        cdp.call("Input.dispatchMouseEvent", params)


def element_center(cdp: Cdp, selector: str) -> dict[str, Any]:
    return cdp.eval(
        f"""
        (() => {{
          const el = document.querySelector({json.dumps(selector)});
          if (!el) throw new Error({json.dumps(f"missing selector: {selector}")});
          el.scrollIntoView({{ block: "center", inline: "center" }});
          const rect = el.getBoundingClientRect();
          const x = rect.left + rect.width / 2;
          const y = rect.top + rect.height / 2;
          const hit = document.elementFromPoint(x, y);
          if (!hit || (hit !== el && !el.contains(hit))) {{
            const name = hit
              ? `${{hit.tagName}}#${{hit.id || ""}}.${{hit.className || ""}}`
              : "none";
            throw new Error(`selector {selector} center hit ${{name}}`);
          }}
          return {{
            x,
            y,
            width: rect.width,
            height: rect.height,
          }};
        }})()
        """
    )


def human_click(cdp: Cdp, selector: str) -> None:
    target = element_center(cdp, selector)
    dispatch_mouse_click(cdp, target["x"], target["y"])


def human_click_and_start_probe(cdp: Cdp, selector: str) -> None:
    cdp.eval(PROBE_JS)
    target = cdp.eval(
        f"""
        (() => {{
          const startProbeForSelector = window.__startReplayPerfProbe;
          if (typeof startProbeForSelector !== "function") {{
            throw new Error("replay perf probe unavailable");
          }}
          return startProbeForSelector({json.dumps(selector)});
        }})()
        """
    )
    dispatch_mouse_click(cdp, target["x"], target["y"])


def human_click_target_and_start_probe(cdp: Cdp, target: dict[str, Any], label: str) -> None:
    cdp.eval(PROBE_JS)
    point = cdp.eval(
        f"""
        (() => {{
          const startProbeAtPoint = window.__startReplayPerfProbeAtPoint;
          if (typeof startProbeAtPoint !== "function") {{
            throw new Error("replay perf probe unavailable");
          }}
          return startProbeAtPoint({float(target["x"])}, {float(target["y"])}, {json.dumps(label)});
        }})()
        """
    )
    dispatch_mouse_click(cdp, point["x"], point["y"])


def human_click_target(cdp: Cdp, target: dict[str, Any]) -> None:
    """Dispatch a browser mouse click at a previously hit-tested target."""
    dispatch_mouse_click(cdp, float(target["x"]), float(target["y"]))


def select_clickable_tile(
    cdp: Cdp, container_selector: str, index: int, item_selector: str = ".collect-tile.replayable"
) -> dict[str, Any]:
    return cdp.eval(
        f"""
        (() => {{
          const host = document.querySelector({json.dumps(container_selector)});
          if (!host) throw new Error({json.dumps(f"missing selector: {container_selector}")});
          const tiles = Array.from(host.querySelectorAll({json.dumps(item_selector)}));
          if (!tiles.length) throw new Error(`no replayable tiles in {container_selector}`);
          const wanted = Math.min({int(index)}, tiles.length - 1);
          const ordered = tiles.slice(wanted).concat(tiles.slice(0, wanted));
          const fractions = [0.5, 0.35, 0.65, 0.2, 0.8];
          const misses = [];
          for (const tile of ordered) {{
            tile.scrollIntoView({{ block: "center", inline: "center" }});
            const rect = tile.getBoundingClientRect();
            const tileMisses = [];
            for (const fx of fractions) {{
              for (const fy of fractions) {{
                const x = rect.left + rect.width * fx;
                const y = rect.top + rect.height * fy;
                const hit = document.elementFromPoint(x, y);
                if (hit && (hit === tile || tile.contains(hit))) {{
                  return {{
                    x,
                    y,
                    index: tiles.indexOf(tile),
                    count: tiles.length,
                    title: tile.title || "",
                    width: rect.width,
                    height: rect.height,
                  }};
                }}
                if (tileMisses.length < 4) {{
                  tileMisses.push(hit
                    ? `${{hit.tagName}}#${{hit.id || ""}}.${{hit.className || ""}}`
                    : "none");
                }}
              }}
            }}
            misses.push({{
              index: tiles.indexOf(tile),
              rect: [rect.left, rect.top, rect.width, rect.height],
              hits: tileMisses,
            }});
          }}
          throw new Error(
            `no hit-tested replayable tile in {container_selector}: ${{JSON.stringify(misses)}}`,
          );
        }})()
        """
    )


def capture_screenshot(cdp: Cdp, name: str) -> str:
    result = cdp.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
    path = Path(tempfile.gettempdir()) / f"eva-replay-perf-{name}-{int(time.time() * 1000)}.png"
    path.write_bytes(base64.b64decode(result["data"]))
    return str(path)


def http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def start_chrome() -> tuple[subprocess.Popen[Any], Cdp]:
    user_dir = tempfile.mkdtemp(prefix="eva-replay-perf-chrome-")
    chrome = subprocess.Popen(
        [
            "google-chrome",
            "--no-sandbox",
            "--autoplay-policy=no-user-gesture-required",
            "--window-size=1600,1000",
            f"--user-data-dir={user_dir}",
            "--remote-debugging-port=0",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    ws_url = None
    assert chrome.stderr is not None
    deadline = time.time() + 15
    while time.time() < deadline:
        line = chrome.stderr.readline()
        if "DevTools listening on " in line:
            ws_url = line.strip().split("DevTools listening on ", 1)[1]
            break
    if ws_url is None:
        chrome.terminate()
        raise RuntimeError("Chrome did not expose DevTools websocket")
    host_port = ws_url.removeprefix("ws://").split("/", 1)[0]
    request = urllib.request.Request(f"http://{host_port}/json/new?about:blank", method="PUT")
    with urllib.request.urlopen(request, timeout=10) as resp:
        page_ws_url = json.loads(resp.read().decode())["webSocketDebuggerUrl"]
    return chrome, Cdp(page_ws_url)


def wait_for(cdp: Cdp, expr: str, timeout: float = 10) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cdp.eval(expr)
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError(expr)


def set_value(cdp: Cdp, selector: str, value: str) -> None:
    error_message = json.dumps(f"missing selector: {selector}")
    cdp.eval(
        f"""
        (() => {{
          const el = document.querySelector({json.dumps(selector)});
          if (!el) throw new Error({error_message});
          el.value = {json.dumps(value)};
          el.dispatchEvent(new Event("input", {{ bubbles: true }}));
          el.dispatchEvent(new Event("change", {{ bubbles: true }}));
          return true;
        }})()
        """
    )


def start_replay_playback(cdp: Cdp) -> None:
    cdp.eval(
        """
        (() => {
          if (typeof window.replayToggle !== "function") {
            throw new Error("window.replayToggle unavailable");
          }
          const p = window.__replayPerfProbe;
          if (p) {
            p.playCommand = performance.now() - p.start;
            p.events.push(["playCommand", "*", p.playCommand, null, null]);
          }
          window.replayToggle();
          return true;
        })()
        """
    )


def rapid_tile_switch(
    cdp: Cdp,
    container_selector: str,
    name: str,
    timeout: float,
    play_seconds: float,
) -> dict[str, Any] | None:
    """Exercise A→B→A with real clicks and measure the final A playback."""
    count = cdp.eval(
        f"document.querySelectorAll({json.dumps(container_selector)} + "
        "' .collect-tile.replayable').length"
    )
    if not count or int(count) < 2:
        return None
    first = select_clickable_tile(cdp, container_selector, 0)
    second = select_clickable_tile(cdp, container_selector, 1)
    human_click_target_and_start_probe(cdp, first, f"{name}-a")
    time.sleep(0.1)
    human_click_target(cdp, second)
    time.sleep(0.1)
    final = select_clickable_tile(cdp, container_selector, 0)
    human_click_target(cdp, final)
    return capture_measurement(cdp, f"{name}-rapid-a-b-a", timeout, play_seconds)


PAGE_READY_EXPR = """
document.readyState === "complete" && !!document.querySelector("#replay-dataset-input")
"""

VIDEOS_VISIBLE_READY_EXPR = """
(() => {
  const p = window.__replayPerfProbe;
  if (!p) return false;
  if (typeof p.scan === "function") p.scan();
  const cells = Array.from(document.querySelectorAll(
    "#cam-strip .cam-cell, #tp-cam-strip .tp-cam-cell",
  ))
    .filter((cell) => getComputedStyle(cell).display !== "none");
  const paintedCells = cells.filter((cell) => {
    const overlay = cell.querySelector(".cam-loading");
    const overlayRect = overlay ? overlay.getBoundingClientRect() : null;
    const overlayVisible = overlay && getComputedStyle(overlay).display !== "none" &&
      overlayRect && overlayRect.width > 0 && overlayRect.height > 0;
    if (overlayVisible) return false;
    return Array.from(cell.querySelectorAll(
      "img.cam-poster, video.cam-video, video.tp-cam",
    )).some((node) => {
      const rect = node.getBoundingClientRect();
      const style = getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden" ||
          rect.width <= 0 || rect.height <= 0) {
        return false;
      }
      if (node.tagName === "IMG") return node.complete && node.naturalWidth > 0;
      return node.readyState >= 2;
    });
  });
  const videos = Object.values(p.videos).filter((v) => v.tag === "VIDEO");
  return paintedCells.length >= 3 && videos.length >= 3 &&
    videos.every((v) => v.visible !== null);
})()
"""

SCRUB_PLAY_VISIBLE_EXPR = """
(() => {
  const button = document.querySelector("#scrub-play");
  return button && getComputedStyle(button).display !== "none";
})()
"""

REPLAY_LOAD_READY_EXPR = """
typeof document.querySelector("#replay-b-episode-confirm").onclick === "function"
"""

REPLAY_NEXT_READY_EXPR = """
(() => {
  const button = document.querySelector("#replay-b-episode-next");
  return button && !button.disabled;
})()
"""

STAGE_VIDEOS_PLAYING_EXPR = """
(() => {
  const videos = Array.from(document.querySelectorAll("#cam-strip video.cam"));
  return videos.length >= 3 && videos.every((v) => !v.paused && v.currentTime > 0);
})()
"""

COLLECT_REPLAYABLE_READY_EXPR = """
document.querySelectorAll("#collect-queue-tiles .collect-tile.replayable").length > 0
"""

COLLECT_REPLAY_BUTTON_READY_EXPR = """
(() => {
  const button = document.querySelector("#review-return-live");
  return button && button.style.display === "none";
})()
"""

COLLECT_REPLAY_STOP_READY_EXPR = """
(() => {
  const button = document.querySelector("#review-return-live");
  return button && button.style.display !== "none";
})()
"""

ROLLOUT_REPLAYABLE_READY_EXPR = """
document.querySelectorAll("#rollout-save-queue-tiles .collect-tile.replayable").length > 0
"""

RL_REPLAYABLE_READY_EXPR = """
document.querySelectorAll("#rl-save-tiles .collect-tile.replayable").length > 0
"""

RESULT_TRIAL_READY_EXPR = """
document.querySelectorAll(".rv-trials .eval-trial.scored").length > 0
"""

RESULT_POP_READY_EXPR = """
(() => {
  const pop = document.querySelector("#trial-pop");
  return pop && pop.classList.contains("open");
})()
"""


def tab_active_expr(tab: str) -> str:
    return f"""
    (() => {{
      const tab = document.querySelector('.tab[data-tab="{tab}"]');
      return tab && tab.classList.contains("active");
    }})()
    """


def read_probe(cdp: Cdp) -> dict[str, Any]:
    return cdp.eval(
        """
        (() => {
          const p = window.__replayPerfProbe;
          if (!p) return null;
          const resources = performance.getEntriesByType("resource")
            .filter((entry) => {
              try {
                const url = new URL(entry.name);
                return [
                  "/api/load_replay_dataset",
                  "/api/replay_series",
                  "/api/replay_video",
                  "/api/replay_transforms",
                  "/api/review_episode",
                  "/api/review_transforms",
                  "/api/episode_cams",
                  "/api/episode_series",
                  "/api/episode_video",
                  "/api/episode_transforms",
                ].includes(url.pathname);
              } catch (e) {
                return false;
              }
            })
            .map((entry) => {
              const url = new URL(entry.name);
              return {
                path: url.pathname,
                query: url.search,
                initiatorType: entry.initiatorType,
                startTime: entry.startTime - p.start,
                duration: entry.duration,
                transferSize: entry.transferSize || 0,
                encodedBodySize: entry.encodedBodySize || 0,
                decodedBodySize: entry.decodedBodySize || 0,
              };
            });
          return {
            videos: Object.values(p.videos),
            events: p.events,
            playCommand: p.playCommand || null,
            resources,
            videoStates: Array.from(document.querySelectorAll(
              "#cam-strip video.cam, #tp-cam-strip video.tp-cam",
            )).map((v) => ({
              key: v.dataset.key || "",
              paused: v.paused,
              readyState: v.readyState,
              currentTime: v.currentTime,
            })),
            rafGaps: p.rafGaps,
            syncSamples: p.syncSamples,
            urdfFrames: p.urdfFrames,
            longTasks: p.longTasks,
            now: performance.now() - p.start,
          };
        })()
        """
    )


def summarize_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for resource in resources:
        query = resource.get("query") or ""
        path = resource.get("path") or ""
        label = path
        if path == "/api/replay_video":
            params = dict(urllib.parse.parse_qsl(query.lstrip("?")))
            label = f"{path}?episode={params.get('episode', '')}&cam={params.get('cam', '')}"
        rows.append(
            {
                "label": label,
                "initiator": resource.get("initiatorType"),
                "start_ms": round(float(resource.get("startTime") or 0), 1),
                "duration_ms": round(float(resource.get("duration") or 0), 1),
                "transfer": int(resource.get("transferSize") or 0),
                "encoded": int(resource.get("encodedBodySize") or 0),
            }
        )
    return sorted(rows, key=lambda row: row["duration_ms"], reverse=True)[:12]


def summarize(name: str, data: dict[str, Any]) -> dict[str, Any]:
    videos = [video for video in data["videos"] if video.get("tag") == "VIDEO"]
    first_frames = [v["firstFrame"] for v in videos if v["firstFrame"] is not None]
    visible = [v["visible"] for v in videos if v["visible"] is not None]
    canplays = [v["canplay"] for v in videos if v["canplay"] is not None]
    plays = [v["play"] for v in videos if v["play"] is not None]
    playings = [v["playing"] for v in videos if v["playing"] is not None]
    waiting = sum(len(v["waiting"]) for v in videos)
    stalled = sum(len(v["stalled"]) for v in videos)
    video_states = data.get("videoStates") or []
    current_times = [v["currentTime"] for v in video_states if v["currentTime"] is not None]
    seeking_after_play = 0
    for v in videos:
        playback_start = v["playing"] if v["playing"] is not None else v["play"]
        if playback_start is None:
            playback_start = data.get("playCommand")
        if playback_start is None:
            continue
        seeking_after_play += sum(1 for row in v["seeking"] if row[0] > playback_start + 250)
    play_markers = [
        marker
        for video in videos
        for marker in (video.get("playing"), video.get("play"))
        if marker is not None
    ]
    stable_start = (min(play_markers) if play_markers else data.get("playCommand"))
    stable_start = None if stable_start is None else float(stable_start) + 250.0
    stable_waiting = sum(
        sum(1 for row in video["waiting"] if stable_start is not None and row[0] >= stable_start)
        for video in videos
    )
    stable_stalled = sum(
        sum(1 for row in video["stalled"] if stable_start is not None and row[0] >= stable_start)
        for video in videos
    )
    def percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))
        return ordered[index]

    raf_gaps = [float(value) for value in data.get("rafGaps") or [] if float(value) >= 0]
    sync_samples = [
        sample
        for sample in data.get("syncSamples") or []
        if len(sample.get("times") or []) >= len(videos)
    ]
    camera_skews = [max(sample["times"]) - min(sample["times"]) for sample in sync_samples]
    urdf_skews = [
        abs(float(sample["cursor"]) - float(sample["urdf"]))
        for sample in sync_samples
        if sample.get("cursor") is not None and sample.get("urdf") is not None
    ]
    media_intervals = {
        v["key"]: [
            float(curr[0]) - float(prev[0])
            for prev, curr in zip(
                v.get("videoFrames") or [], (v.get("videoFrames") or [])[1:], strict=False
            )
            if float(curr[0]) > float(prev[0])
        ]
        for v in videos
    }
    media_p95 = {
        key: percentile(values, 0.95)
        for key, values in media_intervals.items()
    }
    media_max = {
        key: max(values) if values else None
        for key, values in media_intervals.items()
    }
    long_tasks = [
        float(row.get("duration", 0))
        for row in data.get("longTasks") or []
        if float(row.get("t", -1)) >= 0
    ]
    return {
        "name": name,
        "video_count": len(videos),
        "all_first_frame_ms": max(first_frames) if first_frames else None,
        "all_visible_ms": max(visible) if visible else None,
        "all_canplay_ms": max(canplays) if canplays else None,
        "first_play_ms": min(plays) if plays else None,
        "first_playing_ms": min(playings) if playings else None,
        "play_command_ms": data.get("playCommand"),
        "all_unpaused": bool(video_states) and all(not v["paused"] for v in video_states),
        "min_current_time_sec": min(current_times) if current_times else None,
        "waiting_events": waiting,
        "stalled_events": stalled,
        "stable_waiting_events": stable_waiting,
        "stable_stalled_events": stable_stalled,
        "seeking_after_play_events": seeking_after_play,
        "raf_gap_p95_ms": percentile(raf_gaps, 0.95),
        "raf_gap_max_ms": max(raf_gaps) if raf_gaps else None,
        "camera_skew_p95_ms": percentile(camera_skews, 0.95),
        "camera_skew_max_ms": max(camera_skews) if camera_skews else None,
        "urdf_frame_skew_p95": percentile(urdf_skews, 0.95),
        "urdf_frame_skew_max": max(urdf_skews) if urdf_skews else None,
        "urdf_update_count": len(data.get("urdfFrames") or []),
        "video_interval_p95_ms": media_p95,
        "video_interval_max_ms": media_max,
        "long_task_max_ms": max(long_tasks) if long_tasks else None,
        "resources": summarize_resources(data.get("resources") or []),
        "screenshot_path": data.get("screenshotPath"),
        "error": data.get("error"),
        "raw": data,
    }


def capture_measurement(
    cdp: Cdp, name: str, timeout: float, play_seconds: float
) -> dict[str, Any]:
    try:
        wait_for(cdp, VIDEOS_VISIBLE_READY_EXPR, timeout)
        if play_seconds > 0:
            time.sleep(play_seconds)
        data = read_probe(cdp)
    except Exception as exc:
        data = read_probe(cdp) or {"videos": [], "events": [], "videoStates": []}
        data["error"] = str(exc)
    data["screenshotPath"] = capture_screenshot(cdp, name)
    return summarize(name, data)


def ensure_collect_replay_stopped(cdp: Cdp) -> None:
    active = cdp.eval(COLLECT_REPLAY_STOP_READY_EXPR)
    if not active:
        return
    human_click(cdp, "#review-return-live")
    wait_for(cdp, COLLECT_REPLAY_BUTTON_READY_EXPR, 10)


def measurement_failures(
    summaries: list[dict[str, Any]],
    max_visible_ms: float,
    max_raf_p95_ms: float,
    max_raf_gap_ms: float,
    max_camera_p95_ms: float,
    max_camera_skew_ms: float,
    max_urdf_frame_skew: float,
    max_video_p95_ms: float,
    max_video_gap_ms: float,
    max_long_task_ms: float,
) -> list[dict[str, Any]]:
    failures = []
    for row in summaries:
        visible_ms = row.get("all_visible_ms")
        media_p95 = [v for v in (row.get("video_interval_p95_ms") or {}).values() if v is not None]
        media_max = [v for v in (row.get("video_interval_max_ms") or {}).values() if v is not None]
        failed = bool(row.get("error"))
        failed |= max_visible_ms > 0 and (visible_ms is None or float(visible_ms) > max_visible_ms)
        failed |= max_raf_p95_ms > 0 and (
            row.get("raf_gap_p95_ms") is None or float(row["raf_gap_p95_ms"]) > max_raf_p95_ms
        )
        failed |= max_raf_gap_ms > 0 and (
            row.get("raf_gap_max_ms") is None or float(row["raf_gap_max_ms"]) > max_raf_gap_ms
        )
        failed |= max_camera_p95_ms > 0 and (
            row.get("camera_skew_p95_ms") is None
            or float(row["camera_skew_p95_ms"]) > max_camera_p95_ms
        )
        failed |= max_camera_skew_ms > 0 and (
            row.get("camera_skew_max_ms") is None
            or float(row["camera_skew_max_ms"]) > max_camera_skew_ms
        )
        failed |= max_urdf_frame_skew > 0 and (
            row.get("urdf_frame_skew_max") is None
            or float(row["urdf_frame_skew_max"]) > max_urdf_frame_skew
        )
        failed |= max_video_p95_ms > 0 and (
            len(media_p95) < row.get("video_count", 0)
            or any(float(value) > max_video_p95_ms for value in media_p95)
        )
        failed |= max_video_gap_ms > 0 and (
            len(media_max) < row.get("video_count", 0)
            or any(float(value) > max_video_gap_ms for value in media_max)
        )
        failed |= max_long_task_ms > 0 and (
            row.get("long_task_max_ms") is not None
            and float(row["long_task_max_ms"]) > max_long_task_ms
        )
        failed |= row.get("stable_stalled_events", 0) > 0
        if failed:
            failures.append(row)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection-dataset", required=True)
    parser.add_argument("--collection-episode", type=int, default=0)
    parser.add_argument("--rollout-episode", type=int, default=0)
    parser.add_argument("--play-seconds", type=float, default=5.0)
    parser.add_argument("--replay-next-count", type=int, default=0)
    parser.add_argument("--collect-task", default="")
    parser.add_argument("--collect-count", type=int, default=1)
    parser.add_argument("--debug-count", type=int, default=1)
    parser.add_argument("--rl-count", type=int, default=1)
    parser.add_argument("--result-count", type=int, default=1)
    parser.add_argument("--rapid-switches", action="store_true")
    parser.add_argument("--visible-timeout", type=float, default=20.0)
    parser.add_argument("--max-visible-ms", type=float, default=1000.0)
    parser.add_argument("--max-raf-p95-ms", type=float, default=25.0)
    parser.add_argument("--max-raf-gap-ms", type=float, default=100.0)
    parser.add_argument("--max-camera-p95-ms", type=float, default=33.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=80.0)
    parser.add_argument("--max-urdf-frame-skew", type=float, default=1.0)
    parser.add_argument("--max-video-p95-ms", type=float, default=100.0)
    parser.add_argument("--max-video-gap-ms", type=float, default=150.0)
    parser.add_argument("--max-long-task-ms", type=float, default=100.0)
    args = parser.parse_args()

    chrome, cdp = start_chrome()
    summaries = []
    probe_error = None
    try:
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Page.navigate", {"url": args.base_url})
        wait_for(cdp, PAGE_READY_EXPR, 20)

        human_click(cdp, '.tab[data-tab="replay"]')
        wait_for(cdp, tab_active_expr("replay"), 5)
        wait_for(cdp, REPLAY_LOAD_READY_EXPR, 10)
        set_value(cdp, "#replay-dataset-input", args.collection_dataset)
        set_value(cdp, "#replay-episode-input", str(args.collection_episode))
        human_click_and_start_probe(cdp, "#replay-b-episode-confirm")
        wait_for(cdp, SCRUB_PLAY_VISIBLE_EXPR, args.visible_timeout)
        human_click(cdp, "#scrub-play")
        summaries.append(
            capture_measurement(cdp, "replay-load", args.visible_timeout, args.play_seconds)
        )
        wait_for(cdp, SCRUB_PLAY_VISIBLE_EXPR, 10)
        for i in range(args.replay_next_count):
            wait_for(cdp, REPLAY_NEXT_READY_EXPR, 10)
            human_click_and_start_probe(cdp, "#replay-b-episode-next")
            wait_for(cdp, SCRUB_PLAY_VISIBLE_EXPR, args.visible_timeout)
            human_click(cdp, "#scrub-play")
            summaries.append(
                capture_measurement(
                    cdp, f"replay-next-{i + 1}", args.visible_timeout, args.play_seconds
                )
            )

        if args.collect_count > 0:
            human_click(cdp, '.tab[data-tab="collect"]')
            wait_for(cdp, tab_active_expr("collect"), 5)
            if args.collect_task:
                set_value(cdp, "#collect-prompt-list", args.collect_task)
            wait_for(cdp, COLLECT_REPLAYABLE_READY_EXPR, 10)
            for i in range(args.collect_count):
                ensure_collect_replay_stopped(cdp)
                target = select_clickable_tile(cdp, "#collect-queue-tiles", i)
                human_click_target_and_start_probe(cdp, target, f"collect-tile-{i}")
                summaries.append(
                    capture_measurement(
                        cdp, f"collect-{i}", args.visible_timeout, args.play_seconds
                    )
                )

        if args.debug_count > 0:
            human_click(cdp, '.tab[data-tab="debug"]')
            wait_for(cdp, tab_active_expr("debug"), 5)
            wait_for(cdp, ROLLOUT_REPLAYABLE_READY_EXPR, 10)
            for i in range(args.debug_count):
                target = select_clickable_tile(cdp, "#rollout-save-queue-tiles", i)
                human_click_target_and_start_probe(cdp, target, f"debug-tile-{i}")
                summaries.append(
                    capture_measurement(
                        cdp, f"debug-{i}", args.visible_timeout, args.play_seconds
                    )
                )

        if args.rl_count > 0:
            human_click(cdp, '.tab[data-tab="rl"]')
            wait_for(cdp, tab_active_expr("rl"), 5)
            wait_for(cdp, RL_REPLAYABLE_READY_EXPR, 10)
            for i in range(args.rl_count):
                target = select_clickable_tile(cdp, "#rl-save-tiles", i)
                human_click_target_and_start_probe(cdp, target, f"rl-tile-{i}")
                summaries.append(
                    capture_measurement(cdp, f"rl-{i}", args.visible_timeout, args.play_seconds)
                )

        if args.result_count > 0:
            human_click(cdp, '.tab[data-tab="result"]')
            wait_for(cdp, tab_active_expr("result"), 5)
            wait_for(cdp, RESULT_TRIAL_READY_EXPR, 10)
            for i in range(args.result_count):
                human_click(cdp, ".rv-model .rv-node-head")
                wait_for(cdp, ".rv-task .rv-node-head", 5)
                human_click(cdp, ".rv-task .rv-node-head")
                wait_for(cdp, RESULT_TRIAL_READY_EXPR, 5)
                target = select_clickable_tile(
                    cdp, ".rv-trials", i, ".eval-trial.scored"
                )
                human_click_target_and_start_probe(cdp, target, f"result-trial-{i}")
                wait_for(cdp, RESULT_POP_READY_EXPR, 10)
                human_click(cdp, "#tp-play")
                summaries.append(
                    capture_measurement(
                        cdp, f"result-{i}", args.visible_timeout, args.play_seconds
                    )
                )

        if args.rapid_switches:
            for tab, container in (
                ("collect", "#collect-queue-tiles"),
                ("debug", "#rollout-save-queue-tiles"),
                ("rl", "#rl-save-tiles"),
            ):
                human_click(cdp, f'.tab[data-tab="{tab}"]')
                wait_for(cdp, tab_active_expr(tab), 5)
                row = rapid_tile_switch(
                    cdp, container, tab, args.visible_timeout, args.play_seconds
                )
                if row is not None:
                    summaries.append(row)
    except Exception as exc:
        probe_error = str(exc)
        summaries.append(
            {
                "name": "probe-error",
                "video_count": 0,
                "all_first_frame_ms": None,
                "all_visible_ms": None,
                "all_canplay_ms": None,
                "first_play_ms": None,
                "first_playing_ms": None,
                "play_command_ms": None,
                "all_unpaused": False,
                "min_current_time_sec": None,
                "waiting_events": 0,
                "stalled_events": 0,
                "seeking_after_play_events": 0,
                "resources": [],
                "screenshot_path": None,
                "error": probe_error,
                "raw": {},
            }
        )
    finally:
        cdp.close()
        chrome.terminate()
        chrome.wait(timeout=10)

    print(json.dumps(summaries, indent=2))
    if probe_error:
        raise SystemExit(f"probe failed: {probe_error}")
    failures = measurement_failures(
        summaries,
        args.max_visible_ms,
        args.max_raf_p95_ms,
        args.max_raf_gap_ms,
        args.max_camera_p95_ms,
        args.max_camera_skew_ms,
        args.max_urdf_frame_skew,
        args.max_video_p95_ms,
        args.max_video_gap_ms,
        args.max_long_task_ms,
    )
    if failures:
        names = ", ".join(str(row["name"]) for row in failures)
        raise SystemExit(f"visible video threshold failed for: {names}")


if __name__ == "__main__":
    main()
