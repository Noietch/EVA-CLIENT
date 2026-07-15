#!/usr/bin/env python3
"""Generic fake hardware node for debugging the EVA client without real hardware.

Speaks the same ZMQ protocol as every real execution-layer node — PUBlishes
WireObservation frames and SUBscribes to WireAction commands — but synthesizes
everything in software for whichever robot is named:

  * state starts at the robot's initial_qpos and is fed back from the last action
    received (closed-loop follow), so RUN/DEBUG inference visibly moves the arms;
  * images are solid-color frames stamped with a moving band so the console shows
    a live, changing picture for each declared camera;
  * COLLECT teleop is synthesized as a slow sine wobble around the current qpos,
    so action_qpos is never None and the full collection/save pipeline can be
    exercised end to end;
  * when idle (no live policy/teleop), each ARM group's shoulder-pitch joint
    free-runs a small in-phase wobble so the arms always visibly move without
    swinging toward the centerline or self-colliding.

The robot is built from its bundled zoo config via ROBOT_REGISTRY, so a single
node generalizes over single-arm, dual-arm, and multi-group robots: state/eef are
keyed per actuator group exactly like the real node. No real-hardware SDK is
imported, so it runs anywhere.

Usage (per-robot wrappers call ``main`` with the registry name):
  python examples/hardware/<robot>/fake_node.py
  # then, in another shell:
  eva --config configs/01_deploy/<robot>/openpi_qpos.py
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import robots  # noqa: F401  # registers every zoo robot under ROBOT_REGISTRY on import
from core.registry import ROBOT_REGISTRY
from transport.zmq import (
    COLLECTION_START_TARGET,
    COLLECTION_STOP_TARGET,
    HIL_START_TARGET,
    HIL_STOP_TARGET,
    WireObservation,
    pack_observation,
    unpack_action,
)

logger = logging.getLogger(__name__)

# Per-arm EEF pose width: xyz + quat wxyz + gripper.
_EEF_DOF = 8

# No real action within this window => free-run idle wobble so the arms always move.
_IDLE_ACTION_TIMEOUT_S = 0.5

# Idle wobble drives only the shoulder-pitch joint (arm-local index 1) of each arm,
# in phase and small-amplitude, so dual arms mirror each other without swinging toward
# the centerline or folding a single arm onto itself — no self-collision.
_IDLE_WOBBLE_JOINT = 1
_IDLE_WOBBLE_AMP = 0.15

# Distinct base colors per camera so the console tiles are easy to tell apart.
_CAMERA_COLORS = {
    "cam_high": (160, 70, 40),
    "cam_wrist": (40, 90, 160),
    "cam_left_wrist": (40, 90, 160),
    "cam_right_wrist": (40, 160, 90),
}
_DEFAULT_COLOR = (80, 80, 80)
_HIL_SUPPORTED_ROBOTS = frozenset({"r1_lite", "ur5e", "arx_r5", "agilex_piper"})


def _synth_image(
    color: tuple[int, int, int], height: int, width: int, frame_index: int
) -> np.ndarray:
    """Build one solid-color RGB frame whose brightness pulses with the frame index.

    Args:
        color: base (r, g, b) for this camera.
        height: image height in pixels.
        width: image width in pixels.
        frame_index: running frame counter, drives a slow brightness pulse.

    Returns:
        img: [height, width, 3] uint8 RGB image.
    """
    pulse = 0.5 + 0.5 * float(np.sin(frame_index * 0.05))
    rgb = np.asarray(color, dtype=np.float32) * (0.4 + 0.6 * pulse)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = rgb.astype(np.uint8)
    # A moving bright band gives an unmistakable "this is live" cue in the UI.
    band = int((frame_index * 4) % height)
    img[band : min(band + max(height // 20, 2), height), :, :] = 255
    return img


class FakeRobotNode:
    """Software-only stand-in for any zoo robot: synthesizes obs, echoes actions."""

    def __init__(
        self,
        robot_name: str,
        observation_endpoint: str,
        action_endpoint: str,
        publish_rate_hz: float,
        image_height: int,
        image_width: int,
    ) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._obs_pub = self._ctx.socket(zmq.PUB)
        self._obs_pub.bind(observation_endpoint)
        self._action_sub = self._ctx.socket(zmq.SUB)
        self._action_sub.bind(action_endpoint)
        self._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._action_sub.setsockopt(zmq.RCVTIMEO, 0)

        self._robot = ROBOT_REGISTRY.build(robot_name)
        self._action_dim = self._robot.total_action_dim
        self._groups = self._robot.actuator_groups
        self._cameras = self._robot.observation_schema.cameras
        self._publish_rate_hz = publish_rate_hz
        self._image_height = image_height
        self._image_width = image_width

        self._qpos = np.asarray(self._robot.initial_qpos, dtype=np.float32).copy()
        self._hil_supported = robot_name in _HIL_SUPPORTED_ROBOTS
        self._initial_qpos = self._qpos.copy()
        self._collection_active = False
        self._hil_active = False
        self._hil_mode = "relative"
        self._hil_error = ""
        self._hil_input = self._qpos.copy()
        self._hil_input_anchor = self._qpos.copy()
        self._hil_robot_anchor = self._qpos.copy()
        self._frame_index = 0
        self._started_at = time.monotonic()
        self._last_action_time = 0.0
        self._stop = threading.Event()

    def _split_by_group(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Slice a flat full-robot vector into per-actuator-group chunks.

        Args:
            vector: [action_dim] flat vector ordered by actuator group.

        Returns:
            group name -> [group.dof] slice (copies).
        """
        parts: dict[str, np.ndarray] = {}
        offset = 0
        for group in self._groups:
            parts[group.name] = vector[offset : offset + group.dof].copy()
            offset += group.dof
        return parts

    def _zero_eef_by_group(self) -> dict[str, np.ndarray]:
        return {
            group.name: np.zeros(_EEF_DOF, dtype=np.float32)
            for group in self._robot.arm_groups
        }

    def _read_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for cam in self._cameras:
            color = _CAMERA_COLORS.get(cam.observation_key, _DEFAULT_COLOR)
            images[cam.observation_key] = _synth_image(
                color, self._image_height, self._image_width, self._frame_index
            )
        return images

    def start_collection(self) -> None:
        if self._collection_active:
            return
        self._collection_active = True
        logger.info("[FAKE] collection started (synthetic teleop wobble)")

    def stop_collection(self) -> None:
        if not self._collection_active:
            return
        self._collection_active = False
        logger.info("[FAKE] collection stopped")

    def start_hil(self, mode: str) -> None:
        if not self._hil_supported:
            self._hil_active = False
            self._hil_error = f"{self._robot.name} has no HIL leader adapter"
            return
        if mode not in {"absolute", "relative"}:
            self._hil_error = f"Unsupported HIL control mode: {mode}"
            return
        self._hil_mode = mode
        self._hil_input = self._qpos.copy()
        self._hil_input_anchor = self._qpos.copy()
        self._hil_robot_anchor = self._qpos.copy()
        self._hil_error = ""
        self._hil_active = True
        logger.info("[FAKE] HIL started mode=%s", mode)

    def stop_hil(self) -> None:
        self._hil_active = False
        self._hil_error = ""
        logger.info("[FAKE] HIL stopped")

    def hil_snapshot(self) -> dict[str, object]:
        """Return the fake leader, feedback, and HIL activation state."""
        return {
            "robot": self._robot.name,
            "supported": self._hil_supported,
            "active": self._hil_active,
            "mode": self._hil_mode,
            "error": self._hil_error,
            "feedback": {
                name: value.astype(float).tolist()
                for name, value in self._split_by_group(self._qpos).items()
            },
            "hil": {
                name: value.astype(float).tolist()
                for name, value in self._split_by_group(self._hil_input).items()
            },
            "groups": [
                {
                    "name": group.name,
                    "joint_names": list(group.joint_names),
                    "dof": group.dof,
                    "gripper_index": group.gripper_index,
                }
                for group in self._groups
            ],
        }

    def sync_hil_to_feedback(self) -> None:
        self._hil_input = self._qpos.copy()
        if self._hil_active:
            self._hil_input_anchor = self._hil_input.copy()
            self._hil_robot_anchor = self._qpos.copy()

    def adjust_hil_joint(self, group_name: str, joint_index: int, delta: float) -> np.ndarray:
        offset = 0
        for group in self._groups:
            if group.name == group_name:
                if joint_index < 0 or joint_index >= group.dof:
                    raise ValueError(
                        f"joint_index must be in [0, {group.dof - 1}], got {joint_index}"
                    )
                self._hil_input[offset + joint_index] += float(delta)
                return self._hil_input[offset : offset + group.dof].copy()
            offset += group.dof
        raise ValueError(f"Unknown actuator group: {group_name}")

    def set_hil_joint(self, group_name: str, joint_index: int, value: float) -> np.ndarray:
        offset = 0
        for group in self._groups:
            if group.name == group_name:
                if joint_index < 0 or joint_index >= group.dof:
                    raise ValueError(
                        f"joint_index must be in [0, {group.dof - 1}], got {joint_index}"
                    )
                self._hil_input[offset + joint_index] = float(value)
                return self._hil_input[offset : offset + group.dof].copy()
            offset += group.dof
        raise ValueError(f"Unknown actuator group: {group_name}")

    def _drain_actions(self) -> None:
        while True:
            try:
                payload = self._action_sub.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            action = unpack_action(payload)
            if action.target == COLLECTION_START_TARGET:
                self.start_collection()
                continue
            if action.target == COLLECTION_STOP_TARGET:
                self.stop_collection()
                continue
            if action.target == HIL_START_TARGET:
                self.start_hil(action.mode or "relative")
                continue
            if action.target == HIL_STOP_TARGET:
                self.stop_hil()
                continue
            if action.target == "sim":
                continue
            if self._hil_active:
                continue
            # Closed-loop: the commanded joint vector becomes the next reported state.
            vector = np.asarray(action.action, dtype=np.float32).reshape(-1)
            if vector.shape == (self._action_dim,):
                self._qpos = vector.copy()
                self._last_action_time = time.monotonic()

    def _synth_idle_qpos(self) -> np.ndarray:
        """Free-run wobble so the arms keep moving when idle, without self-collision.

        Only each ARM group's shoulder-pitch joint moves, small-amplitude and in
        phase, so dual arms mirror each other and never swing toward the centerline
        or fold; all other joints and the grippers stay at initial_qpos.

        Returns:
            qpos: [action_dim] float32 state.
        """
        t = time.monotonic() - self._started_at
        qpos = self._initial_qpos.copy()
        offset = 0
        for group in self._groups:
            if group.name.endswith("arm") and group.dof > _IDLE_WOBBLE_JOINT:
                qpos[offset + _IDLE_WOBBLE_JOINT] += _IDLE_WOBBLE_AMP * float(np.sin(t * 1.0))
            offset += group.dof
        return qpos.astype(np.float32)

    def _publish_observation(self) -> None:
        action_qpos = None
        action_eef = None
        eef = self._zero_eef_by_group()
        # Idle and COLLECT share one wobble so the visible motion never changes when
        # recording starts; COLLECT only additionally reports it as action_qpos.
        if self._hil_active:
            if self._hil_mode == "relative":
                self._qpos = self._hil_robot_anchor + (
                    self._hil_input - self._hil_input_anchor
                )
            else:
                self._qpos = self._hil_input.copy()
            action_qpos = self._qpos.copy()
            action_eef = np.zeros(_EEF_DOF * len(self._robot.arm_groups), dtype=np.float32)
        elif self._collection_active or (
            time.monotonic() - self._last_action_time > _IDLE_ACTION_TIMEOUT_S
        ):
            self._qpos = self._synth_idle_qpos()
        if self._collection_active and not self._hil_active:
            action_qpos = self._qpos.copy()
            action_eef = np.zeros(_EEF_DOF * len(self._robot.arm_groups), dtype=np.float32)
        obs = WireObservation(
            t=time.monotonic(),
            images=self._read_images(),
            state=self._split_by_group(self._qpos),
            eef=eef,
            action=action_qpos,
            action_eef=action_eef,
            hil_supported=self._hil_supported,
            hil_active=self._hil_active,
            hil_error=self._hil_error,
        )
        self._obs_pub.send(pack_observation(obs))
        self._frame_index += 1

    def serve_forever(self) -> None:
        period = 1.0 / max(self._publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        logger.info(
            "[FAKE] serving: action_dim=%d groups=%s cameras=%s rate=%.1fHz",
            self._action_dim,
            [group.name for group in self._groups],
            [cam.observation_key for cam in self._cameras],
            self._publish_rate_hz,
        )
        while not self._stop.is_set():
            self._drain_actions()
            self._publish_observation()
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def build_arg_parser(robot_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Fake {robot_name} hardware node for EVA debugging."
    )
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)
    parser.add_argument("--eva-url", default="http://127.0.0.1:8080")
    parser.add_argument("--no-open-ui", action="store_true")
    return parser


def _proxy_operator(eva_url: str, intent: str) -> None:
    payload = json.dumps({"intent": intent}).encode("utf-8")
    request = urllib.request.Request(
        f"{eva_url.rstrip('/')}/api/operator_action",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2):
        return


def make_ui_handler(node: FakeRobotNode, eva_url: str) -> type[BaseHTTPRequestHandler]:
    """Build the generic fake HIL control-page request handler."""

    class FakeHilUiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(UI_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/state":
                self._send_json(node.hil_snapshot())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/api/joint":
                    values = node.set_hil_joint(
                        str(body["group"]), int(body["index"]), float(body["value"])
                    )
                    self._send_json({"positions": values.astype(float).tolist()})
                    return
                if self.path == "/api/joint_delta":
                    values = node.adjust_hil_joint(
                        str(body["group"]), int(body["index"]), float(body["delta"])
                    )
                    self._send_json({"positions": values.astype(float).tolist()})
                    return
                if self.path == "/api/sync":
                    node.sync_hil_to_feedback()
                    self._send_json({"ok": True})
                    return
                if self.path == "/api/operator":
                    button = str(body["button"]).lower()
                    intent = {"x": "start", "y": "accept", "cancel": "cancel"}[button]
                    _proxy_operator(eva_url, intent)
                    self._send_json({"ok": True})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except OSError as exc:
                self._send_json(
                    {"error": f"EVA console unavailable: {exc}"},
                    HTTPStatus.BAD_GATEWAY,
                )

        def _read_json(self) -> dict:
            size = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(size).decode("utf-8"))

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

        def _send(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return FakeHilUiHandler


UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>EVA Fake HIL</title>
<style>
body { font: 14px system-ui; margin: 20px; background: #f4f5f7; color: #18202a; }
button { margin: 3px; padding: 7px; }
.bar, .groups { display: flex; gap: 12px; flex-wrap: wrap; }
.panel {
  background: white; border: 1px solid #ccd3dc; border-radius: 8px;
  padding: 12px; min-width: 320px;
}
.joint {
  display: grid; grid-template-columns: 150px 36px 1fr 36px 58px;
  gap: 6px; align-items: center; margin: 5px 0;
}
.status { margin: 10px 0; color: #526170; }
input { width: 100%; }
</style>
</head>
<body>
<h1 id="title">EVA Fake HIL</h1>
<div class="bar">
  <button data-op="x">X / Stop &amp; Take Over</button>
  <button data-op="y">Y / Resume</button>
  <button data-op="cancel">Abandon</button>
  <button id="sync">Sync HIL</button>
</div>
<div id="status" class="status"></div>
<div id="groups" class="groups"></div>
<script>
const post = (path, body) => fetch(path, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify(body),
}).then(response => response.json());
let built = false;
function build(state) {
  if (built) return;
  built = true;
  document.querySelector("#title").textContent = `${state.robot} Fake HIL`;
  const root = document.querySelector("#groups");
  state.groups.forEach(group => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<b>${group.name}</b>`;
    group.joint_names.forEach((name, index) => {
      const row = document.createElement("div");
      const suffix = group.gripper_index === index ? " (gripper)" : "";
      row.className = "joint";
      row.innerHTML = `<span>${name}${suffix}</span><button>-</button>`
        + `<input type="range" min="-3.14" max="3.14" step="0.01">`
        + `<button>+</button><output></output>`;
      const input = row.querySelector("input");
      const output = row.querySelector("output");
      input.oninput = () => {
        output.value = Number(input.value).toFixed(2);
        post("/api/joint", {group: group.name, index, value: Number(input.value)});
      };
      const buttons = row.querySelectorAll("button");
      buttons[0].onclick = () => post(
        "/api/joint_delta", {group: group.name, index, delta: -0.05});
      buttons[1].onclick = () => post(
        "/api/joint_delta", {group: group.name, index, delta: 0.05});
      panel.appendChild(row);
    });
    root.appendChild(panel);
  });
}
function refresh() {
  fetch("/api/state").then(response => response.json()).then(state => {
    build(state);
    document.querySelector("#status").textContent =
      `supported=${state.supported} active=${state.active} `
      + `mode=${state.mode} ${state.error}`;
    state.groups.forEach(group => {
      const panels = [...document.querySelectorAll(".panel")];
      const panel = panels.find(item => item.querySelector("b").textContent === group.name);
      panel.querySelectorAll(".joint").forEach((row, index) => {
        row.querySelector("input").value = state.hil[group.name][index];
        row.querySelector("output").value =
          Number(state.hil[group.name][index]).toFixed(2);
      });
    });
  });
}
document.querySelector("#sync").onclick = () => post("/api/sync", {});
document.querySelectorAll("[data-op]").forEach(button => {
  button.onclick = () => post("/api/operator", {button: button.dataset.op});
});
setInterval(refresh, 250);
refresh();
</script>
</body>
</html>"""


def main(robot_name: str) -> None:
    """Run a fake node for the named zoo robot until SIGINT/SIGTERM.

    Args:
        robot_name: ROBOT_REGISTRY key (zoo robot ``name``, e.g. "ur5e").
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser(robot_name).parse_args()
    node = FakeRobotNode(
        robot_name=robot_name,
        observation_endpoint=args.obs_endpoint,
        action_endpoint=args.action_endpoint,
        publish_rate_hz=args.rate,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    ui_server = ThreadingHTTPServer(
        (args.ui_host, args.ui_port), make_ui_handler(node, args.eva_url)
    )
    threading.Thread(target=ui_server.serve_forever, daemon=True).start()
    ui_url = f"http://{args.ui_host}:{args.ui_port}"
    logger.info("[FAKE] HIL control UI: %s", ui_url)
    if not args.no_open_ui:
        webbrowser.open(ui_url)
    signal.signal(signal.SIGINT, lambda *_args: node.close())
    signal.signal(signal.SIGTERM, lambda *_args: node.close())
    try:
        node.serve_forever()
    finally:
        ui_server.shutdown()
        ui_server.server_close()
        node.close()
