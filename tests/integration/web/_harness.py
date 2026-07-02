"""Reusable harness for web UI HTTP integration tests.

Importable from both ``conftest.py`` and test modules (pytest's prepend import mode
puts this directory on ``sys.path``). Keeps the fixture in conftest thin and lets
non-fixture tests reuse the same server lifecycle via ``serve_console``.

Everything runs offline via the ``debug`` transport + ``mock`` policy, so no
hardware, ROS, or policy server is required (see work_dirs/COMMONS notes).
"""

from __future__ import annotations

import contextlib
import dataclasses
import http.client
import json
import queue
import socket
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from core.app import run as app
from core.app.console.server import (
    ConsoleContext,
    ConsoleRequestHandler,
    ThreadingHTTPServer,
)
from core.app.state import RuntimeState, SessionState
from core.config import ConfigDict, load_config
from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY
from core.types import Observation
from robots.base import Robot
from transport.base import TransportBridge, build_transport


@TRANSPORT_REGISTRY.register("debug")
class _DebugTransport(TransportBridge):
    """Offline test transport: synthetic random images + last-published qpos.

    Test-only stand-in (registered from the harness, not shipped in src) so the
    web console integration suite runs with no ROS/hardware. Produces one random
    [H, W, 3] image per camera and echoes published actions back as the qpos.
    """

    def __init__(self, config: ConfigDict, robot: Robot) -> None:
        self._config = config
        self._robot = robot
        self._shutdown = threading.Event()
        self._qpos = np.asarray(robot.initial_qpos, dtype=np.float32).copy()
        self._camera_names = [cam.observation_key for cam in robot.observation_schema.cameras]

    def get_frame(self) -> Observation | None:
        if self._shutdown.is_set():
            return None
        h = self._config.transport.image_height
        w = self._config.transport.image_width
        images = {
            name: np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            for name in self._camera_names
        }
        return Observation(images=images, state_qpos=self._qpos.copy())

    def publish_action(self, action: np.ndarray, target: str = "real") -> None:
        self._qpos = np.asarray(action, dtype=np.float32).copy()

    def get_latest_qpos(self) -> np.ndarray | None:
        return self._qpos.copy()

    def is_offline(self) -> bool:
        return True

    def close(self) -> None:
        self._shutdown.set()

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()


SYNC_STRATEGY = {"sync": {"type": "BaseInferStrategy", "args": {"execute_horizon": 5}}}
MULTI_STRATEGY = {
    "sync": {"type": "BaseInferStrategy", "args": {"execute_horizon": 5}},
    "rtc": {"type": "RtcInferStrategy", "args": {}},
}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@dataclasses.dataclass
class Response:
    """One HTTP round-trip: status + decoded JSON body (+ raw bytes / headers)."""

    status: int
    json: Any
    raw: bytes
    headers: dict[str, str]


class WebHarness:
    """Drives one web server over real HTTP with a synchronous command pump.

    Holds the same (config, runtime, session) triple the server reads, so a test
    can POST a control verb, ``pump()`` to drain the queue, then GET ``/api/status``
    and assert the state transition — the full browser→backend→state round-trip.
    """

    def __init__(
        self,
        config: ConfigDict,
        runtime: RuntimeState,
        session: SessionState,
        port: int,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.session = session
        self.port = port

    # --- HTTP ---

    def _request(
        self, method: str, path: str, body: dict | None = None, headers: dict | None = None
    ) -> Response:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = json.dumps(body) if body is not None else (None if method == "GET" else "{}")
        hdrs = {"Content-Type": "application/json"} if method != "GET" else {}
        if headers:
            hdrs.update(headers)
        conn.request(method, path, payload, hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        ctype = resp_headers.get("content-type", "")
        decoded: Any = None
        if "application/json" in ctype and raw:
            decoded = json.loads(raw)
        return Response(status=resp.status, json=decoded, raw=raw, headers=resp_headers)

    def get(self, path: str, headers: dict | None = None) -> Response:
        return self._request("GET", path, None, headers)

    def post(self, path: str, body: dict | None = None) -> Response:
        return self._request("POST", path, body or {})

    # --- command queue pump (the main-loop drain step, run synchronously) ---

    def pump(self) -> None:
        """Drain every queued command through the real dispatcher, like the main loop."""
        q = self.runtime.command_queue
        assert q is not None
        while True:
            try:
                cmd = q.get_nowait()
            except queue.Empty:
                break
            effective = self.runtime.active_config or self.config
            app.handle_command(cmd, effective, self.runtime, self.session)
            if self.runtime.prompt_ready is not None:
                self.runtime.prompt_ready.set()

    def do(self, path: str, body: dict | None = None) -> Response:
        """POST a control verb then immediately pump — the click-and-settle step."""
        resp = self.post(path, body)
        self.pump()
        return resp

    def time_post(self, path: str, body: dict | None = None) -> float:
        """Milliseconds the POST itself takes — the latency the UI sees on click.

        POST only enqueues and returns; the work happens in pump(). A small value
        here proves the click doesn't block on the operation behind it.
        """
        start = time.monotonic()
        self.post(path, body)
        return (time.monotonic() - start) * 1000.0

    def time_pump(self) -> float:
        """Milliseconds to drain the queue — where the operation's real cost lands."""
        start = time.monotonic()
        self.pump()
        return (time.monotonic() - start) * 1000.0

    def status(self) -> dict:
        return self.get("/api/status").json


def build_runtime(config: ConfigDict) -> tuple[RuntimeState, SessionState]:
    robot = ROBOT_REGISTRY.build(config.robot.type)
    transport = build_transport(config, robot)
    session = SessionState()
    runtime = RuntimeState(
        robot=robot,
        transport=transport,
        command_queue=queue.Queue(),
        prompt_ready=threading.Event(),
    )
    return runtime, session


_DEFAULTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "00_base" / "defaults.py"

# Legacy flat kwargs -> new nested (section, key) location.
_LEGACY_KWARGS = {
    "robot_type": ("robot", "type"),
    "publish_rate": ("inference_cfg", "publish_rate"),
}


def console_config(**overrides: Any) -> ConfigDict:
    """A minimal offline console config: debug transport + mock policy + one strategy.

    Builds on the project defaults (``configs/00_base/defaults.py``) so the returned
    ConfigDict carries every section the runtime reads, then layers the offline
    transport/policy and any per-test overrides on top.
    """
    cfg = load_config(_DEFAULTS_PATH)
    cfg.transport.type = "debug"
    cfg.policy.type = "mock"
    cfg.policy.backend_options = ConfigDict(dict(chunk_size=8))
    cfg.inference_cfg.publish_rate = 1000
    cfg.inference_cfg.debug_tasks = ("pick up the cup", "pour soybean")
    cfg.inference_strategies = ConfigDict(SYNC_STRATEGY)

    if "supported_inference_strategies" in overrides:
        cfg.inference_strategies = ConfigDict(overrides.pop("supported_inference_strategies"))
    for flat, (section, key) in _LEGACY_KWARGS.items():
        if flat in overrides:
            cfg[section][key] = overrides.pop(flat)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


@contextlib.contextmanager
def serve_console(config: ConfigDict) -> Generator[WebHarness]:
    """Run a console server on an ephemeral port for the duration of the block.

    Builds the server directly (not via start_console_server) so the test owns the
    socket lifecycle. ``scene=None`` disables the 3D canvas (needs URDF/trimesh and is
    irrelevant to the API contract).
    """
    runtime, session = build_runtime(config)
    port = free_port()
    ctx = ConsoleContext(
        config=config,
        runtime=runtime,
        session=session,
        obs_reader=runtime.transport.create_observation_reader(),
        scene=None,
    )
    ConsoleRequestHandler.ctx = ctx
    server = ThreadingHTTPServer(("127.0.0.1", port), ConsoleRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield WebHarness(config, runtime, session, port)
    finally:
        server.shutdown()
        server.server_close()
        runtime.transport.close()
