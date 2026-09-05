from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.config import load_config
from tools.run_r1lite_fake_e2e import write_e2e_configs

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "tools" / "run_r1lite_fake_e2e.py"


def test_write_e2e_configs_pins_ports_and_output_dirs(tmp_path):
    paths = write_e2e_configs(tmp_path, policy_port=19000)

    configs = [load_config(path) for path in paths[:7]]
    for index, config in enumerate(configs):
        assert config.policy.port == 19000
        assert config.rollout.intervention.control_mode == "relative"
        section = config.rollout if index < 3 else config.collection
        if index < 5:
            assert Path(section.storage.log_dir) == paths[index + 7]
    evaluation = configs[5].eval
    assert evaluation.inference_strategy == "sync"
    assert evaluation.checkpoints[0].port == 19000
    assert Path(evaluation.output_dir) == paths.eval_dir
    rl = configs[6].rl_cfg
    assert rl.policies[0].name == "fake_policy"
    assert rl.policies[0].port == 19000
    assert rl.critics[0].name == "fake_critic"
    assert rl.critics[0].port == 19100
    assert Path(rl.data.storage.log_dir) == paths.rl_dir


def test_cli_probes_real_local_http_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        subprocess.run(
            [
                sys.executable,
                str(HARNESS_PATH),
                "--output",
                str(tmp_path),
                "--eva-url",
                url,
                "--fake-url",
                url,
                "--drive-motion",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["fake"] == {"ok": True}
    assert summary["eva"] == {"ok": True}
    assert len(summary["configs"]) == 7
    assert len([path for path, _ in calls if path == "/api/status"]) >= 2
    assert [payload["delta"] for path, payload in calls if path == "/api/command_delta"] == [
        0.03,
        -0.02,
        0.03,
        -0.02,
    ]
