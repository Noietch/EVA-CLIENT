"""Detached workstation service shared by DEVICE and the CLI.

The service reads example defaults and the workstation file, starts device children,
and exposes local owner-only RPC over a Unix socket. EVA disconnects do not stop it.
Only explicit stop requests terminate owned hardware processes.
"""

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from core.devices import DEVICE_KINDS, REPOSITORY_ROOT, DeviceWorkspace
from core.devices.preflight import check_yam_can


class DeviceProcesses:
    def __init__(self, workspace_path: Path) -> None:
        self.workspace = DeviceWorkspace(workspace_path)
        self.boot_id = str(time.monotonic_ns())
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_root = REPOSITORY_ROOT / "logs/device" / self.boot_id
        self.browser_url = ""
        self.commands = {}
        self.ready_pids = {}

    def start(self, component: str | None = None) -> None:
        self.workspace = DeviceWorkspace(self.workspace.path)
        if not self.workspace.saved:
            raise ValueError("Apply the device selection before starting")
        commands = self.workspace.commands()
        if component is not None:
            if component not in commands:
                raise ValueError(f"No local launcher for {component}")
            commands = {component: commands[component]}
        commands = {
            name: command
            for name, command in commands.items()
            if name not in self.processes or self.processes[name].poll() is not None
        }
        for command in commands.values():
            if not os.access(command[0], os.X_OK):
                raise ValueError(f"Device environment is missing: {command[0]}")
            check_yam_can(command, prepare=True)

        # Start only missing components and roll back this attempt on failure
        self.log_root.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            PYTHONPATH=f"{REPOSITORY_ROOT}:{REPOSITORY_ROOT / 'src'}",
        )
        started = []
        try:
            for name, command in commands.items():
                token = secrets.token_urlsafe(24) if "--token-stdin" in command else ""
                with (self.log_root / f"{name}.log").open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=REPOSITORY_ROOT,
                        env=env,
                        stdin=subprocess.PIPE if token else subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                self.processes[name] = process
                self.ready_pids.pop(name, None)
                self.commands[name] = command
                started.append(name)
                if token:
                    process.stdin.write((token + "\n").encode())
                    process.stdin.close()
                    settings = self.workspace.resolve(self.workspace.saved["selected"])[name][
                        "settings"
                    ]
                    scheme = "https" if settings["tls_cert"] else "http"
                    host = settings["public_host"] or settings["host"]
                    if host in {"0.0.0.0", "::"}:
                        host = "127.0.0.1"
                    self.browser_url = f"{scheme}://{host}:{settings['port']}/?token={token}"
        except OSError:
            for name in started:
                self.stop(name)
            raise

    def stop(self, component: str | None = None) -> None:
        names = list(self.processes) if component is None else [component]
        for name in names:
            process = self.processes.get(name)
            if process is None:
                continue
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if name == "robot":
                    raise ValueError("Robot did not finish shutdown; forced termination withheld")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            del self.processes[name]
            self.commands.pop(name, None)
            self.ready_pids.pop(name, None)
        if "teleop" not in self.processes:
            self.browser_url = ""

    def kill_all(self) -> None:
        """Immediately force-kill every owned device process group."""
        processes = list(self.processes.items())
        for _, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        errors = []
        for name, process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                errors.append(f"{name}: process did not exit after SIGKILL")
            finally:
                if process.poll() is not None:
                    self.processes.pop(name, None)
                    self.commands.pop(name, None)
                    self.ready_pids.pop(name, None)
        self.browser_url = ""
        if errors:
            raise RuntimeError("; ".join(errors))

    def mark_ready(self, component: str, pid: int) -> None:
        process = self.processes.get(component)
        if process is None or process.pid != pid or process.poll() is not None:
            raise ValueError("Device process changed or exited during startup")
        self.ready_pids[component] = pid

    def open_pico(self) -> None:
        process = self.processes.get("teleop")
        if (
            process is None or process.poll() is not None
            or self.ready_pids.get("teleop") != process.pid or not self.browser_url
        ):
            raise ValueError("Start VR successfully before opening Pico")
        url = urlsplit(self.browser_url)
        token = parse_qs(url.query)["token"][0]
        result = subprocess.run(
            ["bash", str(REPOSITORY_ROOT / "examples/input_sources/vr_webxr/open_pico.sh")],
            env=dict(os.environ, VR_TOKEN=token, VR_PORT=str(url.port), VR_SCHEME=url.scheme),
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode:
            message = (result.stderr or result.stdout or "Open Pico failed").replace(token, "[redacted]")
            raise ValueError(message[-1000:])

    def status(self, component: str | None = None) -> dict:
        processes = {name: process.poll() for name, process in self.processes.items()}
        failed = {name: code for name, code in processes.items() if code is not None}
        logs = []
        errors = []
        for name in DEVICE_KINDS if component is None else (component,):
            path = self.log_root / f"{name}.log"
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 8192))
                    output = stream.read().decode(errors="replace")
                    logs.append(f"[{name}]\n" + output)
                    if name in failed:
                        lines = [line.strip() for line in output.splitlines() if line.strip()]
                        if lines:
                            errors.append(f"{name}: {lines[-1][:1000]}")
        return {
            "state": "failed" if failed else "started" if processes else "stopped",
            "error": "\n".join(errors) or (f"Device process exited: {failed}" if failed else ""),
            "boot_id": self.boot_id,
            "processes": processes,
            "log": "\n".join(logs),
            "browser_url": self.browser_url,
            "commands": self.commands,
            "pids": {name: process.pid for name, process in self.processes.items()},
            "ready": {
                name: self.ready_pids.get(name) == process.pid and process.poll() is None
                for name, process in self.processes.items()
            },
        }


class DeviceSettings:
    def __init__(self) -> None:
        self.state = "idle"
        self.error = ""
        self.workspace = DeviceWorkspace()
        self.service = DeviceService(self.workspace.path)
        self.restart_requested = False
        self.boot_id = str(time.monotonic_ns())
        self.operations = {}

    def queue_command(self, action: str, component: str | None) -> str:
        names = (component,) if component else DEVICE_KINDS
        if action == "start" and any(
            operation["state"] in {"queued", "starting", "stopping"}
            for operation in self.operations.values()
        ):
            raise ValueError("A device command is already pending")
        request_id = secrets.token_hex(12)
        for name in names:
            self.operations[name] = {"id": request_id, "state": "queued", "error": ""}
        return request_id

    def kill_all(self) -> dict:
        """Immediately clear the device service and any stale UI operation state."""
        result = self.service.request("kill_all")
        self.operations.clear()
        self.state = "idle"
        self.error = ""
        return result

    def save_selection(self, payload, runtime, session) -> None:
        from core.app.state import SessionStatus

        self.state = "applying"
        self.error = ""
        try:
            if (
                runtime.collection_teleop_armed
                or runtime.collection_teleop_active
                or session.manual_publish_active
                or session.status is SessionStatus.RUNNING
                or runtime.rollout_save_ready
            ):
                raise ValueError("Stop control and finish recording before switching devices")
            self.service.request("save", payload)
            self.workspace = DeviceWorkspace(self.workspace.path)
            self.state = "restarting"
            self.restart_requested = True
            session.status = SessionStatus.EXIT
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.state = "failed"
            self.error = str(error)

    def status(self) -> dict:
        result = self.service.request("status")
        result.pop("commands", None)
        result["boot_id"] = self.boot_id
        result["operations"] = dict(self.operations)
        if self.state != "idle":
            result.update(state=self.state, error=self.error)
        return result


class DeviceService:
    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = DeviceWorkspace(workspace).path
        runtime = Path(tempfile.gettempdir()) / f"eva-devices-{os.getuid()}"
        runtime.mkdir(mode=0o700, exist_ok=True)
        info = runtime.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError(f"Device runtime directory must be private and owned: {runtime}")
        key = hashlib.blake2s(str(self.workspace).encode(), digest_size=8).hexdigest()
        self.socket_path = runtime / f"{key}.sock"

    def request(self, action: str, payload: dict | None = None) -> dict:
        deadline = time.monotonic() + 10
        spawned = False
        while True:
            connection = socket.socket(socket.AF_UNIX)
            connection.settimeout(150 if action == "start" else 30 if action == "kill_all" else 20)
            try:
                connection.connect(str(self.socket_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                connection.close()
                if action in {"status", "stop"}:
                    return {
                        "state": "stopped",
                        "error": "",
                        "processes": {},
                        "pids": {},
                        "commands": {},
                        "log": "",
                        "browser_url": "",
                    }
                if not spawned:
                    self.workspace.parent.mkdir(parents=True, exist_ok=True)
                    with (self.workspace.parent / "devices.log").open("ab") as log:
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "core.devices.service",
                                "--workspace",
                                str(self.workspace),
                            ],
                            cwd=REPOSITORY_ROOT,
                            env=dict(
                                os.environ,
                                PYTHONPATH=f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
                                PYTHONUNBUFFERED="1",
                            ),
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                            close_fds=True,
                        )
                    spawned = True
                if time.monotonic() >= deadline:
                    raise TimeoutError("Device service did not start; see devices.log") from None
                time.sleep(0.05)
        with connection, connection.makefile("rwb") as stream:
            stream.write(json.dumps({"action": action, "payload": payload or {}}).encode() + b"\n")
            stream.flush()
            raw = stream.readline(1024 * 1024)
            if not raw:
                raise RuntimeError("Device service closed the connection without a response")
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RuntimeError("Device service returned invalid JSON") from error
            if response.get("ok") is False:
                raise ValueError(response["error"])
            return response


class DeviceRequestHandler(socketserver.StreamRequestHandler):
    manager: DeviceProcesses

    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline(1024 * 1024))
            action = request["action"]
            component = request.get("payload", {}).get("component")
            if component is not None and component not in DEVICE_KINDS:
                raise ValueError(f"Unknown component: {component}")
            if action == "start":
                self.manager.start(component)
            elif action == "stop":
                self.manager.stop(component)
            elif action == "kill_all":
                self.manager.kill_all()
            elif action == "ready":
                self.manager.mark_ready(component, request["payload"]["pid"])
            elif action == "open_pico":
                self.manager.open_pico()
            elif action == "save":
                payload = request["payload"]
                self.manager.workspace = DeviceWorkspace(self.manager.workspace.path)
                values = payload.get("values")
                if values is None:
                    values = self.manager.workspace.resolve(payload["selected"])
                commands = self.manager.workspace.commands(payload["selected"], values)
                if any(
                    process.poll() is None and commands.get(name) != self.manager.commands.get(name)
                    for name, process in self.manager.processes.items()
                ):
                    raise ValueError("Stop affected devices before changing their launch settings")
                self.manager.workspace.save(payload["selected"], values)
            elif action != "status":
                raise ValueError(f"Unknown device operation: {action}")
            result = self.manager.status(component)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            result = {"ok": False, "error": str(error)}
        self.wfile.write(json.dumps(result).encode() + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    import robots  # noqa: F401

    os.umask(0o077)
    args.workspace.parent.mkdir(parents=True, exist_ok=True)
    service = DeviceService(args.workspace)
    with service.socket_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        service.socket_path.unlink(missing_ok=True)
        DeviceRequestHandler.manager = DeviceProcesses(args.workspace)
        with socketserver.UnixStreamServer(
            str(service.socket_path), DeviceRequestHandler
        ) as server:
            server.serve_forever()


if __name__ == "__main__":
    main()
