"""Command-line workstation workflow shared with DEVICE.

Loads discovered example defaults and local overrides, edits device selections,
and controls the detached device service. Motion enable/disable targets the running
EVA console. Exiting this CLI never terminates hardware processes.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

import yaml

from core.config import load_config
from core.devices import REPOSITORY_ROOT, DeviceWorkspace
from core.devices.service import DeviceService


class DeviceCLI:
    def __init__(self, workspace: Path | None, url: str) -> None:
        self.workspace = DeviceWorkspace(workspace)
        self.service = DeviceService(self.workspace.path)
        self.url = url.rstrip("/")
        self.http_client = build_opener(
            *(
                [ProxyHandler({})]
                if urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}
                else []
            )
        )

    def http(self, path: str, body: dict | None = None) -> dict:
        request = Request(self.url + path)
        if body is not None:
            request.data = json.dumps(body).encode()
            request.add_header("Content-Type", "application/json")
        with self.http_client.open(request, timeout=3) as response:
            result = json.load(response)
        if result.get("ok") is False:
            raise ValueError(result["error"])
        return result

    def apply(self, selected: dict, values: dict) -> None:
        payload = {"selected": selected, "values": values}
        try:
            previous = self.http("/api/device_settings")
        except URLError:
            self.service.request("save", payload)
            return
        self.http("/api/device_selection", payload)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            time.sleep(0.25)
            try:
                status = self.http("/api/device_settings")
            except URLError:
                continue
            if status["state"] == "failed":
                raise ValueError(status["error"])
            if status["boot_id"] != previous["boot_id"]:
                return
        raise TimeoutError("EVA has not completed the device configuration reload")

    def run(self, args: argparse.Namespace) -> object:
        config = load_config(REPOSITORY_ROOT / "configs/00_base/defaults.py")
        selected = self.workspace.initial_selection(config)
        values = self.workspace.resolve(selected)
        if args.action == "list":
            return {
                kind: [
                    {"id": name, "label": spec["label"], "robots": spec.get("robots", [])}
                    for name, spec in devices.items()
                ]
                for kind, devices in self.workspace.catalog.items()
            }
        if args.action == "config":
            return {"selected": selected, "values": values}
        if args.action == "select":
            for kind in selected:
                choice = getattr(args, kind)
                if choice:
                    selected[kind] = choice
            values = self.workspace.resolve(selected)
            self.apply(selected, values)
            return {"selected": selected}
        if args.action == "set":
            kind, *parts = args.key.split(".")
            if kind not in values or not parts:
                raise ValueError(f"Unknown device parameter: {args.key}")
            target = values[kind]
            if parts[0] not in target:
                target = (
                    target["client"] if parts[0] in target.get("client", {}) else target["settings"]
                )
            for part in parts[:-1]:
                target = target[part]
            if parts[-1] not in target:
                raise ValueError(f"Unknown device parameter: {args.key}")
            target[parts[-1]] = yaml.safe_load(args.value)
            self.apply(selected, values)
            return {"updated": args.key}
        if args.action == "profile":
            path = self.workspace.import_profile(selected["camera"], args.path.read_text())
            _, settings = self.workspace.profile_values(selected["camera"], str(path))
            values["camera"]["settings"].update(settings)
            self.apply(selected, values)
            return {"profile": str(path)}
        if args.action in {"start", "stop", "status", "logs"}:
            action = "status" if args.action == "logs" else args.action
            component = getattr(args, "component", None)
            if action in {"start", "stop"}:
                try:
                    self.http("/api/status")
                except URLError:
                    pass
                else:
                    self.http("/api/device_" + action, {"component": component})
                    deadline = time.monotonic() + 20
                    expected = {component} if component else set(self.workspace.commands())
                    while time.monotonic() < deadline:
                        time.sleep(0.1)
                        state = self.http("/api/status")
                        if state.get("last_error"):
                            raise ValueError(state["last_error"])
                        processes = self.service.request("status")["processes"]
                        complete = (
                            all(name in processes and processes[name] is None for name in expected)
                            if action == "start"
                            else not expected.intersection(processes)
                        )
                        if complete:
                            break
                    else:
                        raise TimeoutError(f"Device {action} did not complete")
                    action = "status"
            result = self.service.request(action, {"component": component})
            if result.get("error"):
                raise ValueError(result["error"])
            if args.action == "logs":
                return result["log"]
            result.pop("browser_url", None)
            result.pop("commands", None)
            result.pop("log", None)
            return result
        self.http("/api/tab_switch", {"tab": "manual"})
        if args.action == "connect":
            return self.http("/api/status")
        enabled = args.action == "enable"
        return self.http("/api/device_control", {"enabled": enabled})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("list", "config", "status", "connect", "enable", "disable"):
        actions.add_parser(name)
    select = actions.add_parser("select")
    for kind in ("robot", "teleop", "camera"):
        select.add_argument("--" + kind)
    change = actions.add_parser("set")
    change.add_argument("key")
    change.add_argument("value")
    for name in ("start", "stop", "logs"):
        actions.add_parser(name).add_argument(
            "component", nargs="?", choices=("robot", "teleop", "camera")
        )
    profile = actions.add_parser("profile").add_subparsers(dest="profile_action", required=True)
    load = profile.add_parser("load")
    load.add_argument("component", choices=("camera",))
    load.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    import robots  # noqa: F401

    try:
        result = DeviceCLI(args.workspace, args.url).run(args)
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:
        print(f"eva device: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(result if isinstance(result, str) else yaml.safe_dump(result, sort_keys=False), end="\n")
