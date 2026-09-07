"""Example defaults and persistent workstation device selections."""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import yaml

from core.cfg import Config, ConfigDict
from core.devices.hardware import HardwareCatalog
from core.registry import ROBOT_REGISTRY
from teleop_client import validate_client_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEVICE_KINDS = ("robot", "teleop", "camera")


class DeviceWorkspace:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(
            os.environ.get(
                "EVA_WORKSTATION_PATH", Path.home() / ".local/eva-client/hardware_config.yaml"
            )
        )
        self.path = self.path.expanduser().resolve()
        self.hardware = HardwareCatalog()
        self.catalog = self.hardware.catalog
        self.saved = yaml.safe_load(self.path.read_text()) if self.path.exists() else {}
        self.saved = self.saved or {}
        if self.saved and self.saved["selected"].get("teleop") in {"joint", "transport"}:
            # Retired UI selections migrate on read; preserve all parameter overrides.
            robot = self.saved["selected"]["robot"]
            self.saved["selected"]["teleop"] = self.catalog["robot"][robot]["defaults"]["teleop"]
        if self.saved and self.saved["selected"].get("camera") == "x5_d405":
            self.saved["selected"]["camera"] = "x5_d405_day"

    def initial_selection(self, config: ConfigDict) -> dict[str, str]:
        if self.saved:
            return dict(self.saved["selected"])
        return {
            "robot": config.robot.type,
            **self.catalog["robot"][config.robot.type]["defaults"],
        }

    def resolve(self, selected: dict[str, str], *, saved: bool = True) -> dict[str, dict]:
        if set(selected) != set(DEVICE_KINDS):
            raise ValueError("Select Robot, Teleop and Camera")
        if selected["robot"] not in self.catalog["robot"]:
            raise ValueError(f"Unknown robot: {selected['robot']}")
        options = self.hardware.options(selected["robot"])
        robot = ROBOT_REGISTRY.build(selected["robot"])
        result = {}
        for kind in DEVICE_KINDS:
            name = selected[kind]
            if name not in options[kind]:
                raise ValueError(f"{kind} {name} is not supported by {robot.name}")
            spec = options[kind][name]
            defaults = {
                key: copy.deepcopy(spec[key])
                for key in ("settings", "client", "safety", "config")
                if key in spec
            }
            if kind == "robot":
                defaults["mode"] = "real"
            override_key = f"{name}@{selected['robot']}" if kind == "teleop" else name
            overrides = (
                self.saved.get("overrides", {}).get(kind, {}).get(override_key, {}) if saved else {}
            )
            result[kind] = self.merge(defaults, overrides)
        return result

    @staticmethod
    def merge(defaults: dict, overrides: dict) -> dict:
        if not isinstance(overrides, dict) or set(overrides) - set(defaults):
            raise ValueError("Unknown device setting")
        result = copy.deepcopy(defaults)
        for key, value in overrides.items():
            if isinstance(defaults[key], dict):
                result[key] = DeviceWorkspace.merge(defaults[key], value)
            else:
                expected = defaults[key]
                if expected is None:
                    valid = value is None or isinstance(value, (int, float, list))
                elif isinstance(expected, bool):
                    valid = isinstance(value, bool)
                elif isinstance(expected, (int, float)):
                    valid = (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    )
                else:
                    valid = isinstance(value, type(expected))
                if not valid:
                    raise ValueError(f"Invalid value type for {key}")
                result[key] = copy.deepcopy(value)
        return result

    def save(self, selected: dict[str, str], values: dict[str, dict]) -> dict[str, dict]:
        defaults = self.resolve(selected, saved=False)
        if set(values) != set(DEVICE_KINDS):
            raise ValueError("Device settings must include Robot, Teleop and Camera")
        # ``mode`` is a workstation choice, not a hardware SDK setting. Keep it
        # outside the strict hardware settings merge for old saved workspaces.
        robot_mode = values["robot"].pop("mode", "real")
        resolved = {kind: self.merge(defaults[kind], values[kind]) for kind in DEVICE_KINDS}
        resolved["robot"]["mode"] = robot_mode
        if resolved["robot"]["mode"] not in {"real", "fake"}:
            raise ValueError("Robot mode must be real or fake")
        if "client" in resolved["teleop"]:
            robot = ROBOT_REGISTRY.build(selected["robot"])
            validate_client_config(
                resolved["teleop"]["client"],
                arm_group_names=tuple(group.name for group in robot.arm_groups),
            )
            if any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in resolved["teleop"]["safety"].values()
            ):
                raise ValueError("VR safety limits must be finite and positive")
        camera = self.catalog["camera"][selected["camera"]]
        profile_field = camera.get("profile_field")
        if profile_field and resolved["camera"]["settings"].get(profile_field):
            self.read_profile(selected["camera"], resolved["camera"]["settings"][profile_field])

        # Persist choices and parameter overrides without runtime state
        saved = copy.deepcopy(self.saved)
        saved["selected"] = dict(selected)
        overrides = saved.setdefault("overrides", {})
        for kind in DEVICE_KINDS:
            name = selected[kind]
            key = f"{name}@{selected['robot']}" if kind == "teleop" else name
            overrides.setdefault(kind, {})[key] = self.difference(defaults[kind], resolved[kind])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                yaml.safe_dump(saved, stream, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        self.saved = saved
        return resolved

    @staticmethod
    def difference(defaults: dict, values: dict) -> dict:
        result = {}
        for key, value in values.items():
            if isinstance(value, dict):
                changed = DeviceWorkspace.difference(defaults[key], value)
                if changed:
                    result[key] = changed
            elif value != defaults[key]:
                result[key] = value
        return result

    def configure(self, config: ConfigDict) -> ConfigDict:
        if config.transport.get("type") == "dataset":
            return self.hardware.compose(config)
        selected = self.initial_selection(config)
        values = self.resolve(selected)
        config = self.hardware.compose(
            config, selected["robot"], settings=values["robot"]["settings"]
        )
        overrides = self.saved.get("overrides", {}).get("robot", {}).get(selected["robot"], {})
        config = ConfigDict(Config._merge_a_into_b(overrides.get("config", {}), config))
        if values["robot"]["mode"] == "fake":
            config.transport.type = "zmq"
            config.transport.sub_endpoint = values["robot"]["settings"].get(
                "obs_endpoint", "tcp://127.0.0.1:5555"
            )
            config.transport.pub_endpoint = values["robot"]["settings"].get(
                "action_endpoint", "tcp://127.0.0.1:5556"
            )
        robot = ROBOT_REGISTRY.build(config.robot.type)
        options = self.hardware.options(selected["robot"])
        if options["camera"][selected["camera"]].get("disabled"):
            config.transport.disabled_cameras = [
                camera.observation_key for camera in robot.observation_schema.cameras
            ]
            config.collection.schema.cameras = {}
        else:
            config.collection.schema.cameras = {
                camera_id: column
                for camera_id, column in config.collection.schema.cameras.items()
                if camera_id not in config.transport.disabled_cameras
            }
        if "client" in values["teleop"]:
            config.collection.teleop = ConfigDict(
                control_source="client",
                client=values["teleop"]["client"],
                safety=values["teleop"]["safety"],
            )
        else:
            config.collection.teleop.control_source = "transport"
            config.collection.teleop.client = {}
        return config

    def commands(
        self, selected: dict | None = None, values: dict | None = None
    ) -> dict[str, list[str]]:
        selected = self.saved["selected"] if selected is None else selected
        values = self.resolve(selected) if values is None else values
        commands = {}
        options = self.hardware.options(selected["robot"])
        for kind in DEVICE_KINDS:
            spec = options[kind][selected[kind]]
            # Fake robots must not start physical leader adapters such as YAM CAN.
            if (
                kind == "teleop"
                and values["robot"].get("mode", "real") == "fake"
                and spec.get("operation") == "leader"
            ):
                continue
            if kind == "robot" and values[kind].get("mode", "real") == "fake":
                settings = values[kind]["settings"]
                commands[kind] = [
                    sys.executable, "-m", "examples.hardware.fake_common",
                    selected["robot"],
                    "--dynamics-mode", "direct",
                    "--obs-endpoint", settings.get("obs_endpoint", "tcp://127.0.0.1:5555"),
                    "--action-endpoint", settings.get("action_endpoint", "tcp://127.0.0.1:5556"),
                    "--rate", str(settings.get("rate", 30)),
                ]
                continue
            launch = (
                self.catalog["robot"][selected["robot"]].get("launch")
                if kind == "camera" and spec.get("separate")
                else spec.get("launch")
            )
            if launch is None:
                continue
            settings = dict(values[kind]["settings"])
            if kind == "robot":
                if "client" not in values["teleop"]:
                    settings.update(values["teleop"]["settings"])
                if launch.get("disable_camera_option"):
                    robot = ROBOT_REGISTRY.build(selected["robot"])
                    settings[launch["disable_camera_option"]] = [
                        camera.observation_key for camera in robot.observation_schema.cameras
                    ]
            elif kind == "camera":
                settings.update(
                    camera_only=True, camera_endpoint=values["robot"]["settings"]["camera_endpoint"]
                )
            else:
                client = values["teleop"]["client"]
                settings.update(endpoint=client["endpoint"], ack_endpoint=client["ack_endpoint"])
            command = [
                str(REPOSITORY_ROOT / launch["python"]),
                "-m",
                launch["module"],
                *(launch.get("fixed", []) if kind != "camera" else []),
            ]
            repeated = set(spec.get("repeat", []))
            if kind == "robot":
                repeated.update(options["teleop"][selected["teleop"]].get("repeat", []))
                if launch.get("disable_camera_option"):
                    repeated.add(launch["disable_camera_option"])
            for key, value in settings.items():
                if value is None or value == "" or value is False or value == []:
                    continue
                flag = "--" + key.replace("_", "-")
                if value is True:
                    command.append(flag)
                elif isinstance(value, list):
                    if key in repeated:
                        for item in value:
                            rendered = json.dumps(item) if isinstance(item, dict) else str(item)
                            command.extend((flag, rendered))
                    else:
                        command.extend((flag, *map(str, value)))
                elif isinstance(value, dict):
                    command.extend((flag, json.dumps(value, separators=(",", ":"))))
                else:
                    command.extend((flag, str(value)))
            commands[kind] = command
        return commands

    def read_profile(self, camera: str, filename: str) -> dict:
        spec = self.catalog["camera"][camera]
        path = (REPOSITORY_ROOT / filename).resolve()
        allowed = [(REPOSITORY_ROOT / name).resolve() for name in spec.get("profiles", [])]
        if path not in allowed and not path.is_relative_to(
            (self.path.parent / "profiles").resolve()
        ):
            raise ValueError("Select a bundled profile or upload a camera profile")
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Camera profile exceeds 1 MiB")
        if "profile_loader" not in spec:
            raise ValueError("This camera adapter does not support profiles")
        module, name = spec["profile_loader"].split(":")
        loader = getattr(importlib.import_module(module), name)
        for options in spec.get("profile_variants", [{}]):
            loader(path, **options)
        return yaml.safe_load(path.read_text())

    def import_profile(self, camera: str, content: str) -> Path:
        if not isinstance(content, str) or len(content.encode()) > 1024 * 1024:
            raise ValueError("Camera profile must be at most 1 MiB")
        root = self.path.parent / "profiles"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, suffix=".yaml", delete=False
        ) as stream:
            path = Path(stream.name)
            stream.write(content)
        try:
            self.read_profile(camera, str(path))
        except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError):
            path.unlink(missing_ok=True)
            raise
        return path

    def profile_values(self, camera: str, filename: str) -> tuple[dict, dict]:
        spec = self.catalog["camera"][camera]
        data = self.read_profile(camera, filename)
        settings = {spec["profile_field"]: filename}
        if mapping := spec.get("serial_mapping"):
            settings[mapping] = [
                f"{name}={profile['serial']}" for name, profile in data["cameras"].items()
            ]
        return data, settings
