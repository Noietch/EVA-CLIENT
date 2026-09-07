"""Robot hardware defaults and the devices each robot supports."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from core.cfg import Config, ConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class HardwareCatalog:
    @staticmethod
    def _expand_camera_combinations(data: dict, path: Path) -> dict[str, dict]:
        definitions = data.get("camera_devices", {})
        options = copy.deepcopy(data["camera"])
        for option_name, option in options.items():
            cameras = option.pop("cameras", None)
            if not cameras:
                continue
            settings = option.setdefault("settings", {})
            repeated = list(option.get("repeat", []))
            for camera_name in cameras:
                if camera_name not in definitions:
                    raise ValueError(f"{path}: Unknown camera device {camera_name!r}")
                camera = definitions[camera_name]
                for key, value in camera.get("settings", {}).items():
                    if isinstance(value, list):
                        settings.setdefault(key, []).extend(copy.deepcopy(value))
                    elif isinstance(value, dict):
                        target = settings.setdefault(key, {})
                        if not isinstance(target, dict):
                            raise ValueError(
                                f"{path}: Camera combination {option_name!r} "
                                f"has conflicting {key!r}"
                            )
                        overlap = set(target) & set(value)
                        if any(target[name] != value[name] for name in overlap):
                            raise ValueError(
                                f"{path}: Camera combination {option_name!r} "
                                f"has conflicting {key!r}"
                            )
                        target.update(copy.deepcopy(value))
                    elif key not in settings:
                        settings[key] = copy.deepcopy(value)
                    elif settings[key] != value:
                        raise ValueError(
                            f"{path}: Camera combination {option_name!r} has conflicting {key!r}"
                        )
                repeated.extend(camera.get("repeat", []))
                for key in ("profile_field", "serial_mapping", "profile_loader", "profiles"):
                    if key not in camera:
                        continue
                    if key in option and option[key] != camera[key]:
                        raise ValueError(
                            f"{path}: Camera combination {option_name!r} has conflicting {key!r}"
                        )
                    option[key] = copy.deepcopy(camera[key])
            if repeated:
                option["repeat"] = list(dict.fromkeys(repeated))
        return options

    def __init__(self) -> None:
        self.catalog: dict[str, dict] = {kind: {} for kind in ("robot", "teleop", "camera")}
        for path in sorted((REPOSITORY_ROOT / "examples/input_sources").glob("*/config.yaml")):
            for kind, devices in yaml.safe_load(path.read_text()).items():
                self.catalog[kind].update(devices)
        templates = copy.deepcopy(self.catalog)
        self.devices: dict[str, dict] = {}
        for path in sorted((REPOSITORY_ROOT / "examples/hardware").glob("*/config.yaml")):
            data = yaml.safe_load(path.read_text())
            camera_options = self._expand_camera_combinations(data, path)
            for name, spec in data["robot"].items():
                self.catalog["robot"][name] = spec
                self.devices[name] = {}
                for kind in ("teleop", "camera"):
                    device_specs = camera_options if kind == "camera" else data[kind]
                    options = {
                        device: Config._merge_a_into_b(values, templates[kind].get(device, {}))
                        for device, values in device_specs.items()
                    }
                    if kind == "teleop":
                        modes = [option.get("operation") for option in options.values()]
                        if not modes or any(mode not in {"vr", "leader"} for mode in modes):
                            raise ValueError(f"{path}: Operation must be VR or Leader")
                        if len(set(modes)) != len(modes):
                            raise ValueError(f"{path}: Declare at most one adapter per Operation")
                        if spec["defaults"]["teleop"] not in options:
                            raise ValueError(f"{path}: Default Operation must be supported")
                    spec[kind] = list(options)
                    self.devices[name][kind] = options
                    for device, values in options.items():
                        if device not in templates[kind]:
                            self.catalog[kind][device] = values

    def options(self, robot_type: str) -> dict[str, dict]:
        return {"robot": self.catalog["robot"], **self.devices[robot_type]}

    def compose(
        self, config: ConfigDict, robot_type: str | None = None, *, settings: dict | None = None
    ) -> ConfigDict:
        """Apply hardware defaults underneath model, task and dataset overrides."""
        name = robot_type or config.robot.type
        spec = self.catalog["robot"][name]
        defaults = copy.deepcopy(spec["config"])
        settings = spec["settings"] if settings is None else settings
        for setting, field in (
            ("obs_endpoint", "sub_endpoint"), ("action_endpoint", "pub_endpoint")
        ):
            if setting in settings:
                defaults["transport"][field] = settings[setting]
        merged = ConfigDict(Config._merge_a_into_b(config, defaults))
        merged.robot.type = name
        return merged
