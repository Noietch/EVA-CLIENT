"""Robot hardware defaults and the devices each robot supports."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from core.cfg import Config, ConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class HardwareCatalog:
    def __init__(self) -> None:
        self.catalog: dict[str, dict] = {kind: {} for kind in ("robot", "teleop", "camera")}
        for path in sorted((REPOSITORY_ROOT / "examples/input_sources").glob("*/config.yaml")):
            for kind, devices in yaml.safe_load(path.read_text()).items():
                self.catalog[kind].update(devices)
        templates = copy.deepcopy(self.catalog)
        self.devices: dict[str, dict] = {}
        for path in sorted((REPOSITORY_ROOT / "examples/hardware").glob("*/config.yaml")):
            data = yaml.safe_load(path.read_text())
            for name, spec in data["robot"].items():
                self.catalog["robot"][name] = spec
                self.devices[name] = {}
                for kind in ("teleop", "camera"):
                    options = {
                        device: Config._merge_a_into_b(values, templates[kind].get(device, {}))
                        for device, values in data[kind].items()
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
