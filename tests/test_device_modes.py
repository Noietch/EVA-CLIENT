"""Robot mode selection preserves the model and uses the shared fake node."""

import sys

import pytest

import robots  # noqa: F401
from core.devices import DeviceWorkspace


pytestmark = pytest.mark.unit


def test_robot_modes(tmp_path):
    workspace = DeviceWorkspace(tmp_path / "devices.yaml")
    for name, spec in workspace.catalog["robot"].items():
        selected = {"robot": name, **spec["defaults"]}
        values = workspace.resolve(selected, saved=False)
        assert values["robot"]["mode"] == "real"
        real = workspace.commands(selected, values).get("robot")
        values["robot"]["mode"] = "fake"
        workspace.save(selected, values)
        restored = DeviceWorkspace(workspace.path)
        assert restored.saved["selected"]["robot"] == name
        assert restored.resolve(selected)["robot"]["mode"] == "fake"
        fake = restored.commands()["robot"]
        assert fake[:4] == [sys.executable, "-m", "examples.hardware.fake_common", name]
        assert fake[fake.index("--dynamics-mode") + 1] == "direct"
        assert fake != real
        values["robot"]["mode"] = "invalid"
        with pytest.raises(ValueError, match="Robot mode"):
            workspace.save(selected, values)
