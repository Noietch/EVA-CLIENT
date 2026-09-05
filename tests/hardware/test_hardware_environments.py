"""Validate hardware project metadata and parse every shell entrypoint."""

import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

pytestmark = pytest.mark.static
HARDWARE_ROOT = Path(__file__).resolve().parents[2] / "examples/hardware"


def test_hardware_projects_and_entrypoints():
    names = set()
    for directory in sorted(HARDWARE_ROOT.iterdir()):
        if not directory.is_dir() or directory.name.startswith("__"):
            continue
        project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
        assert project["name"] not in names, directory
        names.add(project["name"])
        assert SpecifierSet(project["requires-python"]), directory
        for script in sorted(directory.glob("*.sh")):
            subprocess.run(["bash", "-n", str(script)], check=True, timeout=10)
