from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UR5E_DIR = REPO_ROOT / "examples" / "hardware" / "ur5e"
PYPROJECT = UR5E_DIR / "pyproject.toml"
SETUP_SCRIPT = UR5E_DIR / "setup_env.sh"
HARDWARE_SCRIPT = UR5E_DIR / "run_hardware.sh"
FAKE_SCRIPT = UR5E_DIR / "run_fake_node.sh"
GRIPPER_SCRIPT = UR5E_DIR / "tests" / "run_gripper_test.sh"
TELEOP_SCRIPT = UR5E_DIR / "tests" / "run_teleop_test.sh"
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_ur5e_shell_entrypoints_have_valid_bash_syntax() -> None:
    for script in (
        SETUP_SCRIPT,
        HARDWARE_SCRIPT,
        FAKE_SCRIPT,
        GRIPPER_SCRIPT,
        TELEOP_SCRIPT,
    ):
        subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, check=True)


def test_ur5e_uses_an_isolated_python_project() -> None:
    project = PYPROJECT.read_text()
    root_project = ROOT_PYPROJECT.read_text()
    setup = SETUP_SCRIPT.read_text()

    assert 'name = "eva-ur5e-hardware"' in project
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert 'alicia_d_sdk = { path = "../arx_r5/SDK/alicia_d", editable = true }' in project
    assert "ur-rtde==1.6.3" in project
    assert "pydhgripper==1.0.2" in project
    assert "UV_PROJECT_ENVIRONMENT" in setup
    assert 'uv sync --project "$UR5E_DIR"' in setup
    assert "source .venv/bin/activate" not in setup

    assert 'requires-python = ">=3.11,<3.12"' in root_project
    assert "[project.optional-dependencies]" in root_project
    assert "franky-control" not in root_project
    assert "franka = [" not in root_project
    assert "arx_r5 = [" not in root_project
    assert "camera = [" not in root_project
    assert "ur5e = [" not in root_project
    assert "alicia_d_sdk" not in root_project
    assert "pydhgripper==1.0.2" not in root_project
    assert "ur-rtde==1.6.3" not in root_project
    assert "pyorbbecsdk2" not in root_project


@pytest.mark.parametrize(
    "script_path",
    [HARDWARE_SCRIPT, FAKE_SCRIPT, GRIPPER_SCRIPT, TELEOP_SCRIPT],
    ids=lambda path: path.name,
)
def test_ur5e_scripts_use_the_local_hardware_environment(script_path: Path) -> None:
    script = script_path.read_text()

    assert "examples/hardware/ur5e/.venv" in script
    assert 'UR5E_PYTHON_BIN="${UR5E_VENV_DIR}/bin/python"' in script
    assert "source .venv/bin/activate" not in script
    assert "Run: bash examples/hardware/ur5e/setup_env.sh" in script


@pytest.mark.skipif(
    not (UR5E_DIR / ".venv" / "bin" / "python").is_file(),
    reason="the launcher contract assumes the isolated UR5e .venv",
)
def test_ur5e_fake_node_help_uses_the_local_environment() -> None:
    result = subprocess.run(
        ["bash", str(FAKE_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    help_text = result.stdout + result.stderr
    assert "--obs-endpoint" in help_text
    assert "--action-endpoint" in help_text
    assert "--natural-frequency-hz" in help_text
