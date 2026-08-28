from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARX_R5_DIR = REPO_ROOT / "examples" / "hardware" / "arx_r5"
SCRIPT = ARX_R5_DIR / "run_hardware.sh"
FAKE_SCRIPT = ARX_R5_DIR / "run_fake_node.sh"
SETUP_SCRIPT = ARX_R5_DIR / "setup_env.sh"
CALIBRATE_SCRIPT = ARX_R5_DIR / "utils" / "calibrate_white_balance_x3.sh"
PYPROJECT = ARX_R5_DIR / "pyproject.toml"
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
TELEOP = ARX_R5_DIR / "teleop.py"


def _arx_r5_env_ready() -> bool:
    return (ARX_R5_DIR / ".venv" / "bin" / "python").is_file()


def test_arx_r5_shell_entrypoints_have_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", str(FAKE_SCRIPT)], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", str(CALIBRATE_SCRIPT)], cwd=REPO_ROOT, check=True)


def test_arx_r5_uses_an_isolated_python_project() -> None:
    script = SCRIPT.read_text()
    fake_script = FAKE_SCRIPT.read_text()
    setup = SETUP_SCRIPT.read_text()
    calibrate_script = CALIBRATE_SCRIPT.read_text()
    project = PYPROJECT.read_text()
    root_project = ROOT_PYPROJECT.read_text()

    assert 'ARX_R5_DIR="${REPO_DIR}/examples/hardware/arx_r5"' in script
    assert 'ARX_R5_VENV_DIR="${ARX_R5_DIR}/.venv"' in script
    assert 'ARX_R5_PYTHON_BIN="${ARX_R5_VENV_DIR}/bin/python"' in script
    assert '"${ARX_R5_PYTHON_BIN}" -X faulthandler' in script
    assert "source .venv/bin/activate" not in script
    assert "source .venv/bin/activate" not in fake_script
    assert "source .venv/bin/activate" not in calibrate_script
    assert 'requires-python = ">=3.11,<3.12"' in root_project
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert "eva-client" in project
    assert "alicia_d_sdk" in project
    assert "pyserial==3.5" in project
    assert "pyorbbecsdk2" in project
    assert '"${ARX_R5_PYTHON}" -m venv --clear "${ARX_R5_VENV_DIR}"' in setup
    assert "UV_PROJECT_ENVIRONMENT" in setup
    assert 'uv sync --project "${ARX_R5_DIR}"' in setup
    assert "git clone" not in setup


def test_arx_r5_launcher_uses_only_the_arx_r5_sdk_layout() -> None:
    script = SCRIPT.read_text()
    setup = SETUP_SCRIPT.read_text()
    teleop = TELEOP.read_text()

    assert 'ARX_R5_SDK_DIR="${ARX_R5_DIR}/SDK/ARX_R5_python"' in setup
    assert 'ARX_R5_SDK_DIR="${ARX_R5_SDK_DIR:-$ARX_R5_DIR/SDK/ARX_R5_python}"' in script
    assert 'export VIRTUAL_ENV="${ARX_R5_VENV_DIR}"' in script
    assert 'export PATH="${ARX_R5_VENV_DIR}/bin:${PATH}"' in script
    assert 'PYTHONPATH="${ARX_R5_SDK_DIR}" "${ARX_R5_PYTHON_BIN}" - <<\'PY\'' in setup
    assert "--arx-r5-sdk-dir" in teleop
    assert "--arx-sdk-dir" not in teleop
    assert "args.arx_r5_sdk_dir" in teleop
    assert "examples/hardware/arx_x5/SDK" not in script
    assert "source .venv/bin/activate" not in setup


@pytest.mark.skipif(
    not _arx_r5_env_ready(),
    reason="the launcher contract assumes the isolated ARX R5 .venv",
)
def test_arx_r5_help_forwards_arguments_without_can_access() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    help_text = result.stdout + result.stderr
    assert "--left-can-port" in help_text
    assert "--right-can-port" in help_text
    assert "--orbbec-camera" in help_text
    assert "--orbbec-white-balance-file" in help_text
    assert "--disabled-arm" in help_text
    assert "slcand" not in help_text
    assert "ARX R5 environment not found" not in help_text


def test_arx_r5_entrypoints_use_arx_r5_prefixes() -> None:
    script = SCRIPT.read_text()
    fake_script = FAKE_SCRIPT.read_text()
    calibrate_script = CALIBRATE_SCRIPT.read_text()
    node = (ARX_R5_DIR / "node.py").read_text()
    camera = (ARX_R5_DIR / "camera.py").read_text()

    assert "[arx_r5-run]" in script
    assert "[arx-run]" not in script
    assert "[arx_r5-wb]" in calibrate_script
    assert "[arx-wb]" not in calibrate_script
    assert "examples/hardware/arx_r5/setup_env.sh" in script
    assert "examples/hardware/arx_r5/setup_env.sh" in fake_script
    assert "ARX R5 ZMQ node ready" in node
    assert "Unsupported ARX R5 collection control source" in node
    assert "ARX R5 passive collection mode cannot control HIL" in node
    assert "Activate the ARX R5 hardware environment" in camera
