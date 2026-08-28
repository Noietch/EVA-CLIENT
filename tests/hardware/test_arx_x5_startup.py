from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARX_X5_DIR = REPO_ROOT / "examples" / "hardware" / "arx_x5"
SCRIPT = ARX_X5_DIR / "run_hardware.sh"
FAKE_SCRIPT = ARX_X5_DIR / "run_fake.sh"
RESET_SCRIPT = ARX_X5_DIR / "reset_hardware.sh"
SETUP_SCRIPT = ARX_X5_DIR / "setup_env.sh"
PYPROJECT = ARX_X5_DIR / "pyproject.toml"
PYTHON_VERSION = ARX_X5_DIR / ".python-version"
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
README = ARX_X5_DIR / "README.md"


def _arx_x5_env_is_ready() -> bool:
    python = ARX_X5_DIR / ".venv" / "bin" / "python"
    if not python.is_file():
        return False
    result = subprocess.run(
        [str(python), str(ARX_X5_DIR / "node.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return result.returncode == 0


def test_arx_x5_launcher_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", str(FAKE_SCRIPT)], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", str(RESET_SCRIPT)], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], cwd=REPO_ROOT, check=True)


def test_arx_x5_uses_an_isolated_python_project() -> None:
    script = SCRIPT.read_text()
    fake_script = FAKE_SCRIPT.read_text()
    setup = SETUP_SCRIPT.read_text()
    project = PYPROJECT.read_text()
    root_project = ROOT_PYPROJECT.read_text()

    assert PYTHON_VERSION.read_text().strip() == "3.12"
    assert "examples/hardware/arx_x5/.venv" in script
    assert 'ARX_X5_PYTHON_BIN="${ARX_X5_VENV_DIR}/bin/python"' in script
    assert '"${ARX_X5_PYTHON_BIN}" -X faulthandler' in script
    assert 'export PYTHONPATH="$PWD:$PWD/src"' in script
    assert 'source "${ARX_X5_VENV_DIR}/bin/activate"' not in script
    assert "ARX_X5_ROS_SETUP" not in script
    assert "source .venv/bin/activate" not in script
    assert "${PYTHONPATH:-}" not in script
    assert (
        'ARX_X5_VENV_DIR="${ARX_X5_VENV_DIR:-$REPO_DIR/examples/hardware/arx_x5/.venv}"'
        in fake_script
    )
    assert 'ARX_X5_PYTHON_BIN="${ARX_X5_VENV_DIR}/bin/python"' in fake_script
    assert 'export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src"' in fake_script
    assert 'exec "${ARX_X5_PYTHON_BIN}" -X faulthandler' in fake_script
    assert "${PYTHONPATH:-}" not in fake_script
    assert 'requires-python = ">=3.11,<3.12"' in root_project
    assert 'requires-python = ">=3.12,<3.13"' in project
    assert "pyrealsense2" in project
    assert '"${ARX_X5_PYTHON}" -m venv --clear "${ARX_X5_VENV_DIR}"' in setup
    assert '"${ARX_X5_PYTHON_BIN}" -m pip install "${ARX_X5_DIR}"' in setup
    assert 'ARX_X5_SDK_DIR="${ARX_X5_DIR}/SDK/X5"' in setup
    assert 'ARX_X5_SDK_SUBMODULE_PATH="examples/hardware/arx_x5/SDK/X5"' in setup
    assert 'ARX_X5_REPO_ROOT=""' in setup
    assert (
        'ARX_X5_REPO_ROOT="$(git -C "${ARX_X5_DIR}" rev-parse --show-toplevel 2>/dev/null)"'
    ) in setup
    assert (
        'git -C "${ARX_X5_REPO_ROOT}" submodule update --init --recursive -- '
        '"${ARX_X5_SDK_SUBMODULE_PATH}"'
    ) in setup
    assert 'submodule update --init --recursive -- "${ARX_X5_SDK_SUBMODULE_PATH}"' in setup
    assert "is not available outside a git checkout" in setup
    assert "git clone" not in setup
    assert "official ARX X5 SDK-V2" in setup
    assert "from bimanual import SingleArm" in setup
    assert "forward_kinematics" not in setup
    assert "inverse_kinematics" not in setup


def test_arx_x5_launcher_uses_only_the_arx_x5_sdk_layout() -> None:
    script = SCRIPT.read_text()

    assert 'ARX_X5_SDK_DIR="$PWD/examples/hardware/arx_x5/SDK/X5"' in script
    assert 'PYTHONPATH="$ARX_X5_SDK_DIR:$PYTHONPATH"' in script
    assert "import bimanual" in script
    assert "-name '__init__.cpython-312-*.so'" in script
    assert "official X5 SDK-V2 requires Python 3.12" in script
    assert "setup_env.sh" in script
    assert "AMENT_PREFIX_PATH" not in script

    assert "examples/hardware/arx_r5/SDK" not in script
    assert "ARX_R5_python" not in script
    assert "arx_r5_src" not in script
    assert "ARX_X5_LEGACY" not in script


def test_arx_x5_launcher_matches_r5_startup_defaults_and_node_contract() -> None:
    script = SCRIPT.read_text()
    readme = README.read_text()

    assert 'LEFT_CAN_PORT_VALUE="can1"' in script
    assert 'RIGHT_CAN_PORT_VALUE="can3"' in script
    assert 'LEFT_CAN_DEVICE="/dev/arxcan1"' in script
    assert 'RIGHT_CAN_DEVICE="/dev/arxcan3"' in script
    assert 'ARM_TYPE_VALUE="2"' in script
    assert '--gripper-open-pos "-3.4"' in script
    assert '--gripper-close-pos "0.1"' in script
    assert "LINK6_MASS" not in script
    assert 'ARX_X5_NODE_PATH="examples/hardware/arx_x5/node.py"' in script
    assert "configs/01_deploy/arx_x5/openpi_qpos.py" in script
    assert "examples/hardware/arx_x5/node.py" in script

    for value in (
        "INIT_ARX_CAN",
        "ENABLE_ARX_ARMS",
        "ARX_CAN_SLCAND_SPEED",
        "ARX_CAN_WAIT_SECONDS",
        "--realsense-resolution",
        "ZMQ observation endpoint",
        "ZMQ action endpoint",
    ):
        assert value in script

    assert "examples/hardware/arx_x5/SDK/X5" in readme
    assert "temp/X5-sdk" not in readme
    assert "src/robots/kinematics/pyroki.py" in readme
    assert "configs/02_collection/arx_x5_vr.py" in readme


@pytest.mark.skipif(
    not _arx_x5_env_is_ready(),
    reason="the launcher contract assumes a ready ARX X5 local .venv",
)
def test_arx_x5_help_forwards_arguments_without_sdk_or_can_access() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    help_text = result.stdout + result.stderr
    assert "--left-can-port" in help_text
    assert "--right-can-port" in help_text
    assert "--realsense-camera" in help_text
    assert "--realsense-resolution" in help_text
    assert "--disabled-arm" in help_text
    assert "--left-link6-mass-kg" not in help_text
    assert "--right-link6-mass-kg" not in help_text
    assert "X5 SDK root not found" not in help_text
    assert "slcand" not in help_text

    reset_result = subprocess.run(
        ["bash", str(RESET_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert reset_result.returncode == 0, reset_result.stdout + reset_result.stderr
    reset_help_text = reset_result.stdout + reset_result.stderr
    assert "[arx_x5-reset]" not in reset_help_text
    assert "resetting can" not in reset_help_text


def test_arx_x5_launcher_runs_only_the_hardware_node() -> None:
    script = SCRIPT.read_text()

    assert 'run_python "${ARX_X5_NODE_PATH}"' in script
    assert "start_child" not in script
    assert "VR_NODE_PATH" not in script
    assert "ROOT_VENV_EVA" not in script
    assert "EVA_CONFIG_PATH" not in script
    assert "--web-port" not in script
    assert "hardware node will enable both arms" in script
    assert 'python "$@" >/dev/null' not in script
    assert "> >(sed" in script


def test_arx_x5_reset_launcher_rebuilds_can_and_checks_sdk_health() -> None:
    script = SCRIPT.read_text()
    reset_script = RESET_SCRIPT.read_text()

    assert "RESET_ARX_CAN" in script
    assert "RESET_ARX_X5_HARDWARE" in script
    assert "stop_existing_hardware_node" in script
    assert 'kill -INT "${pids[@]}"' in script
    assert "offline_joints" in script
    assert "right_arm J2 probe" in script
    assert "J2 motion probe failed" in script
    assert "arm.disable()" in script
    assert "X5 reset failed" in script
    assert "export RESET_ARX_CAN=1" in reset_script
    assert "export RESET_ARX_X5_HARDWARE=1" in reset_script
    assert 'run_hardware.sh" "$@"' in reset_script
    assert "ARX方舟无限" in script


def test_arx_x5_launcher_clears_only_2025_gripper_faults_on_every_start() -> None:
    script = SCRIPT.read_text()
    sdk = (ARX_X5_DIR / "sdk.py").read_text()

    assert 'CLEAR_ARX_X5_GRIPPER_ERRORS="${CLEAR_ARX_X5_GRIPPER_ERRORS:-1}"' in script
    assert "clear_arx_x5_gripper_errors" in script
    assert "clear_gripper_error(can_port)" in script
    assert "ARX_X5_2025_GRIPPER_MOTOR_ID = 8" in sdk
    assert 'ARX_X5_MOTOR_CLEAR_ERROR_COMMAND = b"\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xfb"' in sdk
    assert "bus.send(ARX_X5_2025_GRIPPER_MOTOR_ID" in sdk
    assert "set_zero" not in sdk


def test_arx_x5_launcher_checks_ports_and_child_readiness() -> None:
    script = SCRIPT.read_text()

    assert "wait_for_startup_ports" in script
    assert "startup ports still in use" in script
    assert "local ports=(5555 5556)" in script
    assert "8080" not in script
    assert "43876" not in script
