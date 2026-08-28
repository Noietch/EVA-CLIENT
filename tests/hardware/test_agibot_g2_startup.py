from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGIBOT_G2_DIR = REPO_ROOT / "examples" / "hardware" / "agibot_g2"
SETUP_SCRIPT = AGIBOT_G2_DIR / "setup_env.sh"
PYPROJECT = AGIBOT_G2_DIR / "pyproject.toml"
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_agibot_g2_setup_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], cwd=REPO_ROOT, check=True)


def test_agibot_g2_uses_an_isolated_python_project() -> None:
    setup = SETUP_SCRIPT.read_text()
    project = PYPROJECT.read_text()
    root_project = ROOT_PYPROJECT.read_text()

    assert 'name = "eva-agibot-g2-hardware"' in project
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert 'eva-client = { path = "../../..", editable = true }' in project
    assert (
        'UV_PROJECT_ENVIRONMENT="$AGIBOT_G2_VENV_DIR" uv sync --inexact --project "$AGIBOT_G2_DIR"'
    ) in setup
    assert "source .venv/bin/activate" not in setup
    assert 'requires-python = ">=3.11,<3.12"' in root_project


def test_agibot_g2_setup_supports_an_explicit_gdk_source() -> None:
    setup = SETUP_SCRIPT.read_text()

    assert 'AGIBOT_GDK_LOCAL_PATH="${AGIBOT_GDK_LOCAL_PATH:-}"' in setup
    assert 'AGIBOT_GDK_GIT_URL="${AGIBOT_GDK_GIT_URL:-}"' in setup
    assert 'if [[ -n "${AGIBOT_GDK_LOCAL_PATH}" && -n "${AGIBOT_GDK_GIT_URL}" ]]; then' in setup
    assert "Set exactly one of AGIBOT_GDK_LOCAL_PATH or AGIBOT_GDK_GIT_URL" in setup
    assert (
        'if [[ -n "${AGIBOT_GDK_LOCAL_PATH}" && ! -e "${AGIBOT_GDK_LOCAL_PATH}" ]]; then'
    ) in setup
    assert ('uv pip install --python "${AGIBOT_G2_PYTHON_BIN}" "${AGIBOT_GDK_LOCAL_PATH}"') in setup
    assert (
        'uv pip install --python "${AGIBOT_G2_PYTHON_BIN}" "git+${AGIBOT_GDK_GIT_URL}"'
    ) in setup


def test_agibot_g2_setup_distinguishes_hardware_and_offline_readiness() -> None:
    setup = SETUP_SCRIPT.read_text()

    assert "import agibot_gdk" in setup
    assert "AgiBot G2 hardware environment is ready" in setup
    assert "AgiBot G2 offline/fake environment is ready" in setup
    assert "Real hardware requires AGIBOT_GDK_LOCAL_PATH or AGIBOT_GDK_GIT_URL" in setup
    assert "agibot_gdk available:" in setup
    assert "opencv available:" not in setup
    assert "openpi client available:" not in setup
    assert "pyzmq available:" not in setup
    assert "The selected AgiBot GDK installed but agibot_gdk cannot be imported." in setup
