#!/usr/bin/env bash
# Create the dedicated ARX X5 CPython 3.12 environment for hardware and fake-node use.
# The script reads this directory's pyproject and vendored SDK, recreates .venv, installs
# its dependencies, and validates the native SDK import without modifying the root environment.
set -euo pipefail

ARX_X5_DIR="$(cd "$(dirname "$0")" && pwd)"
ARX_X5_VENV_LINK="${ARX_X5_DIR}/.venv"
if [[ -L "${ARX_X5_VENV_LINK}" ]]; then
  ARX_X5_VENV_DIR="$(readlink -f "${ARX_X5_VENV_LINK}")"
else
  ARX_X5_VENV_DIR="${ARX_X5_VENV_LINK}"
fi
ARX_X5_SDK_DIR="${ARX_X5_DIR}/SDK/X5"
ARX_X5_SDK_SUBMODULE_PATH="examples/hardware/arx_x5/SDK/X5"
ARX_X5_PYTHON="${ARX_X5_PYTHON:-$(command -v python3.12 || true)}"
ARX_X5_REPO_ROOT=""

[[ -x "${ARX_X5_PYTHON}" ]] || {
  echo "[arx_x5-setup] Python 3.12 is required by the official ARX X5 SDK-V2" >&2
  exit 1
}

PYTHON_VERSION="$(${ARX_X5_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.12" ]] || {
  echo "[arx_x5-setup] Python 3.12 is required; got ${PYTHON_VERSION}" >&2
  exit 1
}

# Restore the SDK submodule on fresh clones before checking the native module
if [[ ! -d "${ARX_X5_SDK_DIR}/bimanual" ]]; then
  if ! ARX_X5_REPO_ROOT="$(git -C "${ARX_X5_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    echo "[arx_x5-setup] submodule ${ARX_X5_SDK_SUBMODULE_PATH} is not available outside a git checkout" >&2
    exit 2
  fi
  git -C "${ARX_X5_REPO_ROOT}" submodule update --init --recursive -- "${ARX_X5_SDK_SUBMODULE_PATH}"
fi

SDK_MODULE="$(find "${ARX_X5_SDK_DIR}/bimanual" -maxdepth 1 -type f \
  -name '__init__.cpython-312-*.so' -print -quit 2>/dev/null || true)"
[[ -n "${SDK_MODULE}" ]] || {
  echo "[arx_x5-setup] official ARX X5 SDK-V2 is missing: ${ARX_X5_SDK_DIR}" >&2
  exit 2
}

"${ARX_X5_PYTHON}" -m venv --clear "${ARX_X5_VENV_DIR}"
ARX_X5_PYTHON_BIN="${ARX_X5_VENV_DIR}/bin/python"
"${ARX_X5_PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${ARX_X5_PYTHON_BIN}" -m pip install "${ARX_X5_DIR}"

PYTHONPATH="${ARX_X5_SDK_DIR}" "${ARX_X5_PYTHON_BIN}" - <<'PY'
from bimanual import SingleArm, __version__

assert SingleArm is not None
print(f"[arx_x5-setup] official ARX X5 SDK-V2 {__version__} SingleArm import OK")
PY

echo "[arx_x5-setup] environment ready: ${ARX_X5_VENV_DIR}"
echo "[arx_x5-setup] runtime: $(${ARX_X5_PYTHON_BIN} --version 2>&1)"
