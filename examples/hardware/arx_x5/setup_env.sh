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
ARX_X5_PYTHON="${ARX_X5_PYTHON:-$(command -v python3.12 || true)}"

[[ -x "${ARX_X5_PYTHON}" ]] || {
  echo "[arx_x5-setup] Python 3.12 is required by the official ARX X5 SDK-V2" >&2
  exit 1
}

PYTHON_VERSION="$(${ARX_X5_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.12" ]] || {
  echo "[arx_x5-setup] Python 3.12 is required; got ${PYTHON_VERSION}" >&2
  exit 1
}

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
import numpy as np
import bimanual

pose = bimanual.forward_kinematics([0.0, 0.2, 0.3, 0.0, 0.0, 0.0])
qpos = bimanual.inverse_kinematics(pose)
back = bimanual.forward_kinematics(qpos)
if not np.allclose(pose, back, atol=1e-3):
    raise RuntimeError(f"official X5 SDK FK/IK self-check failed: {pose} != {back}")
print(f"[arx_x5-setup] official ARX X5 SDK-V2 {bimanual.__version__} import OK")
PY

echo "[arx_x5-setup] environment ready: ${ARX_X5_VENV_DIR}"
echo "[arx_x5-setup] runtime: $(${ARX_X5_PYTHON_BIN} --version 2>&1)"
