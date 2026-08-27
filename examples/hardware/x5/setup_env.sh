#!/usr/bin/env bash
set -euo pipefail

X5_DIR="$(cd "$(dirname "$0")" && pwd)"
X5_VENV_LINK="${X5_DIR}/.venv"
if [[ -L "${X5_VENV_LINK}" ]]; then
  X5_VENV_DIR="$(readlink -f "${X5_VENV_LINK}")"
else
  X5_VENV_DIR="${X5_VENV_LINK}"
fi
X5_SDK_DIR="${X5_DIR}/SDK/X5"
X5_PYTHON="${X5_PYTHON:-$(command -v python3.12 || true)}"

[[ -x "${X5_PYTHON}" ]] || {
  echo "[x5-setup] Python 3.12 is required by the official ARX5_beta SDK-V2" >&2
  exit 1
}

PYTHON_VERSION="$(${X5_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.12" ]] || {
  echo "[x5-setup] Python 3.12 is required; got ${PYTHON_VERSION}" >&2
  exit 1
}

SDK_MODULE="$(find "${X5_SDK_DIR}/bimanual" -maxdepth 1 -type f \
  -name '__init__.cpython-312-*.so' -print -quit 2>/dev/null || true)"
[[ -n "${SDK_MODULE}" ]] || {
  echo "[x5-setup] official ARX5_beta SDK-V2 is missing: ${X5_SDK_DIR}" >&2
  exit 2
}

"${X5_PYTHON}" -m venv --clear "${X5_VENV_DIR}"
X5_PYTHON_BIN="${X5_VENV_DIR}/bin/python"
"${X5_PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${X5_PYTHON_BIN}" -m pip install "${X5_DIR}"

PYTHONPATH="${X5_SDK_DIR}" "${X5_PYTHON_BIN}" - <<'PY'
import numpy as np
import bimanual

pose = bimanual.forward_kinematics([0.0, 0.2, 0.3, 0.0, 0.0, 0.0])
qpos = bimanual.inverse_kinematics(pose)
back = bimanual.forward_kinematics(qpos)
if not np.allclose(pose, back, atol=1e-3):
    raise RuntimeError(f"official X5 SDK FK/IK self-check failed: {pose} != {back}")
print(f"[x5-setup] official ARX5_beta SDK-V2 {bimanual.__version__} import OK")
PY

echo "[x5-setup] environment ready: ${X5_VENV_DIR}"
echo "[x5-setup] runtime: $(${X5_PYTHON_BIN} --version 2>&1)"
