#!/usr/bin/env bash
# Create the isolated Python 3.11 runtime used by every ARX R5 entrypoint.
# Input: an optional ARX_R5_PYTHON interpreter override and the vendored SDK trees.
# Output: examples/hardware/arx_r5/.venv, which may be a symlink to another local path.
# Side effects: recreates that environment, installs runtime packages, and verifies imports.
set -euo pipefail

ARX_R5_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${ARX_R5_DIR}/../../.." && pwd)"
ARX_R5_VENV_LINK="${ARX_R5_DIR}/.venv"
if [[ -L "${ARX_R5_VENV_LINK}" ]]; then
  ARX_R5_VENV_DIR="$(readlink -f "${ARX_R5_VENV_LINK}")"
else
  ARX_R5_VENV_DIR="${ARX_R5_VENV_LINK}"
fi
ARX_R5_SDK_DIR="${ARX_R5_DIR}/SDK/ARX_R5_python"
ARX_R5_PYTHON="${ARX_R5_PYTHON:-$(command -v python3.11 || true)}"

[[ -x "${ARX_R5_PYTHON}" ]] || {
  echo "[arx_r5-setup] Python 3.11 is required by the ARX R5 hardware environment" >&2
  exit 1
}

PYTHON_VERSION="$(${ARX_R5_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.11" ]] || {
  echo "[arx_r5-setup] Python 3.11 is required; got ${PYTHON_VERSION}" >&2
  exit 1
}

[[ -f "${ARX_R5_SDK_DIR}/bimanual/__init__.py" ]] || {
  echo "[arx_r5-setup] vendored ARX R5 SDK is missing: ${ARX_R5_SDK_DIR}" >&2
  exit 2
}
[[ -f "${ARX_R5_DIR}/SDK/alicia_d/pyproject.toml" ]] || {
  echo "[arx_r5-setup] vendored Alicia-D SDK is missing: ${ARX_R5_DIR}/SDK/alicia_d" >&2
  exit 2
}
if ! command -v uv >/dev/null 2>&1; then
  echo "[arx_r5-setup] uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

"${ARX_R5_PYTHON}" -m venv --clear "${ARX_R5_VENV_DIR}"
ARX_R5_PYTHON_BIN="${ARX_R5_VENV_DIR}/bin/python"
UV_PROJECT_ENVIRONMENT="${ARX_R5_VENV_DIR}" uv sync --project "${ARX_R5_DIR}" --python "${ARX_R5_PYTHON}"

ARX_R5_LIBRARY_PATH="${ARX_R5_SDK_DIR}/bimanual/lib:${ARX_R5_SDK_DIR}/bimanual/lib/arx_r5_src"
export LD_LIBRARY_PATH="${ARX_R5_VENV_DIR}/lib:${ARX_R5_LIBRARY_PATH}:/usr/local/lib:${LD_LIBRARY_PATH:-}"
SDK_MODULE="$(find "${ARX_R5_SDK_DIR}/bimanual/api/arx_r5_python" -maxdepth 1 \
  -type f -name 'arx_r5_python*.so' -print -quit 2>/dev/null || true)"

if [[ -z "${SDK_MODULE}" ]]; then
  "${ARX_R5_PYTHON_BIN}" -c 'import alicia_d_sdk; print(f"[arx_r5-setup] alicia_d_sdk import OK: {alicia_d_sdk.__file__}")'
  echo "[arx_r5-setup] WARN: build SDK/ARX_R5_python before real-hardware startup"
else
  PYTHONPATH="${ARX_R5_SDK_DIR}" "${ARX_R5_PYTHON_BIN}" - <<'PY'
import alicia_d_sdk
import bimanual

print(f"[arx_r5-setup] alicia_d_sdk import OK: {alicia_d_sdk.__file__}")
print(f"[arx_r5-setup] bimanual import OK: {bimanual.__file__}")
PY
fi

echo "[arx_r5-setup] environment ready: ${ARX_R5_VENV_DIR}"
echo "[arx_r5-setup] runtime: $(${ARX_R5_PYTHON_BIN} --version 2>&1)"
