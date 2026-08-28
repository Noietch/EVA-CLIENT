#!/usr/bin/env bash
# Start the software-only ARX R5 ZMQ node from this adapter's isolated environment.
# Inputs are endpoint and dynamics environment variables plus forwarded CLI arguments.
# The script changes to the repository root, prepares import paths, and runs fake_node.py.
# It performs no hardware setup and writes no durable output itself.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARX_R5_VENV_DIR="${REPO_DIR}/examples/hardware/arx_r5/.venv"
ARX_R5_PYTHON_BIN="${ARX_R5_VENV_DIR}/bin/python"
[[ -x "${ARX_R5_PYTHON_BIN}" ]] || {
  printf '[arx_r5-fake] ERROR: ARX R5 environment not found: %s; run examples/hardware/arx_r5/setup_env.sh\n' \
    "${ARX_R5_VENV_DIR}" >&2
  exit 1
}

cd "${REPO_DIR}"
export VIRTUAL_ENV="${ARX_R5_VENV_DIR}"
export PATH="${ARX_R5_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

"${ARX_R5_PYTHON_BIN}" -X faulthandler examples/hardware/arx_r5/fake_node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --rate "${PUBLISH_RATE:-30}" \
  --dynamics-rate "${DYNAMICS_RATE:-200}" \
  --natural-frequency-hz "${NATURAL_FREQUENCY_HZ:-2}" \
  --damping-ratio "${DAMPING_RATIO:-1}" \
  "$@"
