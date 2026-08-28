#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FRANKA_DIR="$REPO_ROOT/examples/hardware/franka"
cd "${REPO_ROOT}"
FRANKA_VENV_DIR="${FRANKA_VENV_DIR:-$FRANKA_DIR/.venv}"
FRANKA_PYTHON_BIN="${FRANKA_VENV_DIR}/bin/python"
[[ -x "${FRANKA_PYTHON_BIN}" ]] || {
  echo "Franka environment not found: ${FRANKA_VENV_DIR}" >&2
  echo "Run: UV_PROJECT_ENVIRONMENT=\"${FRANKA_VENV_DIR}\" uv sync --project \"${FRANKA_DIR}\"" >&2
  exit 1
}
export VIRTUAL_ENV="${FRANKA_VENV_DIR}"
export PATH="${FRANKA_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

exec "${FRANKA_PYTHON_BIN}" -X faulthandler examples/hardware/franka/fake_node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --rate "${PUBLISH_RATE:-30}" \
  --dynamics-rate "${DYNAMICS_RATE:-200}" \
  --natural-frequency-hz "${NATURAL_FREQUENCY_HZ:-2}" \
  --damping-ratio "${DAMPING_RATIO:-1}" \
  "$@"
