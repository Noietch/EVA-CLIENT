#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AGIBOT_G2_DIR="$REPO_ROOT/examples/hardware/agibot_g2"
cd "${REPO_ROOT}"
AGIBOT_G2_VENV_DIR="${AGIBOT_G2_VENV_DIR:-$AGIBOT_G2_DIR/.venv}"
AGIBOT_G2_PYTHON_BIN="${AGIBOT_G2_VENV_DIR}/bin/python"
[[ -x "${AGIBOT_G2_PYTHON_BIN}" ]] || {
  echo "AgiBot G2 environment not found: ${AGIBOT_G2_VENV_DIR}" >&2
  echo "Run: UV_PROJECT_ENVIRONMENT=\"${AGIBOT_G2_VENV_DIR}\" uv sync --project \"${AGIBOT_G2_DIR}\"" >&2
  exit 1
}
export VIRTUAL_ENV="${AGIBOT_G2_VENV_DIR}"
export PATH="${AGIBOT_G2_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

exec "${AGIBOT_G2_PYTHON_BIN}" -X faulthandler examples/hardware/agibot_g2/fake_node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --rate "${PUBLISH_RATE:-30}" \
  --dynamics-rate "${DYNAMICS_RATE:-200}" \
  --natural-frequency-hz "${NATURAL_FREQUENCY_HZ:-2}" \
  --damping-ratio "${DAMPING_RATIO:-1}" \
  "$@"
