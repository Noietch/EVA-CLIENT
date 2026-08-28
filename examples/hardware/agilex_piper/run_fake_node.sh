#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AGILEX_PIPER_DIR="$REPO_ROOT/examples/hardware/agilex_piper"
cd "${REPO_ROOT}"
AGILEX_PIPER_VENV_DIR="${AGILEX_PIPER_VENV_DIR:-$AGILEX_PIPER_DIR/.venv}"
AGILEX_PIPER_PYTHON_BIN="${AGILEX_PIPER_VENV_DIR}/bin/python"
[[ -x "${AGILEX_PIPER_PYTHON_BIN}" ]] || {
  echo "Agilex Piper environment not found: ${AGILEX_PIPER_VENV_DIR}" >&2
  echo "Run: UV_PROJECT_ENVIRONMENT=\"${AGILEX_PIPER_VENV_DIR}\" uv sync --project \"${AGILEX_PIPER_DIR}\"" >&2
  exit 1
}
export VIRTUAL_ENV="${AGILEX_PIPER_VENV_DIR}"
export PATH="${AGILEX_PIPER_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

exec "${AGILEX_PIPER_PYTHON_BIN}" -X faulthandler examples/hardware/agilex_piper/fake_node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --rate "${PUBLISH_RATE:-30}" \
  --dynamics-rate "${DYNAMICS_RATE:-200}" \
  --natural-frequency-hz "${NATURAL_FREQUENCY_HZ:-2}" \
  --damping-ratio "${DAMPING_RATIO:-1}" \
  "$@"
