#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
UR5E_DIR="$REPO_ROOT/examples/hardware/ur5e"
cd "${REPO_ROOT}"

UR5E_VENV_DIR="${UR5E_VENV_DIR:-$REPO_ROOT/examples/hardware/ur5e/.venv}"
UR5E_PYTHON_BIN="${UR5E_VENV_DIR}/bin/python"

if [[ ! -x "$UR5E_PYTHON_BIN" ]]; then
  echo "UR5e environment not found: $UR5E_VENV_DIR" >&2
  echo "Run: bash examples/hardware/ur5e/setup_env.sh" >&2
  exit 1
fi

export VIRTUAL_ENV="${UR5E_VENV_DIR}"
export PATH="${UR5E_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

exec "$UR5E_PYTHON_BIN" -X faulthandler examples/hardware/ur5e/fake_node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --rate "${PUBLISH_RATE:-25}" \
  --dynamics-rate "${DYNAMICS_RATE:-200}" \
  --natural-frequency-hz "${NATURAL_FREQUENCY_HZ:-2}" \
  --damping-ratio "${DAMPING_RATIO:-1}" \
  "$@"
