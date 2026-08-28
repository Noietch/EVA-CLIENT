#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

UR5E_VENV_DIR="${UR5E_VENV_DIR:-$PWD/examples/hardware/ur5e/.venv}"
UR5E_PYTHON_BIN="${UR5E_VENV_DIR}/bin/python"

if [[ ! -x "$UR5E_PYTHON_BIN" ]]; then
  echo "UR5e environment not found: $UR5E_VENV_DIR" >&2
  echo "Run: bash examples/hardware/ur5e/setup_env.sh" >&2
  exit 1
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export VIRTUAL_ENV="${UR5E_VENV_DIR}"
export PATH="${UR5E_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME

exec "$UR5E_PYTHON_BIN" -X faulthandler examples/hardware/ur5e/gripper.py \
  --port "${AG95_PORT:-/dev/ttyUSB1}" \
  "$@"
