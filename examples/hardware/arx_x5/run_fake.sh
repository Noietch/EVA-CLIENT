#!/usr/bin/env bash
# Start the ARX X5 fake node with the directory-local Python environment.
# CLI arguments are forwarded to fake_node.py; the script reads repository sources, binds the
# configured ZMQ ports, and never imports packages from the repository root environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARX_X5_VENV_DIR="${ARX_X5_VENV_DIR:-$REPO_DIR/examples/hardware/arx_x5/.venv}"
ARX_X5_PYTHON_BIN="${ARX_X5_VENV_DIR}/bin/python"

if [[ ! -x "$ARX_X5_PYTHON_BIN" ]]; then
  echo "[arx_x5-run] ARX X5 environment not found: $ARX_X5_VENV_DIR" >&2
  echo "[arx_x5-run] Run: bash examples/hardware/arx_x5/setup_env.sh" >&2
  exit 1
fi

cd "$REPO_DIR"
export VIRTUAL_ENV="${ARX_X5_VENV_DIR}"
export PATH="${ARX_X5_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src"

exec "${ARX_X5_PYTHON_BIN}" -X faulthandler examples/hardware/arx_x5/fake_node.py \
  --obs-endpoint tcp://127.0.0.1:5555 \
  --action-endpoint tcp://127.0.0.1:5556 \
  --dynamics-mode direct \
  "$@"
