#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO_ROOT="$PWD"
FRANKA_DIR="$REPO_ROOT/examples/hardware/franka"
FRANKA_VENV_DIR="${FRANKA_VENV_DIR:-$FRANKA_DIR/.venv}"
FRANKA_PYTHON_BIN="${FRANKA_VENV_DIR}/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$FRANKA_VENV_DIR" uv sync --project "$FRANKA_DIR"

"$FRANKA_PYTHON_BIN" - <<'PY'
import cv2
import franky
import numpy as np
import openpi_client
import zmq

print(f"opencv available: {cv2.__file__}")
print(f"franky available: {franky.__file__}")
print(f"numpy available: {np.__file__}")
print(f"openpi client available: {openpi_client.__file__}")
print(f"pyzmq available: {zmq.__file__}")
PY

echo
echo "Franka hardware environment is ready: $FRANKA_VENV_DIR"
echo "Runtime: $($FRANKA_PYTHON_BIN --version 2>&1)"
