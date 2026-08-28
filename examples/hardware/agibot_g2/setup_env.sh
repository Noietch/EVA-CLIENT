#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO_ROOT="$PWD"
AGIBOT_G2_DIR="$REPO_ROOT/examples/hardware/agibot_g2"
AGIBOT_G2_VENV_DIR="${AGIBOT_G2_VENV_DIR:-$AGIBOT_G2_DIR/.venv}"
AGIBOT_G2_PYTHON_BIN="${AGIBOT_G2_VENV_DIR}/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$AGIBOT_G2_VENV_DIR" uv sync --project "$AGIBOT_G2_DIR"

"$AGIBOT_G2_PYTHON_BIN" - <<'PY'
import cv2
import numpy as np
import openpi_client
import zmq

print(f"opencv available: {cv2.__file__}")
print(f"numpy available: {np.__file__}")
print(f"openpi client available: {openpi_client.__file__}")
print(f"pyzmq available: {zmq.__file__}")
PY

echo
echo "Agibot G2 hardware environment is ready: $AGIBOT_G2_VENV_DIR"
echo "Runtime: $($AGIBOT_G2_PYTHON_BIN --version 2>&1)"
