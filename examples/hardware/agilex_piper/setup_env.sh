#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO_ROOT="$PWD"
AGILEX_PIPER_DIR="$REPO_ROOT/examples/hardware/agilex_piper"
AGILEX_PIPER_VENV_DIR="${AGILEX_PIPER_VENV_DIR:-$AGILEX_PIPER_DIR/.venv}"
AGILEX_PIPER_PYTHON_BIN="${AGILEX_PIPER_VENV_DIR}/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$AGILEX_PIPER_VENV_DIR" uv sync --project "$AGILEX_PIPER_DIR"

"$AGILEX_PIPER_PYTHON_BIN" - <<'PY'
import numpy as np
import openpi_client
import zmq

print(f"numpy available: {np.__file__}")
print(f"openpi client available: {openpi_client.__file__}")
print(f"pyzmq available: {zmq.__file__}")
PY

echo
echo "Agilex Piper hardware environment is ready: $AGILEX_PIPER_VENV_DIR"
echo "Runtime: $($AGILEX_PIPER_PYTHON_BIN --version 2>&1)"
