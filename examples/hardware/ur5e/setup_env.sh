#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO_ROOT="$PWD"
UR5E_DIR="$REPO_ROOT/examples/hardware/ur5e"
UR5E_VENV_DIR="${UR5E_VENV_DIR:-$UR5E_DIR/.venv}"
UR5E_PYTHON_BIN="${UR5E_VENV_DIR}/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$UR5E_VENV_DIR" uv sync --project "$UR5E_DIR"

"$UR5E_PYTHON_BIN" - <<'PY'
import cv2
import numpy as np
import openpi_client
import pyDHgripper
import rtde_control
import rtde_receive
import zmq
from alicia_d_sdk import create_robot

print(f"opencv available: {cv2.__file__}")
print(f"numpy available: {np.__file__}")
print(f"openpi client available: {openpi_client.__file__}")
print(f"pyDHgripper available: {pyDHgripper.__file__}")
print(f"rtde_control available: {rtde_control.__file__}")
print(f"rtde_receive available: {rtde_receive.__file__}")
print(f"pyzmq available: {zmq.__file__}")
print(f"alicia_d_sdk create_robot available: {create_robot.__module__}")
PY

echo
echo "UR5e hardware environment is ready: $UR5E_VENV_DIR"
echo "Runtime: $($UR5E_PYTHON_BIN --version 2>&1)"
