#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO_ROOT="$PWD"
R1_LITE_DIR="$REPO_ROOT/examples/hardware/r1_lite"
R1_LITE_VENV_DIR="${R1_LITE_VENV_DIR:-$R1_LITE_DIR/.venv}"
R1_LITE_PYTHON_BIN="${R1_LITE_VENV_DIR}/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
UV_PROJECT_ENVIRONMENT="$R1_LITE_VENV_DIR" uv sync --project "$R1_LITE_DIR"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
  "$R1_LITE_PYTHON_BIN" - <<'PY'
import numpy as np
import rclpy
from PIL import Image
from sensor_msgs.msg import Joy
from std_msgs.msg import String

print(f"numpy available: {np.__file__}")
print(f"pillow available: {Image.__file__}")
print(f"rclpy available: {rclpy.__file__}")
print(f"sensor_msgs Joy available: {Joy.__module__}.{Joy.__name__}")
print(f"std_msgs String available: {String.__module__}.{String.__name__}")
PY
  echo
  echo "R1 Lite ROS helper environment is ready: $R1_LITE_VENV_DIR"
else
  "$R1_LITE_PYTHON_BIN" - <<'PY'
import numpy as np
from PIL import Image

print(f"numpy available: {np.__file__}")
print(f"pillow available: {Image.__file__}")
PY
  echo
  echo "R1 Lite offline helper environment is ready: $R1_LITE_VENV_DIR"
  echo "Real ROS workflows require /opt/ros/humble/setup.bash."
fi

echo "Runtime: $($R1_LITE_PYTHON_BIN --version 2>&1)"
