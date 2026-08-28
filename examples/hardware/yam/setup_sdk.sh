#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPOSITORY_ROOT="$PWD"
SDK_RELATIVE_DIR="examples/hardware/yam/SDK/i2rt"
SDK_DIR="$REPOSITORY_ROOT/$SDK_RELATIVE_DIR"
PROJECT_DIR="$PWD/examples/hardware/yam"
VENV_DIR="${YAM_VENV_DIR:-$PROJECT_DIR/.venv}"
YAM_REVISION="5dab60a49d3b59e2ae9c46b117765ef20e2f1699"
YAM_RELEASE="Noietch/i2rt main"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

for command in git c++; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing build command: $command" >&2
    echo "Install Git and a C++ compiler before setting up the YAM environment." >&2
    exit 1
  fi
done

if ! git -C "$SDK_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  git submodule update --init --recursive -- "$SDK_RELATIVE_DIR"
fi

SDK_HEAD="$(git -C "$SDK_DIR" rev-parse HEAD)"
if [[ "$SDK_HEAD" != "$YAM_REVISION" ]]; then
  echo "YAM SDK is at $SDK_HEAD, expected tested revision $YAM_RELEASE ($YAM_REVISION)." >&2
  echo "Run: git submodule update --init --recursive -- $SDK_RELATIVE_DIR" >&2
  exit 1
fi
echo "YAM SDK pinned: $YAM_RELEASE ($YAM_REVISION)"

env -u PYTHONHOME -u PYTHONPATH -u VIRTUAL_ENV \
  PYTHONNOUSERSITE=1 \
  UV_CONCURRENT_INSTALLS=1 \
  UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
  uv sync --no-cache --project "$PROJECT_DIR"

env -u PYTHONHOME \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT" \
  "$VENV_DIR/bin/python" - <<'PY'
import cv2
import i2rt
import pyorbbecsdk
import zmq
from importlib.util import find_spec

from core.registry import ROBOT_REGISTRY
from examples.hardware.yam.node import build_arg_parser

robot = ROBOT_REGISTRY.build("dual_yam")
assert robot.total_action_dim == 14
assert build_arg_parser().prog
realsense_spec = find_spec("pyrealsense2")
assert realsense_spec is not None

print(f"i2rt installed: {i2rt.__file__}")
print(f"opencv available: {cv2.__file__}")
print(f"pyorbbecsdk available: {pyorbbecsdk.__file__}")
print(f"optional pyrealsense2 package available: {realsense_spec.origin}")
print(f"pyzmq available: {zmq.__file__}")
print("EVA YAM node import available")
PY

echo
echo "YAM SDK environment is ready: $VENV_DIR"
echo "Detected CAN interfaces:"
find /sys/class/net -maxdepth 1 -type l -name 'can*' -printf '  %f\n' 2>/dev/null || true
echo
echo "Optional boot-time CAN setup (system-wide):"
echo "  sudo sh $SDK_DIR/devices/install_devices.sh"
