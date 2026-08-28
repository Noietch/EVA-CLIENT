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

teleop_args=()
if [[ "${USE_TELEOP:-0}" == "1" ]]; then
  teleop_args=(
    --teleop-port "${TELEOP_PORT:-/dev/ttyACM0}"
    --teleop-joint-coef "${TELEOP_JOINT_COEF:-1,-1,-1,-1,-1,-1}"
  )
fi

has_camera=0
for arg in "$@"; do
  if [[ "$arg" == "--camera" || "$arg" == --camera=* ]]; then
    has_camera=1
    break
  fi
done

camera_args=()
if [[ "$has_camera" == "0" ]]; then
  camera_args=(
    --camera "${UR5E_WRIST_CAMERA:-wrist_image:2:1280:720:25:0}"
    --camera "${UR5E_EXTERIOR_CAMERA:-exterior_image:0:1280:720:25:0}"
  )
fi

exec "$UR5E_PYTHON_BIN" -X faulthandler examples/hardware/ur5e/node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --robot-ip "${UR5E_ROBOT_IP:-127.0.0.1}" \
  --gripper-port "${AG95_PORT:-/dev/ttyUSB1}" \
  --rate "${PUBLISH_RATE:-25}" \
  "${teleop_args[@]}" \
  "${camera_args[@]}" \
  "$@"
