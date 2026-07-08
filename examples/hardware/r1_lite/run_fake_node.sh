#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset FASTRTPS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER RMW_IMPLEMENTATION
export ROS_LOCALHOST_ONLY=1

camera_pid=""
robot_pid=""

cleanup() {
  trap - EXIT INT TERM
  for pid in "${camera_pid}" "${robot_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait "${camera_pid}" "${robot_pid}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

python examples/hardware/r1_lite/fake_node.py \
  --role cameras \
  --rate "${PUBLISH_RATE:-30}" \
  --image-height "${IMAGE_HEIGHT:-360}" \
  --image-width "${IMAGE_WIDTH:-640}" \
  --no-open-ui &
camera_pid=$!

python examples/hardware/r1_lite/fake_node.py \
  --role robot \
  --rate "${PUBLISH_RATE:-30}" \
  --ui-host "${UI_HOST:-127.0.0.1}" \
  --ui-port "${UI_PORT:-8765}" \
  --image-height "${IMAGE_HEIGHT:-360}" \
  --image-width "${IMAGE_WIDTH:-640}" \
  "$@" &
robot_pid=$!

set +e
wait -n "${camera_pid}" "${robot_pid}"
status=$?
set -e
exit "${status}"
