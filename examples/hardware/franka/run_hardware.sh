#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FRANKA_DIR="$REPO_ROOT/examples/hardware/franka"
cd "${REPO_ROOT}"
FRANKA_VENV_DIR="${FRANKA_VENV_DIR:-$FRANKA_DIR/.venv}"
FRANKA_PYTHON_BIN="${FRANKA_VENV_DIR}/bin/python"
[[ -x "${FRANKA_PYTHON_BIN}" ]] || {
  echo "Franka environment not found: ${FRANKA_VENV_DIR}" >&2
  echo "Run: UV_PROJECT_ENVIRONMENT=\"${FRANKA_VENV_DIR}\" uv sync --project \"${FRANKA_DIR}\"" >&2
  exit 1
}
export VIRTUAL_ENV="${FRANKA_VENV_DIR}"
export PATH="${FRANKA_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"

disabled_args=()
if [[ -n "${DISABLED_ARMS:-}" ]]; then
  disabled_args=(--disabled-arm "$DISABLED_ARMS")
fi

exec "${FRANKA_PYTHON_BIN}" -X faulthandler examples/hardware/franka/node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --left-robot-ip "${LEFT_ROBOT_IP:-127.0.0.2}" \
  --right-robot-ip "${RIGHT_ROBOT_IP:-127.0.0.3}" \
  --left-gripper-ip "${LEFT_GRIPPER_IP:-}" \
  --right-gripper-ip "${RIGHT_GRIPPER_IP:-}" \
  --eva-config "${EVA_CONFIG:-configs/01_deploy/dual_franka/openpi_qpos.py}" \
  --rate "${PUBLISH_RATE:-100}" \
  --status-log-interval "${STATUS_LOG_INTERVAL:-5}" \
  --log-file "${LOG_FILE:-Log/franka_node.log}" \
  "${disabled_args[@]}" \
  "$@"
