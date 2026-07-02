#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"

disabled_args=()
if [[ -n "${DISABLED_ARMS:-}" ]]; then
  disabled_args=(--disabled-arm "$DISABLED_ARMS")
fi

python examples/hardware/franka/node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --left-robot-ip "${LEFT_ROBOT_IP:-172.16.0.2}" \
  --right-robot-ip "${RIGHT_ROBOT_IP:-172.16.0.3}" \
  --left-gripper-ip "${LEFT_GRIPPER_IP:-}" \
  --right-gripper-ip "${RIGHT_GRIPPER_IP:-}" \
  --eva-config "${EVA_CONFIG:-configs/01_deploy/dual_franka/openpi_qpos.py}" \
  --rate "${PUBLISH_RATE:-30}" \
  --status-log-interval "${STATUS_LOG_INTERVAL:-5}" \
  --log-file "${LOG_FILE:-Log/franka_node.log}" \
  "${disabled_args[@]}" \
  "$@"
