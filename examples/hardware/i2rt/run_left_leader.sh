#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Defaults verified on the current EVA workstation. Every value can be
# overridden in the environment when adapters or cameras are reassigned.
export I2RT_ROBOT="${I2RT_ROBOT:-i2rt_dual_yam}"
export I2RT_GRIPPER_TYPE="${I2RT_GRIPPER_TYPE:-linear_4310}"
export LEFT_FOLLOWER_CAN="${LEFT_FOLLOWER_CAN:-can0}"
export RIGHT_FOLLOWER_CAN="${RIGHT_FOLLOWER_CAN:-can1}"
export LEFT_LEADER_CAN="${LEFT_LEADER_CAN:-can2}"
export ENABLE_I2RT_LEFT_LEADER="${ENABLE_I2RT_LEFT_LEADER:-1}"
export ENABLE_I2RT_RIGHT_LEADER="${ENABLE_I2RT_RIGHT_LEADER:-0}"
export D405_CAM_HIGH_SERIAL="${D405_CAM_HIGH_SERIAL:-260422275306}"
export COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-0.5}"
export PUBLISH_RATE="${PUBLISH_RATE:-30}"
export CONTROL_RATE="${CONTROL_RATE:-200}"
export I2RT_IDLE_MODE="${I2RT_IDLE_MODE:-hold_position}"
export I2RT_STARTUP_POSITION="${I2RT_STARTUP_POSITION:-zero}"
export I2RT_STARTUP_DURATION="${I2RT_STARTUP_DURATION:-5.0}"
export I2RT_JOINT4_KP="${I2RT_JOINT4_KP:-40.0}"
export I2RT_TRACKING_KI="${I2RT_TRACKING_KI:-2.0}"
export I2RT_TRACKING_TRIM_LIMIT="${I2RT_TRACKING_TRIM_LIMIT:-0.12}"
export I2RT_TRACKING_DEADBAND="${I2RT_TRACKING_DEADBAND:-0.002}"
export I2RT_TRACKING_SETTLE_DELAY="${I2RT_TRACKING_SETTLE_DELAY:-0.15}"
export I2RT_STARTUP_TRIM_DURATION="${I2RT_STARTUP_TRIM_DURATION:-3.0}"

echo "Starting I2RT partial-leader hardware:"
echo "  left follower:  $LEFT_FOLLOWER_CAN"
echo "  right follower: $RIGHT_FOLLOWER_CAN (held when HIL/collection starts)"
echo "  left leader:    $LEFT_LEADER_CAN"
echo "  D405 cam_high:  $D405_CAM_HIGH_SERIAL"
echo "  idle mode:      $I2RT_IDLE_MODE"
echo "  control rate:   ${CONTROL_RATE} Hz"
echo "  startup target: all-zero arm joints (${I2RT_STARTUP_DURATION}s)"
echo "  joint4 kp:      $I2RT_JOINT4_KP"
if [[ -n "${I2RT_GRAVITY_COMP_FACTOR:-}" ]]; then
  echo "  gravity factor: $I2RT_GRAVITY_COMP_FACTOR (override)"
else
  echo "  gravity factor: official SDK defaults"
fi
if [[ -n "${I2RT_GRIPPER_LIMITS:-}" ]]; then
  echo "  gripper limits: $I2RT_GRIPPER_LIMITS (override)"
elif [[ "${I2RT_ALLOW_GRIPPER_CALIBRATION:-0}" == "1" ]]; then
  echo "  gripper limits: automatic calibration explicitly enabled"
else
  echo "  gripper limits: unset; motorized-gripper startup is safety locked"
fi
if [[ -n "${I2RT_EE_MASS:-}" ]]; then
  echo "  end-eff mass:   ${I2RT_EE_MASS} kg (override)"
else
  echo "  end-eff model:  official $I2RT_GRIPPER_TYPE"
fi
echo "  tracking trim:  ki=$I2RT_TRACKING_KI limit=$I2RT_TRACKING_TRIM_LIMIT rad"
exec bash "$SCRIPT_DIR/run_hardware.sh" \
  --camera-width "${D405_WIDTH:-640}" \
  --camera-height "${D405_HEIGHT:-480}" \
  --camera-fps "${D405_FPS:-30}" \
  --camera-timeout-ms "${D405_TIMEOUT_MS:-3000}" \
  "$@"
