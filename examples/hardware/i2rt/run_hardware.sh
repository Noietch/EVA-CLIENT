#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

VENV_DIR="${I2RT_VENV_DIR:-$PWD/examples/hardware/i2rt/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
ROBOT="${I2RT_ROBOT:-i2rt_dual_yam}"
SIM="${I2RT_SIM:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "I2RT environment not found: $VENV_DIR" >&2
  echo "Run: bash examples/hardware/i2rt/setup_sdk.sh" >&2
  exit 1
fi

choose_can() {
  local persistent_name="$1"
  local fallback_name="$2"
  if [[ -e "/sys/class/net/$persistent_name" ]]; then
    printf '%s' "$persistent_name"
  else
    printf '%s' "$fallback_name"
  fi
}

bring_up_can() {
  local interface="$1"
  if ! ip link show "$interface" >/dev/null 2>&1; then
    echo "CAN interface not found: $interface" >&2
    return 1
  fi
  if ip -details link show "$interface" | grep -q 'bitrate 1000000'; then
    if ip link show "$interface" | grep -q 'UP'; then
      return
    fi
  else
    sudo ip link set "$interface" down 2>/dev/null || true
  fi
  sudo ip link set "$interface" up type can bitrate 1000000
}

LEFT_FOLLOWER_CAN="${LEFT_FOLLOWER_CAN:-$(choose_can can_follower_l can0)}"
RIGHT_FOLLOWER_CAN="${RIGHT_FOLLOWER_CAN:-$(choose_can can_follower_r can1)}"
SINGLE_FOLLOWER_CAN="${FOLLOWER_CAN:-$LEFT_FOLLOWER_CAN}"
LEFT_LEADER_CAN="${LEFT_LEADER_CAN:-$(choose_can can_leader_l can2)}"
RIGHT_LEADER_CAN="${RIGHT_LEADER_CAN:-$(choose_can can_leader_r can3)}"

show_help=0
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" || "$arg" == "--list-cameras" ]]; then
    show_help=1
  fi
done

node_args=(
  --robot "$ROBOT"
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}"
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}"
  --arm-type "${I2RT_ARM_TYPE:-yam}"
  --gripper-type "${I2RT_GRIPPER_TYPE:-linear_4310}"
  --rate "${PUBLISH_RATE:-30}"
  --control-rate "${CONTROL_RATE:-200}"
  --command-timeout "${COMMAND_TIMEOUT:-0.5}"
  --idle-mode "${I2RT_IDLE_MODE:-gravity_comp}"
  --startup-position "${I2RT_STARTUP_POSITION:-current}"
  --startup-duration "${I2RT_STARTUP_DURATION:-5.0}"
  --tracking-ki "${I2RT_TRACKING_KI:-0.0}"
  --tracking-trim-limit "${I2RT_TRACKING_TRIM_LIMIT:-0.12}"
  --tracking-deadband "${I2RT_TRACKING_DEADBAND:-0.002}"
  --tracking-settle-delay "${I2RT_TRACKING_SETTLE_DELAY:-0.15}"
  --startup-trim-duration "${I2RT_STARTUP_TRIM_DURATION:-3.0}"
)

if [[ -n "${I2RT_JOINT4_KP:-}" ]]; then
  node_args+=(--joint4-kp "$I2RT_JOINT4_KP")
fi

if [[ -n "${I2RT_EE_MASS:-}" ]]; then
  node_args+=(--end-effector-mass "$I2RT_EE_MASS")
fi

if [[ -n "${I2RT_GRAVITY_COMP_FACTOR:-}" ]]; then
  gravity_values=( ${I2RT_GRAVITY_COMP_FACTOR//,/ } )
  node_args+=(--gravity-comp-factor "${gravity_values[@]}")
fi

if [[ -n "${I2RT_GRIPPER_LIMITS:-}" ]]; then
  gripper_limit_values=( ${I2RT_GRIPPER_LIMITS//,/ } )
  node_args+=(--gripper-limits-override "${gripper_limit_values[@]}")
fi

if [[ "${I2RT_ALLOW_GRIPPER_CALIBRATION:-0}" == "1" ]]; then
  node_args+=(--allow-gripper-calibration)
fi

if [[ "$SIM" == "1" ]]; then
  node_args+=(--sim)
elif [[ "$show_help" == "0" ]]; then
  if [[ "$ROBOT" == "i2rt_yam" ]]; then
    bring_up_can "$SINGLE_FOLLOWER_CAN"
  else
    bring_up_can "$LEFT_FOLLOWER_CAN"
    bring_up_can "$RIGHT_FOLLOWER_CAN"
  fi
fi

if [[ "$ROBOT" == "i2rt_yam" ]]; then
  node_args+=(--follower-can "arm=$SINGLE_FOLLOWER_CAN")
  if [[ -n "${LEADER_CAN:-}" ]]; then
    if [[ "$SIM" != "1" && "$show_help" == "0" ]]; then
      bring_up_can "$LEADER_CAN"
    fi
    node_args+=(--leader-can "arm=$LEADER_CAN")
  fi
else
  node_args+=(
    --follower-can "left_arm=$LEFT_FOLLOWER_CAN"
    --follower-can "right_arm=$RIGHT_FOLLOWER_CAN"
  )
  enable_left_leader="${ENABLE_I2RT_LEFT_LEADER:-${ENABLE_I2RT_LEADERS:-0}}"
  enable_right_leader="${ENABLE_I2RT_RIGHT_LEADER:-${ENABLE_I2RT_LEADERS:-0}}"
  if [[ "$enable_left_leader" == "1" ]]; then
    if [[ "$SIM" != "1" && "$show_help" == "0" ]]; then
      bring_up_can "$LEFT_LEADER_CAN"
    fi
    node_args+=(--leader-can "left_arm=$LEFT_LEADER_CAN")
  fi
  if [[ "$enable_right_leader" == "1" ]]; then
    if [[ "$SIM" != "1" && "$show_help" == "0" ]]; then
      bring_up_can "$RIGHT_LEADER_CAN"
    fi
    node_args+=(--leader-can "right_arm=$RIGHT_LEADER_CAN")
  fi
fi

if [[ -n "${D405_CAM_HIGH_SERIAL:-}" ]]; then
  node_args+=(--camera "cam_high=$D405_CAM_HIGH_SERIAL")
fi
if [[ "$ROBOT" == "i2rt_yam" ]]; then
  if [[ -n "${D405_CAM_WRIST_SERIAL:-}" ]]; then
    node_args+=(--camera "cam_wrist=$D405_CAM_WRIST_SERIAL")
  fi
else
  if [[ -n "${D405_CAM_LEFT_WRIST_SERIAL:-}" ]]; then
    node_args+=(--camera "cam_left_wrist=$D405_CAM_LEFT_WRIST_SERIAL")
  fi
  if [[ -n "${D405_CAM_RIGHT_WRIST_SERIAL:-}" ]]; then
    node_args+=(--camera "cam_right_wrist=$D405_CAM_RIGHT_WRIST_SERIAL")
  fi
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec "$PYTHON_BIN" examples/hardware/i2rt/node.py "${node_args[@]}" "$@"
