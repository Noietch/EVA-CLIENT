#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

VENV_DIR="${YAM_VENV_DIR:-$PWD/examples/hardware/yam/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
ENABLE_LEADERS="${ENABLE_YAM_LEADERS:-1}"
ALLOW_GRIPPER_CALIBRATION="${YAM_ALLOW_GRIPPER_CALIBRATION:-1}"

D405_CAM_HIGH_SERIAL="${D405_CAM_HIGH_SERIAL:-260422275306}"
D405_CAMERA_WIDTH="${D405_CAMERA_WIDTH:-640}"
D405_CAMERA_HEIGHT="${D405_CAMERA_HEIGHT:-480}"
D405_CAMERA_FPS="${D405_CAMERA_FPS:-30}"
D405_CAMERA_TIMEOUT_MS="${D405_CAMERA_TIMEOUT_MS:-3000}"
D405_CAMERA_PROFILE="${D405_CAMERA_PROFILE:-}"
D405_CAMERA_AUTO_EXPOSURE_LIMIT_US="${D405_CAMERA_AUTO_EXPOSURE_LIMIT_US-16000}"
D405_CAMERA_AUTO_GAIN_LIMIT="${D405_CAMERA_AUTO_GAIN_LIMIT-32}"
D405_CAMERA_WARMUP_FRAMES="${D405_CAMERA_WARMUP_FRAMES-90}"
D405_ENABLED_CAMERAS="${D405_ENABLED_CAMERAS-cam_high}"
ORBBEC_CAM_HIGH_SERIAL="${ORBBEC_CAM_HIGH_SERIAL:-CP0HC530000Z}"
ORBBEC_CAM_LEFT_WRIST_SERIAL="${ORBBEC_CAM_LEFT_WRIST_SERIAL:-CV2R1610003Z}"
ORBBEC_CAM_RIGHT_WRIST_SERIAL="${ORBBEC_CAM_RIGHT_WRIST_SERIAL:-CV2L360000CL}"
ORBBEC_CAMERA_WIDTH="${ORBBEC_CAMERA_WIDTH:-640}"
ORBBEC_CAMERA_HEIGHT="${ORBBEC_CAMERA_HEIGHT:-480}"
ORBBEC_CAMERA_FPS="${ORBBEC_CAMERA_FPS:-30}"
ORBBEC_CAMERA_FORMAT="${ORBBEC_CAMERA_FORMAT:-MJPG}"
ORBBEC_CAMERA_TIMEOUT_MS="${ORBBEC_CAMERA_TIMEOUT_MS:-1000}"
ORBBEC_CAMERA_WARMUP_FRAMES="${ORBBEC_CAMERA_WARMUP_FRAMES:-30}"
ORBBEC_CAMERA_BRIGHTNESS="${ORBBEC_CAMERA_BRIGHTNESS:-25}"
ORBBEC_POWER_LINE_FREQUENCY="${ORBBEC_POWER_LINE_FREQUENCY:-50}"
ORBBEC_ENABLED_CAMERAS="${ORBBEC_ENABLED_CAMERAS-cam_left_wrist,cam_right_wrist}"
ORBBEC_USB_RESET="${ORBBEC_USB_RESET:-1}"
YAM_CPU_AFFINITY="${YAM_CPU_AFFINITY-__auto__}"
if [[ "$YAM_CPU_AFFINITY" == "__auto__" ]]; then
  YAM_CPU_AFFINITY=""
  if [[ "$(nproc)" -ge 32 ]]; then
    YAM_CPU_AFFINITY="0-15"
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "YAM environment not found: $VENV_DIR" >&2
  echo "Run: bash examples/hardware/yam/setup_sdk.sh" >&2
  exit 1
fi

choose_can() {
  local persistent_name="$1"
  local fallback_name="$2"
  local serial_number="${3:-}"
  local path interface
  if [[ -e "/sys/class/net/$persistent_name" ]]; then
    printf '%s' "$persistent_name"
    return
  fi
  # Kernel canN numbering changes when a USB-CAN adapter is unplugged or a
  # shared USB hub resets. Prefer the adapter's stable USB serial when known,
  # then fall back to the documented workstation numbering.
  if [[ -n "$serial_number" ]]; then
    for path in /sys/class/net/can*; do
      [[ -e "$path" ]] || continue
      interface="${path##*/}"
      if udevadm info -q property -p "$path" 2>/dev/null \
        | grep -Fqx "ID_SERIAL_SHORT=$serial_number"; then
        printf '%s' "$interface"
        return
      fi
    done
  fi
  printf '%s' "$fallback_name"
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

# Stable serials for the currently verified workstation. Override these four
# variables when the adapters are replaced or moved to another host.
YAM_CAN_SERIAL_LEFT_FOLLOWER="${YAM_CAN_SERIAL_LEFT_FOLLOWER:-2086337D594E5018}"
YAM_CAN_SERIAL_RIGHT_FOLLOWER="${YAM_CAN_SERIAL_RIGHT_FOLLOWER:-20813381594E5018}"
YAM_CAN_SERIAL_LEFT_LEADER="${YAM_CAN_SERIAL_LEFT_LEADER:-325F384633354B04}"
YAM_CAN_SERIAL_RIGHT_LEADER="${YAM_CAN_SERIAL_RIGHT_LEADER:-207C3381594E5018}"

LEFT_FOLLOWER_CAN="${LEFT_FOLLOWER_CAN:-$(choose_can can_follower_l can1 "$YAM_CAN_SERIAL_LEFT_FOLLOWER")}"
RIGHT_FOLLOWER_CAN="${RIGHT_FOLLOWER_CAN:-$(choose_can can_follower_r can2 "$YAM_CAN_SERIAL_RIGHT_FOLLOWER")}"
LEFT_LEADER_CAN="${LEFT_LEADER_CAN:-$(choose_can can_leader_l can0 "$YAM_CAN_SERIAL_LEFT_LEADER")}"
RIGHT_LEADER_CAN="${RIGHT_LEADER_CAN:-$(choose_can can_leader_r can3 "$YAM_CAN_SERIAL_RIGHT_LEADER")}"
LEFT_LEADER_GRIPPER_ENDPOINTS="${LEFT_LEADER_GRIPPER_ENDPOINTS:--0.007669904,-0.708699124}"
RIGHT_LEADER_GRIPPER_ENDPOINTS="${RIGHT_LEADER_GRIPPER_ENDPOINTS:-0.030679616,-0.648873873}"

echo "YAM CAN mapping: follower(left=$LEFT_FOLLOWER_CAN right=$RIGHT_FOLLOWER_CAN) " \
  "leader(left=$LEFT_LEADER_CAN right=$RIGHT_LEADER_CAN)" >&2

if [[ "$ENABLE_LEADERS" != "0" && "$ENABLE_LEADERS" != "1" ]]; then
  echo "ENABLE_YAM_LEADERS must be 0 or 1." >&2
  exit 2
fi
if [[ "$ALLOW_GRIPPER_CALIBRATION" != "0" && "$ALLOW_GRIPPER_CALIBRATION" != "1" ]]; then
  echo "YAM_ALLOW_GRIPPER_CALIBRATION must be 0 or 1." >&2
  exit 2
fi
if [[ "$ORBBEC_USB_RESET" != "0" && "$ORBBEC_USB_RESET" != "1" ]]; then
  echo "ORBBEC_USB_RESET must be 0 or 1." >&2
  exit 2
fi
show_help=0
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" || "$arg" == "--list-cameras" ]]; then
    show_help=1
  fi
done

# Keep a second launcher from resetting cameras that belong to a live node.
if [[ "$show_help" == "0" ]]; then
  YAM_LOCK_FILE="${YAM_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/eva-yam-hardware-${UID}.lock}"
  exec 9>"$YAM_LOCK_FILE"
  if ! flock -n 9; then
    echo "YAM hardware is already running. Stop it with Ctrl-C before restarting." >&2
    exit 1
  fi
fi

if [[ "$show_help" == "0" && "$ORBBEC_ENABLED_CAMERAS" == *,* \
      && -r /sys/module/usbcore/parameters/usbfs_memory_mb ]]; then
  usbfs_memory_mb="$(</sys/module/usbcore/parameters/usbfs_memory_mb)"
  if [[ "$usbfs_memory_mb" =~ ^[0-9]+$ && "$usbfs_memory_mb" -lt 256 ]]; then
    echo "Orbbec multi-camera capture requires usbfs_memory_mb >= 256 (current: $usbfs_memory_mb)." >&2
    echo "Run: echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb" >&2
    exit 1
  fi
fi

reset_orbbec_camera() {
  local serial_number="$1"
  local device properties bus_number device_number
  command -v usbreset >/dev/null 2>&1 || return 0
  for device in /dev/bus/usb/*/*; do
    [[ -e "$device" ]] || continue
    properties="$(udevadm info -q property -n "$device" 2>/dev/null || true)"
    if grep -Fqx "ID_SERIAL_SHORT=$serial_number" <<< "$properties"; then
      bus_number="$(basename "$(dirname "$device")")"
      device_number="$(basename "$device")"
      usbreset "$bus_number/$device_number"
      return
    fi
  done
}

if [[ "$show_help" == "0" && "$ORBBEC_USB_RESET" == "1" ]]; then
  case ",$ORBBEC_ENABLED_CAMERAS," in
    *,cam_left_wrist,*) reset_orbbec_camera "$ORBBEC_CAM_LEFT_WRIST_SERIAL" ;;
  esac
  case ",$ORBBEC_ENABLED_CAMERAS," in
    *,cam_right_wrist,*) reset_orbbec_camera "$ORBBEC_CAM_RIGHT_WRIST_SERIAL" ;;
  esac
  case ",$ORBBEC_ENABLED_CAMERAS," in
    *,cam_high,*) reset_orbbec_camera "$ORBBEC_CAM_HIGH_SERIAL" ;;
  esac
  udevadm settle --timeout=10
  sleep 2
fi

node_args=(
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}"
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}"
  --arm-type "${YAM_ARM_TYPE:-yam}"
  --gripper-type "${YAM_GRIPPER_TYPE:-linear_4310}"
  --rate "${PUBLISH_RATE:-30}"
  --control-rate "${CONTROL_RATE:-200}"
  --command-timeout "${COMMAND_TIMEOUT:-0.5}"
  --idle-mode "${YAM_IDLE_MODE:-gravity_comp}"
  --startup-position "${YAM_STARTUP_POSITION:-zero}"
  --startup-duration "${YAM_STARTUP_DURATION:-5.0}"
  --gripper-max-speed "${YAM_GRIPPER_MAX_SPEED:-2.0}"
  --tracking-ki "${YAM_TRACKING_KI:-0.0}"
  --tracking-trim-limit "${YAM_TRACKING_TRIM_LIMIT:-0.12}"
  --tracking-deadband "${YAM_TRACKING_DEADBAND:-0.002}"
  --tracking-settle-delay "${YAM_TRACKING_SETTLE_DELAY:-0.15}"
  --startup-trim-duration "${YAM_STARTUP_TRIM_DURATION:-3.0}"
)

if [[ -n "${YAM_JOINT4_KP:-}" ]]; then
  node_args+=(--joint4-kp "$YAM_JOINT4_KP")
fi

if [[ -n "${YAM_EE_MASS:-}" ]]; then
  node_args+=(--end-effector-mass "$YAM_EE_MASS")
fi

if [[ -n "${YAM_GRAVITY_COMP_FACTOR:-}" ]]; then
  gravity_values=( ${YAM_GRAVITY_COMP_FACTOR//,/ } )
  node_args+=(--gravity-comp-factor "${gravity_values[@]}")
fi

if [[ -n "${YAM_GRIPPER_LIMITS:-}" ]]; then
  gripper_limit_values=( ${YAM_GRIPPER_LIMITS//,/ } )
  node_args+=(--gripper-limits-override "${gripper_limit_values[@]}")
fi

if [[ "$ALLOW_GRIPPER_CALIBRATION" == "1" ]]; then
  node_args+=(--allow-gripper-calibration)
fi

if [[ "$show_help" == "0" ]]; then
  bring_up_can "$LEFT_FOLLOWER_CAN"
  bring_up_can "$RIGHT_FOLLOWER_CAN"
fi

node_args+=(
  --follower-can "left_arm=$LEFT_FOLLOWER_CAN"
  --follower-can "right_arm=$RIGHT_FOLLOWER_CAN"
)
if [[ "$ENABLE_LEADERS" == "1" ]]; then
  if [[ "$show_help" == "0" ]]; then
    bring_up_can "$LEFT_LEADER_CAN"
    bring_up_can "$RIGHT_LEADER_CAN"
  fi
  node_args+=(
    --leader-cans "$LEFT_LEADER_CAN" "$RIGHT_LEADER_CAN"
    --leader-gripper-endpoint "left_arm=$LEFT_LEADER_GRIPPER_ENDPOINTS"
    --leader-gripper-endpoint "right_arm=$RIGHT_LEADER_GRIPPER_ENDPOINTS"
    --direct-leader-control
  )
fi

if [[ -n "$D405_ENABLED_CAMERAS" ]]; then
  IFS=',' read -ra enabled_cameras <<< "$D405_ENABLED_CAMERAS"
  for camera_key in "${enabled_cameras[@]}"; do
    case "$camera_key" in
      cam_high) node_args+=(--camera "cam_high=$D405_CAM_HIGH_SERIAL") ;;
      *)
        echo "Unknown D405 camera key: $camera_key" >&2
        exit 2
        ;;
    esac
  done
fi

node_args+=(
  --camera-width "$D405_CAMERA_WIDTH"
  --camera-height "$D405_CAMERA_HEIGHT"
  --camera-fps "$D405_CAMERA_FPS"
  --camera-timeout-ms "$D405_CAMERA_TIMEOUT_MS"
)

if [[ -n "$D405_CAMERA_PROFILE" ]]; then
  node_args+=(--camera-profile "$D405_CAMERA_PROFILE")
fi

if [[ -n "${D405_CAMERA_AUTO_EXPOSURE_LIMIT_US:-}" ]]; then
  node_args+=(--camera-auto-exposure-limit-us "$D405_CAMERA_AUTO_EXPOSURE_LIMIT_US")
fi
if [[ -n "${D405_CAMERA_AUTO_GAIN_LIMIT:-}" ]]; then
  node_args+=(--camera-auto-gain-limit "$D405_CAMERA_AUTO_GAIN_LIMIT")
fi
if [[ -n "${D405_CAMERA_EXPOSURE_US:-}" ]]; then
  node_args+=(--camera-exposure-us "$D405_CAMERA_EXPOSURE_US")
fi
if [[ -n "${D405_CAMERA_WARMUP_FRAMES:-}" ]]; then
  node_args+=(--camera-warmup-frames "$D405_CAMERA_WARMUP_FRAMES")
fi

if [[ -n "$ORBBEC_ENABLED_CAMERAS" ]]; then
  IFS=',' read -ra enabled_orbbec_cameras <<< "$ORBBEC_ENABLED_CAMERAS"
  for camera_key in "${enabled_orbbec_cameras[@]}"; do
    case "$camera_key" in
      cam_high)
        node_args+=(--orbbec-camera "cam_high=$ORBBEC_CAM_HIGH_SERIAL")
        ;;
      cam_left_wrist)
        node_args+=(--orbbec-camera "cam_left_wrist=$ORBBEC_CAM_LEFT_WRIST_SERIAL")
        ;;
      cam_right_wrist)
        node_args+=(--orbbec-camera "cam_right_wrist=$ORBBEC_CAM_RIGHT_WRIST_SERIAL")
        ;;
      *)
        echo "Unknown Orbbec camera key: $camera_key" >&2
        exit 2
        ;;
    esac
  done
fi

node_args+=(
  --orbbec-width "$ORBBEC_CAMERA_WIDTH"
  --orbbec-height "$ORBBEC_CAMERA_HEIGHT"
  --orbbec-fps "$ORBBEC_CAMERA_FPS"
  --orbbec-color-format "$ORBBEC_CAMERA_FORMAT"
  --orbbec-timeout-ms "$ORBBEC_CAMERA_TIMEOUT_MS"
  --orbbec-warmup-frames "$ORBBEC_CAMERA_WARMUP_FRAMES"
  --orbbec-brightness "$ORBBEC_CAMERA_BRIGHTNESS"
  --orbbec-power-line-frequency "$ORBBEC_POWER_LINE_FREQUENCY"
)

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
python_command=("$PYTHON_BIN")
if [[ -n "$YAM_CPU_AFFINITY" ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "YAM_CPU_AFFINITY requires taskset." >&2
    exit 1
  fi
  echo "YAM CPU affinity: $YAM_CPU_AFFINITY" >&2
  python_command=(taskset -c "$YAM_CPU_AFFINITY" "$PYTHON_BIN")
fi
exec "${python_command[@]}" examples/hardware/yam/node.py "${node_args[@]}" "$@"
