#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"

# The ARX R5 SDK ships a prebuilt pybind module (arx_r5_python) plus shared
# libraries. Put its package root on PYTHONPATH and its .so dirs on the loader
# path so `import bimanual` resolves.
ARX_SDK_DIR="${ARX_SDK_DIR:-$PWD/examples/hardware/arx/SDK/ARX_R5_python}"
ARX_LIBRARY_PATH="$ARX_SDK_DIR/bimanual/lib:$ARX_SDK_DIR/bimanual/lib/arx_r5_src"
export PYTHONPATH="$ARX_SDK_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$ARX_LIBRARY_PATH:/usr/local/lib:${LD_LIBRARY_PATH:-}"

LEFT_CAN_PORT_VALUE="${LEFT_CAN_PORT:-can0}"
RIGHT_CAN_PORT_VALUE="${RIGHT_CAN_PORT:-can1}"
LEFT_CAN_DEVICE="${LEFT_CAN_DEVICE:-/dev/arxcan0}"
RIGHT_CAN_DEVICE="${RIGHT_CAN_DEVICE:-/dev/arxcan1}"
ARX_CAN_SLCAND_SPEED="${ARX_CAN_SLCAND_SPEED:-s8}"
ARX_CAN_WAIT_SECONDS="${ARX_CAN_WAIT_SECONDS:-5}"
INIT_ARX_CAN="${INIT_ARX_CAN:-1}"
ENABLE_ARX_ARMS="${ENABLE_ARX_ARMS:-0}"
DISABLED_ARMS_VALUE="${DISABLED_ARMS:-}"
ARX_FILTER_SDK_BANNER="${ARX_FILTER_SDK_BANNER:-1}"
ORBBEC_WHITE_BALANCE_FILE="${ORBBEC_WHITE_BALANCE_FILE-examples/hardware/arx/utils/white_balance.yaml}"
SHOW_NODE_HELP=0

log() {
  printf '[arx-run] %s\n' "$*"
}

warn() {
  printf '[arx-run] WARN: %s\n' "$*" >&2
}

die() {
  printf '[arx-run] ERROR: %s\n' "$*" >&2
  exit 1
}

run_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi
  command -v sudo >/dev/null 2>&1 || die "sudo is required for CAN setup"
  sudo -E "$@"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

run_python() {
  if [[ "${ARX_FILTER_SDK_BANNER}" == "0" ]]; then
    python "$@"
    return
  fi

  set +e
  python "$@" \
    > >(sed '/^[[:space:]]*ARX方舟无限[[:space:]]*$/d') \
    2> >(sed '/^[[:space:]]*ARX方舟无限[[:space:]]*$/d' >&2)
  local rc=$?
  set -e
  return "${rc}"
}

interface_exists() {
  ip link show "$1" >/dev/null 2>&1
}

interface_is_up() {
  ip link show "$1" 2>/dev/null | grep -q "<[^>]*UP"
}

wait_for_device() {
  local device="$1"
  local deadline=$((SECONDS + ARX_CAN_WAIT_SECONDS))

  while [[ ! -e "${device}" && "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done

  [[ -e "${device}" ]]
}

stop_matching_slcand() {
  local device="$1"
  local iface="$2"
  local resolved_device=""
  local pid cmdline

  resolved_device="$(readlink -f "${device}" 2>/dev/null || true)"
  for pid in $(pgrep -x slcand 2>/dev/null || true); do
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    if [[ " ${cmdline} " == *" ${device} "* ]] ||
       [[ -n "${resolved_device}" && " ${cmdline} " == *" ${resolved_device} "* ]] ||
       [[ " ${cmdline} " == *" ${iface} "* ]]; then
      log "stopping old slcand pid ${pid} for ${device} ${iface}"
      run_root kill "${pid}" 2>/dev/null || true
    fi
  done
}

start_can_mapping() {
  local mapping="$1"
  local device iface

  [[ "${mapping}" == *:* ]] || die "invalid CAN mapping '${mapping}', expected DEVICE:INTERFACE"
  device="${mapping%%:*}"
  iface="${mapping#*:}"
  [[ -n "${device}" && -n "${iface}" ]] || die "invalid CAN mapping '${mapping}'"

  if ! wait_for_device "${device}"; then
    warn "${device} not found, skip ${iface}"
    return 2
  fi

  if interface_exists "${iface}"; then
    if interface_is_up "${iface}"; then
      log "${iface} already UP (${device})"
      return 0
    fi
    log "${iface} exists but is DOWN, bringing it UP"
    if run_root ip link set "${iface}" up; then
      log "${iface} is UP"
      return 0
    fi
    warn "failed to bring existing ${iface} UP, recreating it"
    run_root ip link set "${iface}" down >/dev/null 2>&1 || true
    stop_matching_slcand "${device}" "${iface}"
    sleep 1
  fi

  log "starting ${device} as ${iface} with slcand -${ARX_CAN_SLCAND_SPEED}"
  stop_matching_slcand "${device}" "${iface}"
  run_root slcand -o -f "-${ARX_CAN_SLCAND_SPEED}" "${device}" "${iface}"
  sleep 0.5
  interface_exists "${iface}" || die "slcand did not create ${iface}"
  run_root ip link set "${iface}" up
  log "${iface} is UP"
}

arm_disabled() {
  local group="$1"
  local item

  for item in ${DISABLED_ARMS_VALUE//,/ }; do
    case "${item}" in
      "${group}"|"${group%_arm}") return 0 ;;
    esac
  done
  return 1
}

init_arx_can() {
  local mappings=()
  local mapping rc ok_count=0 skipped_count=0

  [[ "${INIT_ARX_CAN}" != "0" ]] || {
    log "CAN setup skipped (INIT_ARX_CAN=0)"
    return
  }

  require_command ip
  require_command slcand
  require_command pgrep
  if command -v modprobe >/dev/null 2>&1; then
    run_root modprobe slcan >/dev/null 2>&1 || true
  fi

  if ! arm_disabled left_arm; then
    mappings+=("${LEFT_CAN_DEVICE}:${LEFT_CAN_PORT_VALUE}")
  fi
  if ! arm_disabled right_arm; then
    mappings+=("${RIGHT_CAN_DEVICE}:${RIGHT_CAN_PORT_VALUE}")
  fi

  if [[ "${#mappings[@]}" -eq 0 ]]; then
    log "CAN setup skipped: all ARX arms are disabled"
    return
  fi

  for mapping in "${mappings[@]}"; do
    set +e
    start_can_mapping "${mapping}"
    rc=$?
    set -e
    case "${rc}" in
      0) ok_count=$((ok_count + 1)) ;;
      2) skipped_count=$((skipped_count + 1)) ;;
      *) die "failed to initialize ${mapping}" ;;
    esac
  done

  [[ "${ok_count}" -gt 0 ]] || die "no CAN interface initialized (${skipped_count} skipped)"
  log "CAN setup done: ${ok_count} initialized, ${skipped_count} skipped"
}

enable_arx_arms() {
  [[ "${ENABLE_ARX_ARMS}" != "0" ]] || {
    log "ARX arm enable skipped (ENABLE_ARX_ARMS=0)"
    return
  }

  ARX_ENABLE_SPECS="left_arm:${LEFT_CAN_PORT_VALUE};right_arm:${RIGHT_CAN_PORT_VALUE}" \
  ARX_DISABLED_ARMS="${DISABLED_ARMS_VALUE}" \
  ARX_ARM_TYPE="${ARM_TYPE:-0}" \
  run_python - <<'PY'
import os

import numpy as np

from bimanual import SingleArm


def disabled_groups(raw: str) -> set[str]:
    groups = set()
    for item in raw.replace(",", " ").split():
        if item == "left":
            groups.add("left_arm")
        elif item == "right":
            groups.add("right_arm")
        else:
            groups.add(item)
    return groups


disabled = disabled_groups(os.environ.get("ARX_DISABLED_ARMS", ""))
arm_type = int(os.environ["ARX_ARM_TYPE"])
enabled = 0
for spec in os.environ["ARX_ENABLE_SPECS"].split(";"):
    group_name, can_port = spec.split(":", 1)
    if group_name in disabled:
        print(f"[arx-run] {group_name} enable skipped: disabled")
        continue
    arm = SingleArm({"can_port": can_port, "type": arm_type, "num_joints": 7})
    positions = np.asarray(arm.get_joint_positions(), dtype=float)
    if positions.shape[0] < 6:
        raise RuntimeError(f"{group_name} returned {positions.shape[0]} joints")
    arm.set_joint_positions(positions[:6])
    enabled += 1
    print(f"[arx-run] {group_name} enabled on {can_port}")

if enabled == 0:
    print("[arx-run] no ARX arms enabled")
PY
}

disabled_args=()
if [[ -n "${DISABLED_ARMS:-}" ]]; then
  disabled_args=(--disabled-arm "$DISABLED_ARMS")
fi

collection_args=()
if [[ "${PASSIVE_COLLECTION:-0}" == "1" ]]; then
  collection_args=(--passive-collection)
fi

orbbec_white_balance_args=()
user_set_orbbec_white_balance=0

for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      INIT_ARX_CAN=0
      ENABLE_ARX_ARMS=0
      SHOW_NODE_HELP=1
      ;;
    --orbbec-white-balance-file|--orbbec-white-balance-file=*)
      user_set_orbbec_white_balance=1
      ;;
  esac
done

if [[ "${user_set_orbbec_white_balance}" == "0" &&
      -n "${ORBBEC_WHITE_BALANCE_FILE}" &&
      -f "${ORBBEC_WHITE_BALANCE_FILE}" ]]; then
  orbbec_white_balance_args=(--orbbec-white-balance-file "${ORBBEC_WHITE_BALANCE_FILE}")
fi

append_disabled_arm() {
  if [[ -n "${DISABLED_ARMS_VALUE}" ]]; then
    DISABLED_ARMS_VALUE="${DISABLED_ARMS_VALUE},$1"
  else
    DISABLED_ARMS_VALUE="$1"
  fi
}

args=("$@")
arg_index=0
if [[ "${SHOW_NODE_HELP}" == "0" ]]; then
  while [[ "${arg_index}" -lt "${#args[@]}" ]]; do
    arg="${args[$arg_index]}"
    case "${arg}" in
      --disabled-arm)
        next_index=$((arg_index + 1))
        [[ "${next_index}" -lt "${#args[@]}" ]] || die "--disabled-arm requires a value"
        append_disabled_arm "${args[$next_index]}"
        arg_index="${next_index}"
        ;;
      --disabled-arm=*)
        append_disabled_arm "${arg#*=}"
        ;;
    esac
    arg_index=$((arg_index + 1))
  done
fi

init_arx_can
enable_arx_arms

if [[ "${SHOW_NODE_HELP}" == "1" ]]; then
  run_python examples/hardware/arx/node.py --help
  exit
fi

log "ZMQ observation endpoint: ${OBS_ENDPOINT:-tcp://127.0.0.1:5555} (not HTTP)"
log "ZMQ action endpoint: ${ACTION_ENDPOINT:-tcp://127.0.0.1:5556} (not HTTP)"
if [[ "${#orbbec_white_balance_args[@]}" -gt 0 ]]; then
  log "Orbbec white balance file: ${ORBBEC_WHITE_BALANCE_FILE}"
fi
log "Open the EVA web console from another terminal with:"
log "  eva --config ${EVA_CONFIG:-configs/01_deploy/arx_r5/openpi_qpos.py} --web-port ${WEB_PORT:-8080}"
log "Then browse to: http://127.0.0.1:${WEB_PORT:-8080}"

run_python examples/hardware/arx/node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --left-can-port "${LEFT_CAN_PORT_VALUE}" \
  --right-can-port "${RIGHT_CAN_PORT_VALUE}" \
  --arm-type "${ARM_TYPE:-0}" \
  --gripper-open-pos "${GRIPPER_OPEN_POS:-5.0}" \
  --initial-gripper-scalar "${GRIPPER_INITIAL_SCALAR:-1.0}" \
  --eva-config "${EVA_CONFIG:-configs/01_deploy/arx_r5/openpi_qpos.py}" \
  --rate "${PUBLISH_RATE:-30}" \
  --orbbec-resolution "${ORBBEC_RESOLUTION:-640x480}" \
  --orbbec-fps "${ORBBEC_FPS:-30}" \
  --orbbec-color-format "${ORBBEC_COLOR_FORMAT:-MJPG}" \
  --status-log-interval "${STATUS_LOG_INTERVAL:-5}" \
  "${disabled_args[@]}" \
  "${collection_args[@]}" \
  "${orbbec_white_balance_args[@]}" \
  "$@"
