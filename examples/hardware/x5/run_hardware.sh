#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
X5_VENV_DIR="$PWD/examples/hardware/x5/.venv"
X5_PYTHON_BIN="${X5_VENV_DIR}/bin/python"
[[ -x "${X5_PYTHON_BIN}" ]] || {
  printf '[x5-run] ERROR: X5 environment not found: %s; run examples/hardware/x5/setup_env.sh\n' \
    "${X5_VENV_DIR}" >&2
  exit 1
}
export VIRTUAL_ENV="${X5_VENV_DIR}"
export PATH="${X5_VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"

# Official ARX5_beta SDK-V2, commit 1aa8a7d2ddefced2229f953feffc248be1c0b44d.
X5_SDK_DIR="$PWD/examples/hardware/x5/SDK/X5"
X5_BIMANUAL_DIR="$X5_SDK_DIR/bimanual"
export PYTHONPATH="$X5_SDK_DIR:$PYTHONPATH"

LEFT_CAN_PORT_VALUE="can1"
RIGHT_CAN_PORT_VALUE="can3"
LEFT_CAN_DEVICE="/dev/arxcan1"
RIGHT_CAN_DEVICE="/dev/arxcan3"
ARX_CAN_SLCAND_SPEED="s8"
ARX_CAN_WAIT_SECONDS="5"
X5_PORT_WAIT_SECONDS="5"
INIT_ARX_CAN="1"
ENABLE_ARX_ARMS="0"
RESET_ARX_CAN="${RESET_ARX_CAN:-0}"
RESET_X5_HARDWARE="${RESET_X5_HARDWARE:-0}"
CLEAR_X5_GRIPPER_ERRORS="${CLEAR_X5_GRIPPER_ERRORS:-1}"
X5_RESET_ATTEMPTS="${X5_RESET_ATTEMPTS:-3}"
X5_RESET_SETTLE_SECONDS="${X5_RESET_SETTLE_SECONDS:-1}"
DISABLED_ARMS_VALUE=""
# This machine uses the X5-2025 dual-rail gripper. SDK type 2 expects the
# physical gripper range [-3.4, 0.1] (open -> closed).
ARM_TYPE_VALUE="2"
X5_FILTER_SDK_BANNER="${X5_FILTER_SDK_BANNER:-1}"
SHOW_NODE_HELP=0

X5_NODE_PATH="examples/hardware/x5/node.py"

log() {
  printf '[x5-run] %s\n' "$*"
}

warn() {
  printf '[x5-run] WARN: %s\n' "$*" >&2
}

die() {
  printf '[x5-run] ERROR: %s\n' "$*" >&2
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
  if [[ "${X5_FILTER_SDK_BANNER}" == "0" ]]; then
    "${X5_PYTHON_BIN}" -X faulthandler "$@"
    return
  fi

  set +e
  "${X5_PYTHON_BIN}" -X faulthandler "$@" \
    > >(sed '/^[[:space:]]*ARX方舟无限[[:space:]]*$/d') \
    2> >(sed '/^[[:space:]]*ARX方舟无限[[:space:]]*$/d' >&2)
  local rc=$?
  set -e
  return "${rc}"
}

port_is_listening() {
  local port="$1"

  ss -H -ltn "sport = :${port}" | grep -q .
}

wait_for_startup_ports() {
  local ports=(5555 5556)
  local deadline=$((SECONDS + X5_PORT_WAIT_SECONDS))
  local port occupied

  require_command ss
  while true; do
    occupied=()
    for port in "${ports[@]}"; do
      if port_is_listening "${port}"; then
        occupied+=("${port}")
      fi
    done
    if [[ "${#occupied[@]}" -eq 0 ]]; then
      return
    fi
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      die "startup ports still in use after ${X5_PORT_WAIT_SECONDS}s: ${occupied[*]}"
    fi
    sleep 0.2
  done
}

stop_existing_hardware_node() {
  [[ "${RESET_X5_HARDWARE}" != "0" ]] || return 0

  local proc pid cmdline
  local pids=()
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
    [[ -r "${proc}/cmdline" ]] || continue
    cmdline="$(tr '\0' '\n' <"${proc}/cmdline" 2>/dev/null || true)"
    case "${cmdline}" in
      *"${X5_NODE_PATH}"*) pids+=("${pid}") ;;
    esac
  done

  [[ "${#pids[@]}" -gt 0 ]] || {
    log "no existing X5 hardware node to stop"
    return 0
  }

  log "stopping existing X5 hardware node(s): ${pids[*]}"
  kill -INT "${pids[@]}" 2>/dev/null || true
  for _ in {1..100}; do
    local running=0
    for pid in "${pids[@]}"; do
      kill -0 "${pid}" 2>/dev/null && running=1
    done
    [[ "${running}" -ne 0 ]] || return 0
    sleep 0.1
  done

  warn "X5 hardware node did not stop after SIGINT; sending SIGTERM"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in {1..30}; do
    local running=0
    for pid in "${pids[@]}"; do
      kill -0 "${pid}" 2>/dev/null && running=1
    done
    [[ "${running}" -ne 0 ]] || return 0
    sleep 0.1
  done
  die "existing X5 hardware node did not stop: ${pids[*]}"
}

check_x5_sdk() {
  local sdk_module python_version

  [[ -d "${X5_SDK_DIR}" ]] || die \
    "vendored official X5 SDK root not found: ${X5_SDK_DIR}; restore the tracked SDK files"
  sdk_module="$(find "${X5_BIMANUAL_DIR}" -maxdepth 1 -type f \
    -name '__init__.cpython-312-*.so' -print -quit 2>/dev/null || true)"
  [[ -n "${sdk_module}" ]] || die \
    "official X5 SDK-V2 CPython 3.12 module is missing under ${X5_BIMANUAL_DIR}"
  python_version="$("${X5_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "${python_version}" == "3.12" ]] || die \
    "official X5 SDK-V2 requires Python 3.12; got ${python_version}; run examples/hardware/x5/setup_env.sh"
  run_python -c "import bimanual; assert bimanual.__version__ == '1.0.0'"

  [[ -f "examples/hardware/x5/robot.py" ]] || die \
    "X5 hardware adapter is missing: examples/hardware/x5/robot.py"
  [[ -f "${X5_NODE_PATH}" ]] || die "X5 execution node is missing: ${X5_NODE_PATH}"
  log "X5 SDK: official ARX5_beta SDK-V2 1.0.0 root=${X5_SDK_DIR}"
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

  if interface_exists "${iface}" && [[ "${RESET_ARX_CAN}" != "0" ]]; then
    log "resetting ${iface} (${device})"
    run_root ip link set "${iface}" down >/dev/null 2>&1 || true
    stop_matching_slcand "${device}" "${iface}"
    for _ in {1..20}; do
      interface_exists "${iface}" || break
      sleep 0.1
    done
    interface_exists "${iface}" && die "${iface} still exists after slcand reset"
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

reset_x5_arms() {
  [[ "${RESET_X5_HARDWARE}" != "0" ]] || return 0

  [[ "${X5_RESET_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || \
    die "X5_RESET_ATTEMPTS must be a positive integer"

  ARX_RESET_SPECS="left_arm:${LEFT_CAN_PORT_VALUE};right_arm:${RIGHT_CAN_PORT_VALUE}" \
  ARX_DISABLED_ARMS="${DISABLED_ARMS_VALUE}" \
  ARX_ARM_TYPE="${ARM_TYPE_VALUE}" \
  ARX_RESET_ATTEMPTS="${X5_RESET_ATTEMPTS}" \
  ARX_RESET_SETTLE_SECONDS="${X5_RESET_SETTLE_SECONDS}" \
  run_python - <<'PY'
import os
import time

import numpy as np

from examples.hardware.x5.sdk import SingleArm
from examples.hardware.x5.robot import X5_ARM_JOINT_LIMITS


def disabled_groups(raw: str) -> set[str]:
    groups = set()
    for item in raw.replace(",", " ").split():
        groups.add({"left": "left_arm", "right": "right_arm"}.get(item, item))
    return groups


def joint_names(values: tuple[object, ...]) -> str:
    names = []
    for value in values:
        text = str(value)
        if text.isdigit():
            index = int(text)
            text = f"J{index + 1}" if 0 <= index < 6 else f"J{index}"
        names.append(text)
    return ",".join(names) or "none"


disabled = disabled_groups(os.environ.get("ARX_DISABLED_ARMS", ""))
arm_type = int(os.environ["ARX_ARM_TYPE"])
attempts = int(os.environ["ARX_RESET_ATTEMPTS"])
settle_s = float(os.environ["ARX_RESET_SETTLE_SECONDS"])
failed = []

for spec in os.environ["ARX_RESET_SPECS"].split(";"):
    group_name, can_port = spec.split(":", 1)
    if group_name in disabled:
        print(f"[x5-reset] {group_name} skipped: disabled", flush=True)
        continue

    last_error = "unknown reset failure"
    for attempt in range(1, attempts + 1):
        arm = None
        healthy = False
        cleanup_errors = []
        try:
            print(
                f"[x5-reset] {group_name} attempt {attempt}/{attempts}: "
                f"enable and inspect {can_port}",
                flush=True,
            )
            arm = SingleArm({"can_port": can_port, "type": arm_type})
            time.sleep(max(settle_s, 0.0))
            offline = tuple(arm.offline_joints)
            fault = arm.fault
            status = arm.get_status()
            print(
                f"[x5-reset] {group_name}: offline_joints={joint_names(offline)} "
                f"fault={fault!r} status={status!r}",
                flush=True,
            )
            if not offline and fault is None:
                healthy = True
            else:
                last_error = f"offline_joints={joint_names(offline)}, fault={fault!r}"

            if healthy and group_name == "right_arm":
                start = np.asarray(arm.get_joint_positions(), dtype=float)[:6]
                if start.shape != (6,) or not np.all(np.isfinite(start)):
                    raise RuntimeError(f"invalid qpos before J2 probe: {start}")
                probe_target = start.copy()
                probe_step = 0.01
                if probe_target[1] + probe_step > X5_ARM_JOINT_LIMITS[1, 1] - 0.005:
                    probe_step = -probe_step
                probe_target[1] += probe_step

                arm.set_joint_positions(start.tolist(), duration=0.3)
                time.sleep(0.5)
                accepted = arm.set_joint_positions(probe_target.tolist(), duration=0.8)
                time.sleep(1.0)
                reached = np.asarray(arm.get_joint_positions(), dtype=float)[:6]
                motion = abs(float(reached[1] - start[1]))
                current = float(np.asarray(arm.get_joint_currents(), dtype=float)[1])
                print(
                    f"[x5-reset] right_arm J2 probe: accepted={accepted!r} "
                    f"command={probe_step:+.5f} motion={motion:.5f} current={current:.3f}",
                    flush=True,
                )
                arm.set_joint_positions(start.tolist(), duration=0.8)
                time.sleep(1.0)
                if not accepted or motion < 0.004:
                    healthy = False
                    last_error = (
                        f"J2 motion probe failed: command={probe_step:+.5f}, "
                        f"motion={motion:.5f}, current={current:.3f}"
                    )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[x5-reset] {group_name}: {last_error}", flush=True)
        finally:
            if arm is not None:
                try:
                    arm.disable()
                except Exception as exc:
                    cleanup_errors.append(f"disable failed: {exc}")
                    print(f"[x5-reset] {group_name}: disable failed: {exc}", flush=True)
                try:
                    arm.close()
                except Exception as exc:
                    cleanup_errors.append(f"close failed: {exc}")
                    print(f"[x5-reset] {group_name}: close failed: {exc}", flush=True)
        if cleanup_errors:
            last_error = ", ".join(cleanup_errors)
        elif healthy:
            last_error = ""
            break
        if last_error and attempt < attempts:
            time.sleep(max(settle_s, 0.0))

    if last_error:
        failed.append(f"{group_name}({last_error})")
    else:
        print(f"[x5-reset] {group_name} reset passed", flush=True)

if failed:
    raise SystemExit("X5 reset failed: " + "; ".join(failed))

print("[x5-reset] both enabled arms passed; starting hardware node", flush=True)
PY
}

clear_x5_gripper_errors() {
  [[ "${CLEAR_X5_GRIPPER_ERRORS}" != "0" ]] || return 0

  ARX_GRIPPER_SPECS="left_arm:${LEFT_CAN_PORT_VALUE};right_arm:${RIGHT_CAN_PORT_VALUE}" \
  ARX_DISABLED_ARMS="${DISABLED_ARMS_VALUE}" \
  run_python - <<'PY'
import os

from examples.hardware.x5.sdk import clear_gripper_error


disabled = set()
for item in os.environ.get("ARX_DISABLED_ARMS", "").replace(",", " ").split():
    disabled.add({"left": "left_arm", "right": "right_arm"}.get(item, item))

for spec in os.environ["ARX_GRIPPER_SPECS"].split(";"):
    group_name, can_port = spec.split(":", 1)
    if group_name in disabled:
        print(f"[x5-run] {group_name} gripper fault clear skipped: disabled", flush=True)
        continue
    result = clear_gripper_error(can_port)
    print(
        f"[x5-run] {group_name} gripper state cleared on {can_port}: "
        f"error={result['error']} ({result['error_name']}) "
        f"temperature={result['temperature']:.1f}C",
        flush=True,
    )
PY
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
    log "CAN setup skipped: all X5 arms are disabled"
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
    log "separate arm pre-enable skipped; hardware node will enable both arms"
    return
  }

  ARX_ENABLE_SPECS="left_arm:${LEFT_CAN_PORT_VALUE};right_arm:${RIGHT_CAN_PORT_VALUE}" \
  ARX_DISABLED_ARMS="${DISABLED_ARMS_VALUE}" \
  ARX_ARM_TYPE="${ARM_TYPE_VALUE}" \
  run_python - <<'PY'
import os

import numpy as np

from examples.hardware.x5.sdk import SingleArm


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
        print(f"[x5-run] {group_name} enable skipped: disabled")
        continue
    arm = SingleArm({"can_port": can_port, "type": arm_type, "num_joints": 7})
    positions = np.asarray(arm.get_joint_positions(), dtype=float)
    if positions.shape[0] < 6:
        raise RuntimeError(f"{group_name} returned {positions.shape[0]} joints")
    arm.set_joint_positions(positions[:6].tolist())
    enabled += 1
    print(f"[x5-run] {group_name} enabled on {can_port} type={arm_type}")
    arm.protect_mode()
    arm.close()

if enabled == 0:
    print("[x5-run] no X5 arms enabled")
PY
}

for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      INIT_ARX_CAN=0
      ENABLE_ARX_ARMS=0
      RESET_ARX_CAN=0
      RESET_X5_HARDWARE=0
      CLEAR_X5_GRIPPER_ERRORS=0
      SHOW_NODE_HELP=1
      ;;
  esac
done

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

if [[ "${SHOW_NODE_HELP}" == "0" ]]; then
  check_x5_sdk
  stop_existing_hardware_node
  wait_for_startup_ports
fi
init_arx_can
clear_x5_gripper_errors
reset_x5_arms
enable_arx_arms

if [[ "${SHOW_NODE_HELP}" == "1" ]]; then
  run_python "${X5_NODE_PATH}" --help
  exit
fi

log "starting X5 motors + D405 hardware node"
log "ZMQ observation endpoint: tcp://127.0.0.1:5555 (not HTTP)"
log "ZMQ action endpoint: tcp://127.0.0.1:5556 (not HTTP)"

run_python "${X5_NODE_PATH}" \
  --obs-endpoint "tcp://127.0.0.1:5555" \
  --action-endpoint "tcp://127.0.0.1:5556" \
  --left-can-port "${LEFT_CAN_PORT_VALUE}" \
  --right-can-port "${RIGHT_CAN_PORT_VALUE}" \
  --arm-type "${ARM_TYPE_VALUE}" \
  --gripper-open-pos "-3.4" \
  --gripper-close-pos "0.1" \
  --initial-gripper-scalar "1.0" \
  --eva-config "configs/01_deploy/arx_x5/openpi_qpos.py" \
  --rate "30" \
  --realsense-resolution "640x480" \
  --realsense-fps "30" \
  --realsense-profile "day" \
  --status-log-interval "${X5_STATUS_LOG_INTERVAL_S:-1}" \
  "$@"
