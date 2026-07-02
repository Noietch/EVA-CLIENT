#!/bin/bash

set -euo pipefail

DEFAULT_PIPER_ROOT="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic"
DEFAULT_CAMERA_ROOT="/home/agilex/cobot_magic/camera_ws"
PIPER_ROOT="${PIPER_ROOT:-${DEFAULT_PIPER_ROOT}}"
CAMERA_ROOT="${CAMERA_ROOT:-${DEFAULT_CAMERA_ROOT}}"
CAMERA_LAUNCH_PKG="${CAMERA_LAUNCH_PKG:-astra_camera}"
CAMERA_LAUNCH_FILE="${CAMERA_LAUNCH_FILE:-multi_camera.launch}"
MODE="1"
AUTO_ENABLE="true"
RUN_CAN_CONFIG="true"
RUN_BUILD="false"
START_ROSCORE="true"
START_CAMERA="true"
ROSCORE_LOG="${PIPER_ROOT}/roscore.log"
CAMERA_PID=""
ROSCORE_PID=""

usage() {
    cat <<'EOF'
Usage: ./scripts/piper/run_agilex_ros.sh [options]

Options:
  --mode <0|1>           Launch mode for piper node. Default: 1
  --auto-enable <bool>   auto_enable passed to roslaunch. Default: true
  --skip-can             Skip running can_config.sh
  --skip-camera          Skip launching camera_ws
  --build                Run catkin_make before launch
  --no-roscore           Assume roscore is already running
  -h, --help             Show this help

Env:
  PIPER_ROOT             Override Piper workspace path
  CAMERA_ROOT            Override camera workspace path
  CAMERA_LAUNCH_PKG      Override camera launch package
  CAMERA_LAUNCH_FILE     Override camera launch file
EOF
}

log() {
    echo "[start_piper_ros] $*"
}

cleanup() {
    local exit_code=$?
    if [[ -n "${CAMERA_PID}" ]] && kill -0 "${CAMERA_PID}" >/dev/null 2>&1; then
        kill "${CAMERA_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${ROSCORE_PID}" ]] && kill -0 "${ROSCORE_PID}" >/dev/null 2>&1; then
        kill "${ROSCORE_PID}" >/dev/null 2>&1 || true
    fi
    exit "${exit_code}"
}

cleanup_camera_processes() {
    pkill -f "roslaunch ${CAMERA_LAUNCH_PKG} ${CAMERA_LAUNCH_FILE}" >/dev/null 2>&1 || true
    pkill -f astra_camera_node >/dev/null 2>&1 || true
}

wait_for_roscore() {
    local retries=20
    local delay=1

    for ((i=1; i<=retries; i++)); do
        if rostopic list >/dev/null 2>&1; then
            return 0
        fi
        sleep "${delay}"
    done

    return 1
}

wait_for_topics() {
    local retries=20
    local delay=1
    local topics=("$@")

    for ((i=1; i<=retries; i++)); do
        local all_found="true"
        for topic in "${topics[@]}"; do
            if ! rostopic list 2>/dev/null | grep -Fx "${topic}" >/dev/null; then
                all_found="false"
                break
            fi
        done
        if [[ "${all_found}" == "true" ]]; then
            return 0
        fi
        sleep "${delay}"
    done

    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --auto-enable)
            AUTO_ENABLE="${2:-}"
            shift 2
            ;;
        --skip-can)
            RUN_CAN_CONFIG="false"
            shift
            ;;
        --skip-camera)
            START_CAMERA="false"
            shift
            ;;
        --build)
            RUN_BUILD="true"
            shift
            ;;
        --no-roscore)
            START_ROSCORE="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "${PIPER_ROOT}" ]]; then
    echo "PIPER_ROOT not found: ${PIPER_ROOT}" >&2
    exit 1
fi

if [[ "${START_CAMERA}" == "true" && ! -d "${CAMERA_ROOT}" ]]; then
    echo "CAMERA_ROOT not found: ${CAMERA_ROOT}" >&2
    exit 1
fi

if [[ "${MODE}" != "0" && "${MODE}" != "1" ]]; then
    echo "--mode must be 0 or 1" >&2
    exit 1
fi

if [[ "${AUTO_ENABLE}" != "true" && "${AUTO_ENABLE}" != "false" ]]; then
    echo "--auto-enable must be true or false" >&2
    exit 1
fi

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
    echo "ROS Noetic not found at /opt/ros/noetic/setup.bash" >&2
    exit 1
fi

source /opt/ros/noetic/setup.bash
trap cleanup EXIT INT TERM

if [[ "${START_CAMERA}" == "true" && ( ! -f "${CAMERA_ROOT}/devel/setup.bash" || "${RUN_BUILD}" == "true" ) ]]; then
    log "Building camera workspace with catkin_make"
    (
        cd "${CAMERA_ROOT}"
        catkin_make
    )
fi

if [[ ! -f "${PIPER_ROOT}/devel/setup.bash" || "${RUN_BUILD}" == "true" ]]; then
    log "Building workspace with catkin_make"
    (
        cd "${PIPER_ROOT}"
        catkin_make
    )
fi

if [[ "${START_CAMERA}" == "true" ]]; then
    source "${CAMERA_ROOT}/devel/setup.bash"
fi

source "${PIPER_ROOT}/devel/setup.bash"

if [[ "${START_ROSCORE}" == "true" ]]; then
    if rostopic list >/dev/null 2>&1; then
        log "roscore is already running"
    else
        log "Starting roscore in background, log: ${ROSCORE_LOG}"
        nohup roscore >"${ROSCORE_LOG}" 2>&1 &
        ROSCORE_PID=$!
        if ! wait_for_roscore; then
            echo "Failed to start roscore. Check ${ROSCORE_LOG}" >&2
            exit 1
        fi
    fi
fi

if [[ "${START_CAMERA}" == "true" ]]; then
    cleanup_camera_processes
    log "Launching camera stack: ${CAMERA_LAUNCH_PKG} ${CAMERA_LAUNCH_FILE}"
    roslaunch "${CAMERA_LAUNCH_PKG}" "${CAMERA_LAUNCH_FILE}" &
    CAMERA_PID=$!
    if ! wait_for_topics \
        "/camera_f/color/image_raw" \
        "/camera_l/color/image_raw" \
        "/camera_r/color/image_raw"; then
        echo "Failed to detect camera topics. Check terminal output above." >&2
        exit 1
    fi
fi

if [[ "${RUN_CAN_CONFIG}" == "true" ]]; then
    log "Running CAN configuration"
    (
        cd "${PIPER_ROOT}"
        bash ./can_config.sh
    )
fi

log "Launching piper: puppet_mode=${MODE}, master_mode=0, auto_enable=${AUTO_ENABLE}"
cd "${PIPER_ROOT}"
roslaunch piper start_ms_piper.launch puppet_mode:="${MODE}" master_mode:="0" auto_enable:="${AUTO_ENABLE}"
