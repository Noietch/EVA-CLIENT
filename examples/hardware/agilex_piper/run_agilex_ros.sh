#!/bin/bash

set -euo pipefail

DEFAULT_PIPER_ROOT="/home/agilex/cobot_magic/Piper_ros_private-ros-noetic"
DEFAULT_CAMERA_ROOT="/home/agilex/cobot_magic/camera_ws"
PIPER_ROOT="${PIPER_ROOT:-${DEFAULT_PIPER_ROOT}}"
CAMERA_ROOT="${CAMERA_ROOT:-${DEFAULT_CAMERA_ROOT}}"
PIPER_PYTHON_ENV="${PIPER_PYTHON_ENV:-/home/agilex/miniconda3/envs/aloha}"
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
PIPER_LAUNCH_PID=""

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
  PIPER_PYTHON_ENV       Python environment containing piper_sdk
  CAMERA_LAUNCH_PKG      Override camera launch package
  CAMERA_LAUNCH_FILE     Override camera launch file
EOF
}

log() {
    echo "[start_piper_ros] $*"
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    stop_process_tree "${PIPER_LAUNCH_PID}"
    cleanup_piper_processes
    stop_process_tree "${CAMERA_PID}"
    cleanup_camera_processes
    stop_process_tree "${ROSCORE_PID}"
    exit "${exit_code}"
}

stop_process_tree() {
    local root_pid="$1"
    [[ -n "${root_pid}" && "${root_pid}" =~ ^[0-9]+$ ]] || return
    kill -0 "${root_pid}" >/dev/null 2>&1 || return

    local -a tree_pids=()
    collect_process_tree "${root_pid}" tree_pids
    local pid
    for pid in "${tree_pids[@]}"; do
        kill -INT "${pid}" >/dev/null 2>&1 || true
    done
    for _ in {1..20}; do
        local alive="false"
        for pid in "${tree_pids[@]}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                alive="true"
                break
            fi
        done
        [[ "${alive}" == "true" ]] || return
        sleep 0.1
    done
    for pid in "${tree_pids[@]}"; do
        kill -TERM "${pid}" >/dev/null 2>&1 || true
    done
    sleep 0.2
    for pid in "${tree_pids[@]}"; do
        kill -KILL "${pid}" >/dev/null 2>&1 || true
    done
}

collect_process_tree() {
    local root_pid="$1"
    local -n output="$2"
    output+=("${root_pid}")
    local child_pid
    while read -r child_pid; do
        [[ -n "${child_pid}" ]] || continue
        collect_process_tree "${child_pid}" "$2"
    done < <(pgrep -P "${root_pid}" 2>/dev/null || true)
}

cleanup_camera_processes() {
    stop_matching_processes \
        "roslaunch ${CAMERA_LAUNCH_PKG} ${CAMERA_LAUNCH_FILE}" \
        "${CAMERA_ROOT}/devel/lib/${CAMERA_LAUNCH_PKG}/astra_camera_node"
}

cleanup_piper_processes() {
    stop_matching_processes \
        "roslaunch piper start_ms_piper.launch" \
        "${PIPER_ROOT}/src/piper/scripts/piper_start_ms_node.py"
}

stop_matching_processes() {
    local pattern pid
    local -a pids=()
    for pattern in "$@"; do
        while read -r pid; do
            [[ -n "${pid}" && "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
            if [[ ! " ${pids[*]} " =~ " ${pid} " ]]; then
                pids+=("${pid}")
            fi
        done < <(pgrep -f -- "${pattern}" 2>/dev/null || true)
    done
    for pid in "${pids[@]}"; do
        stop_process_tree "${pid}"
    done
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

# The upstream Piper ROS node uses /usr/bin/env python3. Device starts inherit
# EVA's Python 3.11 environment, while the verified Piper SDK is installed in
# the ROS workspace's Python 3.8 runtime. Put that runtime first for roslaunch
# child nodes without changing EVA's own environment.
if [[ -x "${PIPER_PYTHON_ENV}/bin/python3" ]]; then
    export PATH="${PIPER_PYTHON_ENV}/bin:${PATH}"
fi
if ! python3 -c 'import piper_sdk' >/dev/null 2>&1; then
    echo "Piper Python environment cannot import piper_sdk: ${PIPER_PYTHON_ENV}" >&2
    echo "Set PIPER_PYTHON_ENV to the environment that contains piper_sdk." >&2
    exit 1
fi

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
    if ! (
        cd "${PIPER_ROOT}"
        sudo -n bash ./can_config.sh
    ); then
        if ! command -v pkexec >/dev/null 2>&1; then
            echo "Cannot configure Piper CAN: no PolicyKit authorization dialog is available. Configure CAN locally before retrying." >&2
            exit 1
        fi
        pkexec --disable-internal-agent bash -c 'cd "$1" && exec bash ./can_config.sh' _ "${PIPER_ROOT}"
    fi
fi

log "Launching piper: puppet_mode=${MODE}, master_mode=0, auto_enable=${AUTO_ENABLE}"
cd "${PIPER_ROOT}"
cleanup_piper_processes
roslaunch piper start_ms_piper.launch puppet_mode:="${MODE}" master_mode:="0" auto_enable:="${AUTO_ENABLE}" &
PIPER_LAUNCH_PID=$!
if ! wait_for_topics \
    "/master/joint_left" \
    "/master/joint_right" \
    "/puppet/joint_left" \
    "/puppet/joint_right"; then
    echo "Failed to detect Piper leader/follower topics. Check terminal output above." >&2
    exit 1
fi
log "Piper leader/follower topics are ready"
wait "${PIPER_LAUNCH_PID}"
