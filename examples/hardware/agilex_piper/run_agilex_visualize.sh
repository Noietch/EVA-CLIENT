#!/usr/bin/env bash

set -euo pipefail

DEFAULT_ALOHA_DES_ROOT="/home/agilex/cobot_magic/aloha_des"
ALOHA_DES_ROOT="${ALOHA_DES_ROOT:-${DEFAULT_ALOHA_DES_ROOT}}"
RVIZ_LAUNCH_PKG="${RVIZ_LAUNCH_PKG:-aloha_new_description}"
RVIZ_LAUNCH_FILE="${RVIZ_LAUNCH_FILE:-display_aloha_live.launch}"
MASTER_LEFT_TOPIC="${MASTER_LEFT_TOPIC:-/puppet/joint_left}"
MASTER_RIGHT_TOPIC="${MASTER_RIGHT_TOPIC:-/puppet/joint_right}"
PUPPET_LEFT_TOPIC="${PUPPET_LEFT_TOPIC:-${ACTION_LEFT_TOPIC:-/sim_joint_left}}"
PUPPET_RIGHT_TOPIC="${PUPPET_RIGHT_TOPIC:-${ACTION_RIGHT_TOPIC:-/sim_joint_right}}"
RUN_BUILD="false"

usage() {
    cat <<'EOF'
Usage: ./scripts/piper/run_agilex_visualize.sh [options] [roslaunch args...]

Options:
  --build        Run catkin_make in aloha_des before launch
  -h, --help     Show this help

Env:
  ALOHA_DES_ROOT Override aloha_des workspace path
  RVIZ_LAUNCH_PKG  Override launch package
  RVIZ_LAUNCH_FILE Override launch file
  MASTER_LEFT_TOPIC  Override left arm real-state topic for visualization
  MASTER_RIGHT_TOPIC Override right arm real-state topic for visualization
  PUPPET_LEFT_TOPIC  Override left arm preview topic for visualization
  PUPPET_RIGHT_TOPIC Override right arm preview topic for visualization
  ACTION_LEFT_TOPIC  Backward-compatible alias of PUPPET_LEFT_TOPIC
  ACTION_RIGHT_TOPIC Backward-compatible alias of PUPPET_RIGHT_TOPIC
EOF
}

log() {
    echo "[run_agilex_visualize] $*"
}

has_roslaunch_arg() {
    local name="$1"
    local arg
    for arg in "${ROSLAUNCH_ARGS[@]}"; do
        if [[ "${arg}" == "${name}:="* ]]; then
            return 0
        fi
    done
    return 1
}

ROSLAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            RUN_BUILD="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ROSLAUNCH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -d "${ALOHA_DES_ROOT}" ]]; then
    echo "ALOHA_DES_ROOT not found: ${ALOHA_DES_ROOT}" >&2
    exit 1
fi

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
    echo "ROS Noetic not found at /opt/ros/noetic/setup.bash" >&2
    exit 1
fi

source /opt/ros/noetic/setup.bash

if [[ ! -f "${ALOHA_DES_ROOT}/devel/setup.bash" || "${RUN_BUILD}" == "true" ]]; then
    log "Building aloha_des workspace with catkin_make"
    (
        cd "${ALOHA_DES_ROOT}"
        catkin_make --pkg aloha_new_description
    )
fi

source "${ALOHA_DES_ROOT}/devel/setup.bash"

if ! has_roslaunch_arg "master_left_topic"; then
    ROSLAUNCH_ARGS+=("master_left_topic:=${MASTER_LEFT_TOPIC}")
fi
if ! has_roslaunch_arg "master_right_topic"; then
    ROSLAUNCH_ARGS+=("master_right_topic:=${MASTER_RIGHT_TOPIC}")
fi
if ! has_roslaunch_arg "puppet_left_topic"; then
    ROSLAUNCH_ARGS+=("puppet_left_topic:=${PUPPET_LEFT_TOPIC}")
fi
if ! has_roslaunch_arg "puppet_right_topic"; then
    ROSLAUNCH_ARGS+=("puppet_right_topic:=${PUPPET_RIGHT_TOPIC}")
fi

log "Launching RViz: ${RVIZ_LAUNCH_PKG} ${RVIZ_LAUNCH_FILE}"
log "Master state topics: left=${MASTER_LEFT_TOPIC} right=${MASTER_RIGHT_TOPIC}"
log "Puppet preview topics: left=${PUPPET_LEFT_TOPIC} right=${PUPPET_RIGHT_TOPIC}"
exec roslaunch "${RVIZ_LAUNCH_PKG}" "${RVIZ_LAUNCH_FILE}" "${ROSLAUNCH_ARGS[@]}"
