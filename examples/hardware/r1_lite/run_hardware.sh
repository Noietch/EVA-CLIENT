#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"

export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$HOME/.ros/fastdds_r1lite_super_client.xml}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

if [[ ! -f "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]]; then
  printf '[r1lite-run] ERROR: missing FastDDS profile: %s\n' "${FASTRTPS_DEFAULT_PROFILES_FILE}" >&2
  exit 1
fi

R1LITE_HOST="${R1LITE_HOST:-r1lite}"
CAT_HOST="${CAT_HOST:-cat}"
R1LITE_SESSION="${R1LITE_SESSION:-eva_r1lite_robot}"
CAT_SESSION="${CAT_SESSION:-eva_r1lite_teleop}"
EVA_CONFIG="${EVA_CONFIG:-configs/02_collection/r1lite.py}"

start_remote_tmux() {
  local host="$1"
  local session="$2"
  local command="$3"

  ssh "${host}" bash -s -- "${session}" "${command}" "${host}" <<'EOF'
set -euo pipefail
session="$1"
command="$2"
host="$3"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "[r1lite-run] ${host}:${session} already running"
else
  tmux new-session -d -s "${session}" "${command}"
  echo "[r1lite-run] started ${host}:${session}"
fi
EOF
}

start_remote_tmux "${R1LITE_HOST}" "${R1LITE_SESSION}" "cd ~/r1lite/robot && bash start_robot.sh"
# start_remote_tmux "${CAT_HOST}" "${CAT_SESSION}" "bash /home/cat/start_tele.sh"

has_config=0
for arg in "$@"; do
  if [[ "${arg}" == "--config" || "${arg}" == --config=* ]]; then
    has_config=1
  fi
done

if [[ "${has_config}" == "1" ]]; then
  exec eva "$@"
fi

exec eva --config "${EVA_CONFIG}" "$@"
