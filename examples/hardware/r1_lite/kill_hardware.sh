#!/usr/bin/env bash
set -euo pipefail

R1LITE_HOST="${R1LITE_HOST:-r1lite@192.168.12.12}"
CAT_HOST="${CAT_HOST:-cat@192.168.12.13}"
R1LITE_KILL_CMD="${R1LITE_KILL_CMD:-bash ~/workspace/robot/kill_robot.sh}"
CAT_KILL_CMD="${CAT_KILL_CMD:-bash ~/kill_tele.sh}"
START_R1LITE="${START_R1LITE:-1}"
START_CAT="${START_CAT:-1}"

run_remote_kill() {
  local host="$1"
  local command="$2"

  printf '[r1lite-kill] running %s: %s\n' "${host}" "${command}"
  ssh "${host}" "${command}"
}

if [[ "${START_R1LITE}" == "1" ]]; then
  run_remote_kill "${R1LITE_HOST}" "${R1LITE_KILL_CMD}"
fi

if [[ "${START_CAT}" == "1" ]]; then
  run_remote_kill "${CAT_HOST}" "${CAT_KILL_CMD}"
fi
