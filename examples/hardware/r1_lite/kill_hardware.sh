#!/usr/bin/env bash
set -euo pipefail

R1LITE_HOST="${R1LITE_HOST:-r1lite}"
CAT_HOST="${CAT_HOST:-cat}"
R1LITE_SESSION="${R1LITE_SESSION:-eva_r1lite_robot}"
CAT_SESSION="${CAT_SESSION:-eva_r1lite_teleop}"

stop_remote_tmux() {
  local host="$1"
  local session="$2"

  ssh "${host}" bash -s -- "${session}" <<'EOF'
session="$1"
tmux kill-session -t "${session}" 2>/dev/null || true
EOF
}

stop_remote_tmux "${R1LITE_HOST}" "${R1LITE_SESSION}"
stop_remote_tmux "${CAT_HOST}" "${CAT_SESSION}"
