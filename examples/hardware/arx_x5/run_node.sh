#!/usr/bin/env bash
# In-sim teleoperation launcher for ARX X5.
#
# Brings up the full teleop -> eva_client -> eva_sim collection chain:
#   1. eva_sim (its own venv) as the ARX X5 follower + renderer, binding the
#      observation/action ZMQ endpoints.
#   2. this repo's ARX X5 sim-teleop node (eva-client venv), which reads the
#      Alicia-D leader arms through the real ARX example SDK and publishes
#      absolute QPos to eva_sim.
#
# EVA itself records the episodes; start it in a THIRD terminal once this script
# reports both processes are up:
#   eva --config configs/02_collection/arx_x5.py --web-port 8080
#
# Required: point EVA_SIM_DIR at the eva_sim checkout (or pass it as $1).
set -euo pipefail

cd "$(dirname "$0")/../../.."
EVA_CLIENT_DIR="$PWD"

EVA_SIM_DIR="${EVA_SIM_DIR:-${1:-}}"
TASK="${TASK:-pick_and_place}"
OBS_ENDPOINT="${OBS_ENDPOINT:-tcp://127.0.0.1:5555}"
ACTION_ENDPOINT="${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}"
PUBLISH_RATE="${PUBLISH_RATE:-30}"
ALICIA_READER="${ALICIA_READER:-duo}"
RECORD_ROOT="${RECORD_ROOT:-output/episodes}"
EVA_SIM_EXTRA_ARGS="${EVA_SIM_EXTRA_ARGS:-}"

log() { printf '[arx-x5-run] %s\n' "$*"; }
die() { printf '[arx-x5-run] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -n "${EVA_SIM_DIR}" ]] || die "set EVA_SIM_DIR (or pass the eva_sim path as the first arg)"
[[ -d "${EVA_SIM_DIR}" ]] || die "eva_sim dir not found: ${EVA_SIM_DIR}"
[[ -f "${EVA_SIM_DIR}/scripts/run.py" ]] || die "eva_sim run.py missing under ${EVA_SIM_DIR}"
[[ -d "${EVA_SIM_DIR}/.venv" ]] || die "eva_sim venv missing: ${EVA_SIM_DIR}/.venv"

SIM_PID=""
cleanup() {
  [[ -n "${SIM_PID}" ]] && kill "${SIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) eva_sim follower + renderer, in its own venv (needs Isaac Sim + EULA).
log "starting eva_sim (${EVA_SIM_DIR}) task=${TASK} robot=arx_x5"
(
  cd "${EVA_SIM_DIR}"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
  # Proxy must be off for local ZMQ / Isaac Sim runtime.
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  exec python scripts/run.py \
    --task "${TASK}" \
    --robot arx_x5 \
    --obs-endpoint "${OBS_ENDPOINT}" \
    --action-endpoint "${ACTION_ENDPOINT}" \
    --record-root "${RECORD_ROOT}" \
    ${EVA_SIM_EXTRA_ARGS}
) &
SIM_PID=$!
log "eva_sim pid=${SIM_PID}; waiting for it to bind ${OBS_ENDPOINT}"
sleep "${EVA_SIM_WARMUP_SECONDS:-20}"
kill -0 "${SIM_PID}" 2>/dev/null || die "eva_sim exited during startup; check its logs above"

# 2) ARX X5 sim-teleop node, in the eva-client venv (reads the leader arms).
cd "${EVA_CLIENT_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

log "eva_sim is up. In a THIRD terminal, start EVA to record:"
log "  cd ${EVA_CLIENT_DIR} && source .venv/bin/activate"
log "  eva --config configs/02_collection/arx_x5.py --web-port ${WEB_PORT:-8080}"
log "starting ARX X5 sim-teleop node (reader=${ALICIA_READER}, rate=${PUBLISH_RATE}Hz)"

exec python examples/hardware/arx_x5/node.py \
  --obs-endpoint "${OBS_ENDPOINT}" \
  --action-endpoint "${ACTION_ENDPOINT}" \
  --rate "${PUBLISH_RATE}" \
  --alicia-reader "${ALICIA_READER}" \
  "$@"
