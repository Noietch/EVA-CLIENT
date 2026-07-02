#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python examples/hardware/ur5e/teleop.py \
  --port "${TELEOP_PORT:-/dev/ttyACM0}" \
  "$@"
