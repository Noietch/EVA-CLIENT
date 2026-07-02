#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python examples/hardware/franka/fake_node.py \
  --obs-endpoint "${OBS_ENDPOINT:-tcp://127.0.0.1:5555}" \
  --action-endpoint "${ACTION_ENDPOINT:-tcp://127.0.0.1:5556}" \
  --rate "${PUBLISH_RATE:-30}" \
  "$@"
