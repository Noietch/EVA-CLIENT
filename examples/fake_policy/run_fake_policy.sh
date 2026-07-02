#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}:$PWD/src"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python examples/fake_policy/fake_policy_server.py \
  --host "${POLICY_HOST:-127.0.0.1}" \
  --port "${POLICY_PORT:-9000}" \
  --chunk-size "${CHUNK_SIZE:-50}" \
  "$@"
