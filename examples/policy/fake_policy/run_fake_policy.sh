#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
repo_root="$(pwd -P)"
export PYTHONPATH="$repo_root:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

exec "$repo_root/.venv/bin/python" "$repo_root/examples/policy/fake_policy/fake_policy_server.py" \
  --host "${POLICY_HOST:-127.0.0.1}" \
  --port "${POLICY_PORT:-9000}" \
  --chunk-size "${CHUNK_SIZE:-50}" \
  "$@"
