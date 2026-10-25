#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
example_dir="$(pwd -P)"

exec "$example_dir/.venv/bin/python" "$example_dir/server.py" \
  --host "${POLICY_HOST:-127.0.0.1}" \
  --port "${POLICY_PORT:-5558}" \
  --checkpoint "${DREAM_CHECKPOINT:-dreamlabs/dream-1-beta}" \
  --robot "${DREAM_ROBOT:-yam}" \
  "$@"
