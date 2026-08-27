#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
  echo "Create the repository's .venv before running this script." >&2
  exit 1
fi

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

exec python examples/input_sources/vr_webxr/node.py \
  --host 127.0.0.1 \
  --port 43876 \
  --endpoint tcp://127.0.0.1:8765 \
  --ack-endpoint tcp://127.0.0.1:8766 \
  "$@"
