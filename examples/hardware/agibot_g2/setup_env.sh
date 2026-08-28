#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO_ROOT="$PWD"
AGIBOT_G2_DIR="$REPO_ROOT/examples/hardware/agibot_g2"
AGIBOT_G2_VENV_DIR="${AGIBOT_G2_VENV_DIR:-$AGIBOT_G2_DIR/.venv}"
AGIBOT_G2_PYTHON_BIN="${AGIBOT_G2_VENV_DIR}/bin/python"
AGIBOT_GDK_LOCAL_PATH="${AGIBOT_GDK_LOCAL_PATH:-}"
AGIBOT_GDK_GIT_URL="${AGIBOT_GDK_GIT_URL:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ -n "${AGIBOT_GDK_LOCAL_PATH}" && -n "${AGIBOT_GDK_GIT_URL}" ]]; then
  echo "Set exactly one of AGIBOT_GDK_LOCAL_PATH or AGIBOT_GDK_GIT_URL." >&2
  exit 1
fi

if [[ -n "${AGIBOT_GDK_LOCAL_PATH}" && ! -e "${AGIBOT_GDK_LOCAL_PATH}" ]]; then
  echo "AgiBot GDK local path not found: ${AGIBOT_GDK_LOCAL_PATH}" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$AGIBOT_G2_VENV_DIR" uv sync --inexact --project "$AGIBOT_G2_DIR"

if [[ -n "${AGIBOT_GDK_LOCAL_PATH}" ]]; then
  uv pip install --python "${AGIBOT_G2_PYTHON_BIN}" "${AGIBOT_GDK_LOCAL_PATH}"
elif [[ -n "${AGIBOT_GDK_GIT_URL}" ]]; then
  uv pip install --python "${AGIBOT_G2_PYTHON_BIN}" "git+${AGIBOT_GDK_GIT_URL}"
fi

if "$AGIBOT_G2_PYTHON_BIN" - <<'PY'
import importlib.util

if importlib.util.find_spec("agibot_gdk") is None:
    raise SystemExit(1)
PY
then
  "$AGIBOT_G2_PYTHON_BIN" - <<'PY'
import agibot_gdk

print(f"agibot_gdk available: {agibot_gdk.__file__}")
PY
  echo
  echo "AgiBot G2 hardware environment is ready: $AGIBOT_G2_VENV_DIR"
elif [[ -n "${AGIBOT_GDK_LOCAL_PATH}" || -n "${AGIBOT_GDK_GIT_URL}" ]]; then
  echo "The selected AgiBot GDK installed but agibot_gdk cannot be imported." >&2
  exit 2
else
  echo
  echo "AgiBot G2 offline/fake environment is ready: $AGIBOT_G2_VENV_DIR"
  echo "Real hardware requires AGIBOT_GDK_LOCAL_PATH or AGIBOT_GDK_GIT_URL."
fi

echo "Runtime: $($AGIBOT_G2_PYTHON_BIN --version 2>&1)"
