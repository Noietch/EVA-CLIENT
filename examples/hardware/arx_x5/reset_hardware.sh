#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RESET_ARX_CAN=1
export RESET_ARX_X5_HARDWARE=1
exec bash "${SCRIPT_DIR}/run_hardware.sh" "$@"
