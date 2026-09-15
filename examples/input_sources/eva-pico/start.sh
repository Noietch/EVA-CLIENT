#!/usr/bin/env bash
set -euo pipefail

PORT="${VR_PORT:-43876}"
TOKEN="${VR_TOKEN:-eva}"
PACKAGE="${PICO_PACKAGE:-org.eva.pico.input}"
SERIAL="${PICO_SERIAL:-}"
ADB="${ADB:-adb}"
LAUNCH_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --launch-only) LAUNCH_ONLY=1 ;;
    --serial=*) SERIAL="${arg#*=}" ;;
    --help|-h)
      cat <<'EOF'
Usage: start.sh [--launch-only] [--serial=SERIAL]

Launch the EVA-VR native PICO APK. The EVA WebSocket node must already be
running on the host; --launch-only is accepted for the device service.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v "$ADB" >/dev/null 2>&1; then
  echo "adb not found; install Android platform-tools or set ADB=/path/to/adb" >&2
  exit 1
fi

if [[ -z "$SERIAL" ]]; then
  SERIAL="$($ADB devices | awk '$2 == "device" {print $1; exit}')"
fi
if [[ -z "$SERIAL" ]]; then
  echo "No authorized PICO device found. Check adb devices -l." >&2
  exit 1
fi

URL="ws://127.0.0.1:${PORT}/ws?token=${TOKEN}"
"$ADB" -s "$SERIAL" reverse "tcp:${PORT}" "tcp:${PORT}" >/dev/null
"$ADB" -s "$SERIAL" shell am force-stop "$PACKAGE"
"$ADB" -s "$SERIAL" shell am start -S -n "${PACKAGE}/.MainActivity" \
  --es server_url "$URL" >/dev/null

echo "EVA-VR started on ${SERIAL}: ${URL}"
