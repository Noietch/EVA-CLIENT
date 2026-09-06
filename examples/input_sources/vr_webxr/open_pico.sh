#!/usr/bin/env bash
set -euo pipefail

ADB="$(command -v adb || true)"
if [[ -z "$ADB" && -x "$HOME/.local/android-sdk/platform-tools/adb" ]]; then
  ADB="$HOME/.local/android-sdk/platform-tools/adb"
fi
if [[ -z "$ADB" ]]; then
  echo "adb was not found. Install Android Platform Tools and ensure adb is in PATH." >&2
  exit 1
fi

if (( $# > 1 )); then
  echo "Usage: $0 [PICO_SERIAL]" >&2
  exit 2
fi

PICO_SERIAL="${1:-${PICO_SERIAL:-}}"
VR_TOKEN="${VR_TOKEN:-}"

if [[ -z "$VR_TOKEN" ]]; then
  echo "VR_TOKEN is required; use the token printed by the WebXR node." >&2
  exit 2
fi

if [[ -z "$PICO_SERIAL" ]]; then
  connected_devices=()
  while IFS= read -r serial; do
    connected_devices+=("$serial")
  done < <("$ADB" devices | awk 'NR > 1 && $2 == "device" { print $1 }')

  case "${#connected_devices[@]}" in
    0)
      echo "No authorized ADB device is connected." >&2
      echo "Check the USB connection and accept the USB debugging prompt on the headset." >&2
      exit 1
      ;;
    1)
      PICO_SERIAL="${connected_devices[0]}"
      ;;
    *)
      echo "Multiple ADB devices are connected; pass the PICO serial explicitly:" >&2
      printf '  %s\n' "${connected_devices[@]}" >&2
      echo "Usage: $0 <PICO_SERIAL>" >&2
      exit 1
      ;;
  esac
fi

if [[ "$("$ADB" -s "$PICO_SERIAL" get-state 2>/dev/null || true)" != "device" ]]; then
  echo "ADB device is not available or authorized: $PICO_SERIAL" >&2
  exit 1
fi

readonly PORT="${VR_PORT:-43876}"
readonly SCHEME="${VR_SCHEME:-http}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid VR_PORT" >&2
  exit 2
fi
if [[ "$SCHEME" != http && "$SCHEME" != https ]]; then
  echo "Invalid VR_SCHEME" >&2
  exit 2
fi
readonly VR_URL="${SCHEME}://127.0.0.1:${PORT}/?token=${VR_TOKEN}&mode=ar&reload=$(date +%s)"

echo "Using PICO device: $PICO_SERIAL"
"$ADB" -s "$PICO_SERIAL" reverse "tcp:${PORT}" "tcp:${PORT}"
"$ADB" -s "$PICO_SERIAL" shell \
  "am start -S -a android.intent.action.VIEW -d '$VR_URL'"

echo "Opened WebXR on Pico."
