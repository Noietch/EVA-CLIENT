#!/usr/bin/env bash
set -euo pipefail

ADB="$(command -v adb || true)"
if [[ -z "$ADB" && -x "$HOME/.local/android-sdk/platform-tools/adb" ]]; then
  ADB="$HOME/.local/android-sdk/platform-tools/adb"
fi
if [[ -z "$ADB" && -x "$HOME/Library/Android/sdk/platform-tools/adb" ]]; then
  ADB="$HOME/Library/Android/sdk/platform-tools/adb"
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
readonly HAPTIC_TEST="${VR_HAPTIC_TEST:-0}"
default_mode=ar
if [[ "$HAPTIC_TEST" == 1 ]]; then default_mode=vr; fi
readonly MODE="${VR_MODE:-$default_mode}"
if [[ "$MODE" != ar && "$MODE" != vr ]] || [[ "$HAPTIC_TEST" != 0 && "$HAPTIC_TEST" != 1 ]]; then
  echo "VR_MODE must be ar or vr; VR_HAPTIC_TEST must be 0 or 1." >&2
  exit 2
fi
# Tokens are URL query values. The node generates URL-safe tokens by default.
if [[ ! "$VR_TOKEN" =~ ^[a-zA-Z0-9._~-]+$ ]]; then
  echo "VR_TOKEN must use URL-safe characters (letters, digits, dot, underscore, tilde, hyphen)." >&2
  exit 2
fi
readonly VR_URL="${SCHEME}://127.0.0.1:${PORT}/?token=${VR_TOKEN}&mode=${MODE}&haptic_test=${HAPTIC_TEST}&reload=$(date +%s)"
readonly BROWSER_PACKAGE="${WOLVIC_PACKAGE:-com.pico.browser}"
if [[ ! "$BROWSER_PACKAGE" =~ ^[a-zA-Z0-9_.]+$ ]]; then
  echo "Invalid browser package name." >&2
  exit 2
fi

echo "Using PICO device: $PICO_SERIAL"
package_path="$("$ADB" -s "$PICO_SERIAL" shell pm path "$BROWSER_PACKAGE" 2>/dev/null || true)"
if [[ "$package_path" != *package:* ]]; then
  echo "Browser package was not found: $BROWSER_PACKAGE" >&2
  echo "Install a browser first, or set WOLVIC_PACKAGE to the installed package name." >&2
  exit 1
fi
"$ADB" -s "$PICO_SERIAL" reverse "tcp:${PORT}" "tcp:${PORT}"
# adb shell joins arguments before remote shell parsing: preserve URL ampersands.
launch_status=0
launch_output="$("$ADB" -s "$PICO_SERIAL" shell \
  "am start -S -a android.intent.action.VIEW -d '$VR_URL' -p '$BROWSER_PACKAGE'" 2>&1)" || launch_status=$?
if (( launch_status != 0 )) || [[ "$launch_output" == *Error:* || "$launch_output" == *Exception* ]]; then
  echo "Browser launch failed: ${launch_output//"$VR_TOKEN"/[redacted]}" >&2
  exit 1
fi

echo "Opened WebXR in browser ($BROWSER_PACKAGE)."
if [[ "$HAPTIC_TEST" == 1 ]]; then
  echo "Haptic test: enter XR, then pull each controller trigger. Teleoperation frames are disabled."
fi
