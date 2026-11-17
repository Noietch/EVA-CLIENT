#!/usr/bin/env bash
set -euo pipefail

PORT="${VR_PORT:-43876}"
TOKEN="${VR_TOKEN:-eva}"
PACKAGE="${PICO_PACKAGE:-org.eva.pico.input}"
SERIAL="${PICO_SERIAL:-}"
ADB="${ADB:-}"
ADB_SERVER_PORT="${ADB_SERVER_PORT:-}"
ADB_WAS_EXPLICIT=0
[[ -n "$ADB" ]] && ADB_WAS_EXPLICIT=1
LAUNCH_ONLY=0
PREPARE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --launch-only) LAUNCH_ONLY=1 ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    --serial=*) SERIAL="${arg#*=}" ;;
    --help|-h)
      cat <<'EOF'
Usage: start.sh [--launch-only|--prepare-only] [--serial=SERIAL]

Launch the EVA-VR native PICO APK. The EVA WebSocket node must already be
running on the host; --launch-only is accepted for the device service.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

adb_is_available() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]]
  else
    command -v "$candidate" >/dev/null 2>&1
  fi
}

adb_devices_for() {
  local candidate="$1"
  local port="$2"
  if [[ -n "$port" ]]; then
    "$candidate" -P "$port" devices 2>/dev/null || true
  else
    "$candidate" devices 2>/dev/null || true
  fi
}

adb_has_device_for() {
  local candidate="$1"
  local port="$2"
  local wanted="$3"
  local devices
  devices="$(adb_devices_for "$candidate" "$port")"
  if [[ -n "$wanted" ]]; then
    awk -v serial="$wanted" '$1 == serial && $2 == "device" {found=1} END {exit !found}' <<<"$devices"
  else
    awk '$2 == "device" {found=1} END {exit !found}' <<<"$devices"
  fi
}

ADB_CANDIDATES=()
ADB_PORTS=()
add_adb_candidate() {
  local candidate="$1"
  local port="$2"
  local existing
  adb_is_available "$candidate" || return 0
  for existing in "${ADB_CANDIDATES[@]:-}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done
  ADB_CANDIDATES+=("$candidate")
  ADB_PORTS+=("$port")
}

if (( ADB_WAS_EXPLICIT )); then
  add_adb_candidate "$ADB" "${ADB_SERVER_PORT:-5037}"
else
  # Use the shared Android ADB server, whether started by VrPico or a shell.
  SYSTEM_ADB="$(command -v adb 2>/dev/null || true)"
  add_adb_candidate "$SYSTEM_ADB" "${ADB_SERVER_PORT:-5037}"
  add_adb_candidate "${ANDROID_HOME:-$HOME/Library/Android/sdk}/platform-tools/adb" "${ADB_SERVER_PORT:-5037}"
  add_adb_candidate "$HOME/.local/android-sdk/platform-tools/adb" "${ADB_SERVER_PORT:-5037}"
fi

if [[ "${#ADB_CANDIDATES[@]}" -eq 0 ]]; then
  echo "adb not found; install Android platform-tools or set ADB=/path/to/adb" >&2
  exit 1
fi

# Select an available ADB binary that can see the connected PICO.
SELECTED_INDEX=-1
for i in "${!ADB_CANDIDATES[@]}"; do
  if adb_has_device_for "${ADB_CANDIDATES[$i]}" "${ADB_PORTS[$i]}" "$SERIAL"; then
    SELECTED_INDEX="$i"
    break
  fi
done
if (( SELECTED_INDEX < 0 )); then
  echo "No authorized PICO device found on the available ADB servers." >&2
  for i in "${!ADB_CANDIDATES[@]}"; do
    echo "--- ${ADB_CANDIDATES[$i]} -P ${ADB_PORTS[$i]} ---" >&2
    adb_devices_for "${ADB_CANDIDATES[$i]}" "${ADB_PORTS[$i]}" >&2
  done
  exit 1
fi

ADB="${ADB_CANDIDATES[$SELECTED_INDEX]}"
ADB_SERVER_PORT="${ADB_PORTS[$SELECTED_INDEX]}"
ADB_ARGS=(-P "$ADB_SERVER_PORT")
adb_cmd() { "$ADB" "${ADB_ARGS[@]}" "$@"; }

SERIALS=()
if [[ -n "$SERIAL" ]]; then
  if adb_has_device_for "$ADB" "$ADB_SERVER_PORT" "$SERIAL"; then
    SERIALS=("$SERIAL")
  fi
else
  while read -r candidate state _; do
    [[ "$state" == "device" ]] || continue
    SERIALS+=("$candidate")
  done < <(adb_cmd devices | tail -n +2)
fi
if [[ "${#SERIALS[@]}" -eq 0 ]]; then
  echo "No authorized PICO device found. Check adb devices -l." >&2
  exit 1
fi

URL="ws://127.0.0.1:${PORT}/ws?token=${TOKEN}"
for SERIAL in "${SERIALS[@]}"; do
  if ! adb_cmd -s "$SERIAL" reverse --list | awk -v port="tcp:${PORT}" \
      '$2 == port && $3 == port {found=1} END {exit !found}'; then
    adb_cmd -s "$SERIAL" reverse "tcp:${PORT}" "tcp:${PORT}" >/dev/null
  fi
done
if (( PREPARE_ONLY )); then
  echo "EVA-VR ADB reverse ready on ${#SERIALS[@]} PICO device(s): ${URL}"
  exit 0
fi
for SERIAL in "${SERIALS[@]}"; do
  adb_cmd -s "$SERIAL" shell am force-stop "$PACKAGE"
  adb_cmd -s "$SERIAL" shell am start -S -n "${PACKAGE}/.MainActivity" \
    --es server_url "$URL" >/dev/null
done

echo "EVA-VR started on ${#SERIALS[@]} PICO device(s): ${URL}"
