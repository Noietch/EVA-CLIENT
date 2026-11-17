#!/usr/bin/env bash
set -euo pipefail

# Called only after Python preflight validates the configured follower interfaces.
# Password entry belongs to the desktop's PolicyKit agent, never to EVA.
if (( $# == 0 )); then
  echo "Usage: $0 CAN_INTERFACE [...]" >&2
  exit 2
fi
for interface in "$@"; do
  if [[ ! "$interface" =~ ^[a-zA-Z0-9_-]+$ ]] || [[ ! -e "/sys/class/net/$interface/type" ]]; then
    echo "Invalid CAN interface: $interface" >&2
    exit 2
  fi
  if [[ "$(<"/sys/class/net/$interface/type")" != 280 ]]; then
    echo "Not a CAN interface: $interface" >&2
    exit 2
  fi
done
for interface in "$@"; do
  flags="$(<"/sys/class/net/$interface/flags")"
  if (( flags & 1 )); then
    continue
  fi
  if ! sudo -n ip link set "$interface" up type can bitrate 1000000 2>/dev/null; then
    if ! command -v pkexec >/dev/null 2>&1; then
      echo "Cannot enable $interface: no PolicyKit authorization dialog is available. Configure CAN locally before retrying." >&2
      exit 1
    fi
    if ! pkexec --disable-internal-agent "$(command -v ip)" link set "$interface" up type can bitrate 1000000; then
      echo "Cannot enable $interface: desktop authorization was cancelled, denied, or unavailable. Authorize CAN on the EVA host, then retry." >&2
      exit 1
    fi
  fi
done
