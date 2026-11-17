#!/usr/bin/env bash
set -euo pipefail

# Called only after Python preflight validates the configured X5 arm interfaces.
# Password entry belongs to the desktop's PolicyKit agent, never to EVA.
if (( $# == 0 )); then
  echo "Usage: $0 CAN_INTERFACE [...]" >&2
  exit 2
fi

# The vendor udev rule exposes each CANable adapter as /dev/arxcan<N>; slcand is what
# turns that tty into the can<N> netdev the X5 SDK opens.
SLCAN_DEVICE_PREFIX="/dev/arxcan"
SLCAND_SPEED="${ARX_CAN_SLCAND_SPEED:-s8}"

run_privileged() {
  local binary="$1"
  shift
  if sudo -n "$binary" "$@" 2>/dev/null; then
    return 0
  fi
  if ! command -v pkexec >/dev/null 2>&1; then
    echo "Cannot run $binary: no PolicyKit authorization dialog is available. Configure CAN locally before retrying." >&2
    return 1
  fi
  if ! pkexec --disable-internal-agent "$binary" "$@"; then
    echo "Cannot run $binary: desktop authorization was cancelled, denied, or unavailable. Authorize CAN on the EVA host, then retry." >&2
    return 1
  fi
}

for interface in "$@"; do
  if [[ ! "$interface" =~ ^can[0-9]+$ ]]; then
    echo "Invalid CAN interface: $interface" >&2
    exit 2
  fi
done

SLCAND="$(command -v slcand || true)"
IP="$(command -v ip || true)"

# Create missing interfaces first: until slcand binds the adapter there is no can<N>
# netdev to inspect or raise.
for interface in "$@"; do
  if [[ -e "/sys/class/net/$interface/type" ]]; then
    continue
  fi
  device="${SLCAN_DEVICE_PREFIX}${interface#can}"
  if [[ ! -e "$device" ]]; then
    echo "Missing SLCAN adapter $device for $interface. Check the CANable connection and /etc/udev/rules.d/arx_can.rules." >&2
    exit 1
  fi
  if [[ -z "$SLCAND" ]]; then
    echo "slcand is required to create $interface." >&2
    exit 1
  fi
  if ! run_privileged "$SLCAND" -o -f "-${SLCAND_SPEED}" "$device" "$interface"; then
    exit 1
  fi
  sleep 0.5
  if [[ ! -e "/sys/class/net/$interface/type" ]]; then
    echo "slcand did not create $interface from $device." >&2
    exit 1
  fi
done

for interface in "$@"; do
  if [[ "$(<"/sys/class/net/$interface/type")" != 280 ]]; then
    echo "Not a CAN interface: $interface" >&2
    exit 2
  fi
  flags="$(<"/sys/class/net/$interface/flags")"
  if (( flags & 1 )); then
    continue
  fi
  if ! run_privileged "$IP" link set "$interface" up; then
    exit 1
  fi
done
