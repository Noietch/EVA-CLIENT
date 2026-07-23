#!/usr/bin/env bash
# Read-only inspection: this script never changes CAN state or starts a camera stream.
set -euo pipefail

can_found=0
for path in /sys/class/net/can*; do
  if [[ ! -e "$path" ]]; then
    continue
  fi
  can_found=1
  interface="${path##*/}"
  state="$(<"$path/operstate")"
  echo "$interface state=$state"
  ip -details link show "$interface" | sed 's/^/  /'
done

if [[ "$can_found" == "0" ]]; then
  echo "No can* network interfaces detected"
fi

if command -v lsusb >/dev/null 2>&1; then
  realsense_lines="$(lsusb | grep -Ei 'RealSense|Intel.*(D405|Movidius)|8086:0b5b' || true)"
  if [[ -n "$realsense_lines" ]]; then
    echo "$realsense_lines"
  else
    echo "No RealSense D405 USB entry detected by lsusb"
  fi
else
  echo "lsusb is unavailable; install usbutils for USB camera inspection"
fi
