#!/usr/bin/env bash
set -euo pipefail

BITRATE="${I2RT_CAN_BITRATE:-1000000}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/hardware/i2rt/bring_up_can.sh [CAN_INTERFACE ...]

Examples:
  bash examples/hardware/i2rt/bring_up_can.sh
  bash examples/hardware/i2rt/bring_up_can.sh can0 can1 can2
  bash examples/hardware/i2rt/bring_up_can.sh can_follower_l can_follower_r can_leader_l

With no arguments, the script configures can0, can1, and can2 at 1 Mbit/s.
It only changes CAN network-interface state; it does not start an I2RT node,
connect to motors, command an arm, or open a camera stream.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  interfaces=(can0 can1 can2)
else
  interfaces=("$@")
fi

for interface in "${interfaces[@]}"; do
  if [[ ! "$interface" =~ ^can[[:alnum:]_]*$ ]]; then
    echo "Invalid CAN interface name: $interface" >&2
    exit 2
  fi
  if [[ ! -e "/sys/class/net/$interface" ]]; then
    echo "CAN interface not found: $interface" >&2
    exit 2
  fi
done

echo "Requesting sudo once to configure CAN interfaces..."
sudo -v

for interface in "${interfaces[@]}"; do
  if ip -details link show "$interface" | grep -q "bitrate $BITRATE" \
    && ip link show "$interface" | grep -q '<[^>]*UP[^>]*>'; then
    echo "$interface is already UP at $BITRATE bit/s"
    continue
  fi

  sudo ip link set "$interface" down
  sudo ip link set "$interface" up type can bitrate "$BITRATE"
  echo "$interface configured at $BITRATE bit/s"
done

echo
echo "Final CAN state:"
for interface in "${interfaces[@]}"; do
  ip -brief link show "$interface"
  ip -details link show "$interface" | grep -E 'can state|bitrate' | sed 's/^/  /'
done
