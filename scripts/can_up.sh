#!/usr/bin/env bash
# Bring up the OpenArm CAN interfaces (CAN-FD, 1 Mbit/s nominal, 5 Mbit/s data).
# Usage: can_up.sh [iface ...]   (default: can0 can1). Needs sudo.
set -euo pipefail
(($#)) || set -- can0 can1
for d in "$@"; do
  sudo ip link set "$d" down 2>/dev/null || true
  sudo ip link set "$d" type can bitrate 1000000 dbitrate 5000000 fd on
  sudo ip link set "$d" up
  ip -details link show "$d" | head -3
done
