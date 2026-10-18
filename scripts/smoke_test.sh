#!/usr/bin/env bash
# Headless end-to-end check on mock hardware: bringup, planned + Cartesian pose goals on both
# arms, and the /openarm/<arm>/target_pose topic interface. Run with `pixi run smoke`.
set -uo pipefail

# Usage: smoke_test.sh [launch file]   (default demo.launch.py = SagresOverdose v1.0 model)
LAUNCH=${1:-demo.launch.py}

# Isolated DDS domain so the mock stack and its test commands can never reach a real robot
# running on the default domain. Refuse to run if anything is already on it.
export ROS_DOMAIN_ID=${SMOKE_DOMAIN_ID:-77}
if [[ -n "$(timeout 10 ros2 node list --no-daemon 2>/dev/null)" ]]; then
  echo "ROS nodes already running on domain $ROS_DOMAIN_ID; refusing to start the mock stack."
  exit 1
fi
LOG=$(mktemp -t openarm_smoke.XXXX.log)
echo "launching $LAUNCH (log: $LOG)"
# Own process group: background jobs of a non-interactive shell ignore SIGINT, so stop the
# whole launch tree with SIGTERM instead.
# table:=false: the test moves (TCP at x 0.2, z 0.85) go through the table box.
setsid ros2 launch openarm_ee_control "$LAUNCH" rviz:=false use_fake_hardware:=true table:=false >"$LOG" 2>&1 &
LAUNCH_PID=$!
cleanup() {
  kill -TERM -- "-$LAUNCH_PID" 2>/dev/null
  for _ in $(seq 1 15); do kill -0 "$LAUNCH_PID" 2>/dev/null || return; sleep 1; done
  kill -KILL -- "-$LAUNCH_PID" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  grep -q "You can start planning now" "$LOG" && grep -q "ee_pose_server.*Ready" "$LOG" && break
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then echo "bringup died, log: $LOG"; exit 1; fi
  sleep 1
done
sleep 3  # controllers are spawned after a short delay

fails=0
run() {
  echo "== send_pose $*"
  local out
  out=$(ros2 run openarm_ee_control send_pose "$@" 2>&1)
  echo "$out" | grep -oE "(OK|FAILED): .*" | sed 's/\x1b\[[0-9;]*m//g'
  echo "$out" | grep -q "OK:" || fails=$((fails + 1))
}

run --arm right --xyz 0.20 0.0 0.15 --relative
run --arm left  --xyz 0.20 0.0 0.15 --relative
run --arm right --xyz 0.0 0.0 0.05 --relative --cartesian
run --arm left  --xyz 0.0 0.05 0.0 --relative --cartesian

echo "== topic interface: /openarm/right/target_pose"
cur=$(timeout 10 ros2 topic echo --once /openarm/right/ee_pose geometry_msgs/msg/PoseStamped)
x=$(echo "$cur" | awk '/position:/{f=1} f&&/x:/{print $2; exit}')
y=$(echo "$cur" | awk '/position:/{f=1} f&&/y:/{print $2; exit}')
z=$(echo "$cur" | awk '/position:/{f=1} f&&/z:/{print $2; exit}')
ori=$(echo "$cur" | awk '/orientation:/{f=1;next} f&&/[xyzw]:/{printf "%s %s, ", $1, $2}' | sed 's/, $//')
timeout 30 ros2 topic echo /openarm/right/status std_msgs/msg/String --field data >"$LOG.status" 2>&1 &
ECHO_PID=$!
sleep 1
ros2 topic pub --once /openarm/right/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world}, pose: {position: {x: $x, y: $y, z: $(python3 -c "print($z - 0.05)")}, orientation: {$ori}}}" >/dev/null
for _ in $(seq 1 25); do grep -qE "succeeded|failed" "$LOG.status" && break; sleep 1; done
kill "$ECHO_PID" 2>/dev/null
status=$(grep -E "succeeded|failed" "$LOG.status" | head -1)
echo "status: ${status:-<none>}"
[[ "$status" == succeeded ]] || fails=$((fails + 1))

if ((fails)); then echo "SMOKE TEST FAILED ($fails step(s)); bringup log: $LOG"; exit 1; fi
echo "SMOKE TEST PASSED"
