"""Send one end-effector pose goal and wait for the result (needs bringup.launch.py running).

Examples:
  ros2 run openarm_ee_control send_pose --arm right --print
  ros2 run openarm_ee_control send_pose --arm right --xyz 0.0 0.0 0.05 --relative
  ros2 run openarm_ee_control send_pose --arm left  --xyz 0.35 0.25 0.3 --rpy 0 1.57 0
  ros2 run openarm_ee_control send_pose --arm left  --xyz 0 0 -0.05 --relative --cartesian
"""

import argparse
import math
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from openarm_ee_control.commander import ArmCommander, ArmConfig, PlanOptions


def quat_from_rpy(r, p, y) -> Quaternion:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return Quaternion(x=sr * cp * cy - cr * sp * sy, y=cr * sp * cy + sr * cp * sy,
                      z=cr * cp * sy - sr * sp * cy, w=cr * cp * cy + sr * sp * sy)


def fmt(ps: PoseStamped) -> str:
    p, q = ps.pose.position, ps.pose.orientation
    return (f"[{ps.header.frame_id}] xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) "
            f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--xyz", type=float, nargs=3, metavar=("X", "Y", "Z"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rpy", type=float, nargs=3, metavar=("R", "P", "Y"),
                   help="orientation in radians (default: keep current orientation)")
    g.add_argument("--quat", type=float, nargs=4, metavar=("X", "Y", "Z", "W"))
    ap.add_argument("--relative", action="store_true",
                    help="--xyz is an offset from the current EE position")
    ap.add_argument("--cartesian", action="store_true", help="straight-line Cartesian path")
    ap.add_argument("--frame", default="world")
    ap.add_argument("--ee-link", help="default: openarm_<arm>_grasp_frame (v2.0) or "
                    "openarm_<arm>_hand_tcp (v1.0), whichever exists")
    ap.add_argument("--velocity-scaling", type=float, default=0.2)
    ap.add_argument("--print", action="store_true", help="only print the current EE pose")
    return ap.parse_args(argv)


TCP_CANDIDATES = ("openarm_{}_grasp_frame", "openarm_{}_hand_tcp")


def detect_ee_link(cmd: ArmCommander, side: str, timeout: float = 10.0) -> str | None:
    """Return the first TCP link (v2.0, then v1.0 naming) that appears in TF."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pattern in TCP_CANDIDATES:
            link = pattern.format(side)
            if cmd.tf.can_transform(cmd.planning_frame, link, Time()):
                return link
        time.sleep(0.2)
    return None


def run(node: Node, args) -> int:
    arm = ArmConfig(args.arm, f"{args.arm}_arm", args.ee_link or "", f"{args.arm}_arm_controller")
    cmd = ArmCommander(node, arm, planning_frame=args.frame)
    log = node.get_logger()
    if not arm.ee_link:
        arm.ee_link = detect_ee_link(cmd, args.arm)
        if arm.ee_link is None:
            log.error(f"no TCP link found in TF (tried {TCP_CANDIDATES}); pass --ee-link")
            return 1

    current = cmd.current_pose(timeout=10.0)
    log.info(f"current {arm.ee_link}: {fmt(current)}")
    if args.print:
        return 0
    if args.xyz is None:
        log.error("--xyz is required unless --print is given")
        return 2

    target = PoseStamped()
    target.header.frame_id = args.frame
    pos = target.pose.position
    if args.relative:
        c = current.pose.position
        pos.x, pos.y, pos.z = c.x + args.xyz[0], c.y + args.xyz[1], c.z + args.xyz[2]
    else:
        pos.x, pos.y, pos.z = args.xyz
    if args.rpy:
        target.pose.orientation = quat_from_rpy(*args.rpy)
    elif args.quat:
        target.pose.orientation = Quaternion(x=args.quat[0], y=args.quat[1],
                                             z=args.quat[2], w=args.quat[3])
    else:
        target.pose.orientation = current.pose.orientation

    if not cmd.wait_for_servers(30.0):
        log.error("move_group services or the arm controller are not available")
        return 1
    log.info(f"target  {arm.ee_link}: {fmt(target)}")
    res = cmd.move_to_pose(target, PlanOptions(cartesian=args.cartesian,
                                               velocity_scaling=args.velocity_scaling,
                                               acceleration_scaling=args.velocity_scaling))
    (log.info if res.ok else log.error)(str(res))
    if res.ok:
        log.info(f"reached {arm.ee_link}: {fmt(cmd.current_pose())}")
    return 0 if res.ok else 1


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rclpy.init()
    node = Node("send_pose")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        code = run(node, args)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
