"""Replay an R2R2R episode on OpenArm through ee_pose_server, step by step.

Reads robot_data.h5 written by R2R2R's data logger (env_*/robot_data/robot_data.h5):
  ee_poses            (N, 14)  [x y z qw qx qy qz] left, then right, robot root frame
  gripper_binary_cmd  (N, 2)   1 = closed (columns: see --gripper-cols)

The arms first go to the episode's start pose (joint angles of the first frame if the episode
has joint_angles + joint_names.txt, else its TCP poses; OMPL, one arm at a time). Then the
trajectory is subsampled like the policy's actions (--subsample, tinydp subsample_steps) and
executed in moves of --steps-per-move waypoints, each one Cartesian path; grippers switch
where their command changes. Run it before a policy to check frames, TCP, gripper and step
timing (RViz: /openarm/<arm>/reference_path and .../planned_path).

--save-start FILE writes that start pose as YAML (no robot needed); the policy node goes there
before every episode (put it in the checkpoint folder as start_pose.yaml).

Modes: dry  = show the trajectory and check it, never move
       step = confirm every move with Enter
       auto = run through, dropping back to step mode on a safety-limit violation

Examples:
  ros2 run openarm_ee_control traj_replay output_data/openarm_cup_table/successes/env_0_... --mode dry
  ros2 run openarm_ee_control traj_replay EPISODE --arms right --steps-per-move 4
  ros2 run openarm_ee_control traj_replay EPISODE --save-start CKPT_DIR/start_pose.yaml
"""

import argparse
import sys
import threading

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from openarm_ee_control.r2r2r_data import load_episode, save_start, start_from_episode
from openarm_ee_control.step_executor import SIDES, SafetyLimits, StepExecutor, ask


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode folder (env_*) or robot_data.h5")
    ap.add_argument("--arms", nargs="+", choices=SIDES, default=list(SIDES))
    ap.add_argument("--mode", choices=["dry", "step", "auto"], default="step")
    ap.add_argument("--subsample", type=int, default=2,
                    help="keep every n-th logged pose (tinydp subsample_steps)")
    ap.add_argument("--steps-per-move", type=int, default=8)
    ap.add_argument("--start", type=int, default=0, help="first logged index")
    ap.add_argument("--end", type=int, default=None, help="last logged index (exclusive)")
    ap.add_argument("--frame", default="world", help="frame the episode poses are in")
    ap.add_argument("--left-ee-link", default="openarm_left_hand_tcp")
    ap.add_argument("--right-ee-link", default="openarm_right_hand_tcp")
    ap.add_argument("--gripper-cols", type=int, nargs=2, default=[0, 1],
                    metavar=("LEFT", "RIGHT"), help="columns of gripper_binary_cmd")
    ap.add_argument("--no-approach", action="store_true",
                    help="don't move to the first pose with OMPL before replaying")
    ap.add_argument("--save-start", metavar="FILE",
                    help="write the episode's first frame as a policy start pose (YAML) and "
                         "exit without moving, e.g. <ckpt-dir>/start_pose.yaml")
    ap.add_argument("--max-step", type=float, default=0.05, help="m, per waypoint")
    ap.add_argument("--workspace-min", type=float, nargs=3, default=[-0.3, -0.9, 0.6])
    ap.add_argument("--workspace-max", type=float, nargs=3, default=[1.0, 0.9, 1.9])
    return ap.parse_args(argv)


def run(node: Node, args) -> int:
    log = node.get_logger()
    episode = load_episode(args.episode, args.gripper_cols)
    sl = slice(args.start, args.end, args.subsample)
    episode = {s: (episode[s][0][sl], episode[s][1][sl]) for s in args.arms}
    n = len(episode[args.arms[0]][0])
    if n == 0:
        log.error("no poses left after --start/--end/--subsample")
        return 2

    limits = SafetyLimits(tuple(args.workspace_min), tuple(args.workspace_max), args.max_step)
    ex = StepExecutor(node, args.frame,
                      {"left": args.left_ee_link, "right": args.right_ee_link}, limits)
    err = ex.wait_ready(args.arms)
    if err:
        log.error(err)
        return 1

    for s in args.arms:
        poses, closed = episode[s]
        ex.show_reference(s, poses)
        d = np.linalg.norm(poses[0][:3, 3] - ex.ee_pose(s)[:3, 3])
        switches = int(np.count_nonzero(np.diff(closed.astype(int))))
        log.info(f"{s}: {n} waypoints, first one {d * 100:.1f} cm from the current TCP, "
                 f"{switches} gripper switch(es)")
        problems = ex.check(s, poses[1:], start=poses[0])
        for p in problems[:10]:
            log.warn(p)
        if len(problems) > 10:
            log.warn(f"... {len(problems) - 10} more")

    if args.mode == "dry":
        log.info("dry run: trajectory published on /openarm/<arm>/reference_path, nothing moved")
        return 0

    mode = args.mode
    if not args.no_approach:
        start = start_from_episode(args.episode, args.start)
        if ask(f"[Enter] go to the start pose ({start.describe()}), q = quit: ") == "q":
            return 0
        res = ex.go_to_start(start, args.arms)
        log.info(f"start pose: {res}")
        if any(st != "succeeded" for st in res.values()):
            return 1
        closed = {s: bool(episode[s][1][0]) for s in args.arms if episode[s][1][0]}
        if closed:
            log.info(f"grippers: {ex.set_grippers(closed)}")

    i = 0
    k = max(1, args.steps_per_move)
    while i < n:
        chunk = {s: (episode[s][0][i:i + k], episode[s][1][i:i + k]) for s in args.arms}
        problems = []
        for s in args.arms:
            ex.show(s, chunk[s][0])
            problems += ex.check(s, chunk[s][0])
        for p in problems:
            log.warn(p)
        if problems and mode == "auto":
            log.warn("safety-limit violation: switching to step mode")
            mode = "step"
        if mode == "step":
            a = ask(f"waypoints {i}..{i + len(chunk[args.arms[0]][0]) - 1} of {n}: "
                    "[Enter] execute, a = auto, s = skip, q = quit: ")
            if a == "q":
                return 0
            if a == "s":
                i += k
                continue
            if a == "a":
                mode = "auto"
        try:
            res = ex.execute(chunk)
        except KeyboardInterrupt:
            ex.stop(args.arms)
            log.warn("stopped (Ctrl+C); back to step mode")
            mode = "step"
            continue
        log.info(f"waypoints {i}..{i + res.executed - 1}: {res.message}")
        if not res.ok:
            mode = "step"
            i += res.executed
            if ask("move failed: [Enter] continue from here, q = quit: ") == "q":
                return 1
            continue
        i += k
    log.info("replay finished")
    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.save_start:
        start = start_from_episode(args.episode, args.start)
        save_start(start, args.save_start)
        print(f"wrote {args.save_start}: {start.describe()}")
        if not start.joints:
            print("note: no joint_angles/joint_names.txt in the episode; the start pose uses "
                  "TCP poses, so the arm posture (elbow) may differ from the sim")
        sys.exit(0)
    # Plain KeyboardInterrupt on Ctrl+C: rclpy's handler would shut the context down, and the
    # stop command could no longer be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("traj_replay")
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
