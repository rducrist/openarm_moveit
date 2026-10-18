"""Run an R2R2R-trained tinydp diffusion policy on OpenArm, step by step, via ee_pose_server.

Episode:
  go to start pose   joint angles (or TCP poses) the sim episodes start from: --start, or
                     <ckpt-dir>/start_pose.yaml (write it with traj_replay --save-start)
  start policy       then repeat steps until --max-steps, 'h' (back to start) or 'q'

One step:
  observe  camera frame + TCP poses of both arms (TF at the frame's stamp) + commanded grippers
  predict  chunk of absolute TCP poses + gripper values for the policy's arm(s)
  check    workspace box and step sizes; the chunk is shown on /openarm/<arm>/planned_path
  execute  the first --steps-per-move waypoints as one Cartesian move per arm; grippers switch
           where the predicted gripper value crosses --gripper-threshold
  record   history entries for the executed waypoints (observation.py)

Modes: dry  = predict and show only; the policy never moves the arm
       step = Enter executes each step (default)
       auto = run until --max-steps; Ctrl+C stops the arm and returns to step mode
Going to the start pose and starting the policy are always confirmed with Enter.

Needs the bringup (demo.launch.py) with ee_pose_server and a camera publishing
sensor_msgs/Image. Run it in the pixi `policy` environment (torch + tinydp):
  pixi run -e policy ros2 run openarm_policy policy_node --ckpt-dir /path/to/ckpt \\
      --image-topic /camera/color/image_raw --mode dry
--fake replaces the network by small scripted motions (no torch needed) to test the loop.
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from openarm_ee_control.geometry import poses_from_vec10, vec10_from_pose
from openarm_ee_control.r2r2r_data import load_start
from openarm_ee_control.step_executor import SIDES, SafetyLimits, StepExecutor, StepResult, ask
from openarm_policy.observation import (Camera, History, Observation, RecordedObservation,
                                        pick_history)

COLS = {"left": slice(0, 10), "right": slice(10, 20)}


class FakePolicy:
    """Scripted stand-in with the TinyDPPolicy interface: 3 mm steps along a slowly turning
    direction in the y-z plane, gripper closed on every other pair of predictions."""

    def __init__(self, arms, obs_horizon=4, action_horizon=16, image_size=224):
        self.arms, self.obs_horizon = arms, obs_horizon
        self.action_horizon, self.image_size = action_horizon, image_size
        self.calls = 0

    def describe(self) -> str:
        return (f"FAKE policy: arms={self.arms}, obs history {self.obs_horizon}, "
                f"chunk {self.action_horizon}")

    def predict(self, images, proprio):
        out = np.repeat(proprio[-1:].astype(np.float64), self.action_horizon, axis=0)
        phi = 2 * np.pi * self.calls / 12
        direction = np.array([0.0, np.cos(phi), np.sin(phi)])
        grip = float((self.calls // 2) % 2)
        for s in self.arms:
            c = COLS[s]
            for j in range(self.action_horizon):
                out[j, c][:3] = proprio[-1, c][:3] + 0.003 * (j + 1) * direction
                out[j, c.start + 9] = grip
        self.calls += 1
        return out


def detect_image_topic(node: Node, timeout: float = 5.0) -> str | None:
    deadline = time.monotonic() + timeout
    topics = []
    while time.monotonic() < deadline:
        topics = [n for n, types in node.get_topic_names_and_types()
                  if "sensor_msgs/msg/Image" in types]
        if len(topics) == 1:
            return topics[0]
        time.sleep(0.5)
    node.get_logger().error(f"pass --image-topic; sensor_msgs/Image topics: {topics or 'none'}")
    return None


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", help="folder with run.yaml, action_statistics.json, "
                    "checkpoint_<N>.pt (and optionally start_pose.yaml)")
    ap.add_argument("--ckpt-id", type=int, help="default: highest N")
    ap.add_argument("--weights", choices=["model", "ema"], default="model",
                    help="'model' is what tinydp's DiffusionWrapper loads")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ddim-steps", type=int, default=0,
                    help="denoise with N DDIM steps instead of the full DDPM chain (faster)")
    ap.add_argument("--start", help="start pose: YAML (traj_replay --save-start) or an R2R2R "
                    "episode (frame 0); default <ckpt-dir>/start_pose.yaml if it exists")
    ap.add_argument("--image-topic", help="default: the only sensor_msgs/Image topic")
    ap.add_argument("--mode", choices=["dry", "step", "auto"], default="step")
    ap.add_argument("--steps-per-move", type=int, default=8,
                    help="predicted waypoints executed per step")
    ap.add_argument("--max-steps", type=int, default=200, help="steps per episode")
    ap.add_argument("--settle", type=float, default=0.3,
                    help="s to wait after a move before taking the next frame")
    ap.add_argument("--gripper-threshold", type=float, default=0.5)
    ap.add_argument("--frame", default="world", help="frame of the policy's poses")
    ap.add_argument("--left-ee-link", default="openarm_left_hand_tcp")
    ap.add_argument("--right-ee-link", default="openarm_right_hand_tcp")
    ap.add_argument("--max-step", type=float, default=0.05, help="m, per waypoint")
    ap.add_argument("--workspace-min", type=float, nargs=3, default=[-0.3, -0.9, 0.6])
    ap.add_argument("--workspace-max", type=float, nargs=3, default=[1.0, 0.9, 1.9])
    ap.add_argument("--fake", action="store_true", help="scripted motions instead of the model")
    ap.add_argument("--fake-history", type=int, default=4)
    ap.add_argument("--fake-arms", nargs="+", choices=SIDES, default=["right"])
    args = ap.parse_args(argv)
    if not args.fake and not args.ckpt_dir:
        ap.error("--ckpt-dir is required (or --fake)")
    return args


def summarize(side: str, poses: np.ndarray, closed: np.ndarray, current: np.ndarray) -> str:
    first = np.linalg.norm(poses[0][:3, 3] - current[:3, 3]) * 1000
    length = np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)) * 100
    grip = "".join("C" if c else "o" for c in closed)
    return f"{side}: first waypoint {first:.1f} mm away, path {length:.1f} cm, gripper {grip}"


class PolicyRunner:
    def __init__(self, node: Node, args, policy, ex: StepExecutor, cam: Camera, start):
        self.node, self.args, self.policy, self.ex, self.cam = node, args, policy, ex, cam
        self.start = start
        self.log = node.get_logger()
        self.mode = args.mode
        self.history = History(policy.obs_horizon)
        self._warned_no_frames = False

    # ------------------------------------------------------------------ observations
    def observe(self, frame) -> RecordedObservation:
        tcp, vec = {}, []
        for s in SIDES:
            T = self.ex.ee_pose(s, frame.stamp)
            tcp[s] = T[:3, 3]
            vec.append(vec10_from_pose(T, 1.0 if frame.gripper[s] else 0.0))
        return RecordedObservation(Observation(frame.image, np.concatenate(vec)), tcp)

    def fresh(self) -> Observation:
        frame = self.cam.wait_frame(newer_than=self.node.get_clock().now(), timeout=5.0)
        if frame is None:
            raise RuntimeError("camera stopped publishing")
        return self.observe(frame).obs

    # ------------------------------------------------------------------ episode start
    def check_training_range(self):
        """Warn if the policy arm's TCP is outside the positions the policy was trained on."""
        if not hasattr(self.policy, "action_min"):
            return
        for s in self.policy.arms:
            c = COLS[s]
            lo, hi = self.policy.action_min[c][:3], self.policy.action_max[c][:3]
            p = self.ex.ee_pose(s)[:3, 3]
            off = float(np.linalg.norm(np.maximum(lo - p, 0) + np.maximum(p - hi, 0)))
            if off > 0.02:
                rng = ", ".join(f"{a} {lo[i]:.2f}..{hi[i]:.2f}" for i, a in enumerate("xyz"))
                self.log.warn(f"{s} TCP ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) is "
                              f"{off * 100:.0f} cm outside the positions in the training "
                              f"data ({rng})")

    def prepare_episode(self) -> bool:
        """Go to the start pose (confirmed), then confirm the policy start. False = quit."""
        if self.start is not None:
            while True:
                a = ask(f"[Enter] go to the start pose ({self.start.describe()}), "
                        "s = stay here, q = quit: ")
                if a == "q":
                    return False
                if a == "s":
                    break
                res = self.ex.go_to_start(self.start, SIDES)
                self.log.info(f"start pose: {res}")
                if all(st == "succeeded" for st in res.values()):
                    break
                self.log.error("could not reach the start pose")
        else:
            self.log.warn("no start pose (--start or <ckpt-dir>/start_pose.yaml): the policy "
                          "starts from the current pose")
            if self.mode != "dry":
                if ask(f"[Enter] open the {'/'.join(self.policy.arms)} gripper(s), "
                       "q = quit: ") == "q":
                    return False
                self.log.info(f"grippers: "
                              f"{self.ex.set_grippers({s: False for s in self.policy.arms})}")
        self.check_training_range()
        if ask(f"[Enter] start the policy ({self.mode} mode), q = quit: ") == "q":
            return False
        self.history.reset(self.fresh())
        return True

    # ------------------------------------------------------------------ steps
    def step(self, i: int) -> str:
        """One step. Returns 'next', 'again' (re-predict), 'restart' or 'quit'."""
        args, ex, log = self.args, self.ex, self.log
        images, proprio = self.history.arrays()
        t0 = time.monotonic()
        out = self.policy.predict(images, proprio)
        dt = time.monotonic() - t0
        k = min(args.steps_per_move, len(out))
        chunk, problems, lines = {}, [], []
        for s in self.policy.arms:
            poses, grip = poses_from_vec10(out[:, COLS[s]])
            closed = grip > args.gripper_threshold
            ex.show(s, poses)
            chunk[s] = (poses[:k], closed[:k])
            problems += ex.check(s, poses[:k])
            lines.append(summarize(s, poses[:k], closed[:k], ex.ee_pose(s)))
        log.info(f"step {i}: inference {dt:.2f} s; " + "; ".join(lines))
        for p in problems:
            log.warn(p)

        if self.mode == "dry":
            a = ask("[Enter] observe + predict again, h = back to start, q = quit: ")
            if a in ("q", "h"):
                return {"q": "quit", "h": "restart"}[a]
            self.history.extend([self.fresh()])
            return "next"
        if problems and self.mode == "auto":
            log.warn("safety-limit violation: switching to step mode")
            self.mode = "step"
        if self.mode == "step":
            a = ask(f"[Enter] execute {k} waypoints, r = re-predict, a = auto, "
                    "h = back to start, q = quit: ")
            if a in ("q", "h", "r"):
                return {"q": "quit", "h": "restart", "r": "again"}[a]
            if a == "a":
                self.mode = "auto"

        record = self.policy.obs_horizon > 1 and k > 1
        if record:
            self.cam.start_recording()
        try:
            res = ex.execute(chunk)
        except KeyboardInterrupt:
            ex.stop(self.policy.arms)
            res = StepResult(False, "stopped (Ctrl+C)")
        frames = self.cam.stop_recording() if record else []
        msg = f"step {i}: executed {res.executed}/{k} waypoints: {res.message}"
        if res.ok:  # separate call sites: rclpy rejects changing severities per call site
            log.info(msg)
        else:
            log.warn(msg)
        time.sleep(args.settle)
        final = self.fresh()
        if res.executed > 0:
            if record and not frames and not self._warned_no_frames:
                log.warn("no camera frames during the move: history entries repeat the "
                         "frame after the move")
                self._warned_no_frames = True
            positions = {s: chunk[s][0][:res.executed, :3, 3] for s in self.policy.arms}
            self.history.extend(pick_history(frames, positions, final,
                                             self.policy.obs_horizon, self.observe))
        else:
            self.history.extend([final])
        if not res.ok:
            self.mode = "step"
        return "next"

    def episode(self) -> bool:
        """Run steps until --max-steps / 'h' (True = another episode) or 'q' (False)."""
        i = 0
        while i < self.args.max_steps:
            try:
                outcome = self.step(i)
            except KeyboardInterrupt:
                self.ex.stop(self.policy.arms)
                self.log.warn("stopped (Ctrl+C); back to step mode")
                self.mode = "step"
                continue
            if outcome == "quit":
                return False
            if outcome == "restart":
                return True
            if outcome == "next":
                i += 1
        self.log.info(f"episode reached --max-steps {self.args.max_steps}")
        return True

    def run(self) -> int:
        while self.prepare_episode():
            if not self.episode():
                break
        return 0


def run(node: Node, args) -> int:
    log = node.get_logger()
    if args.fake:
        policy = FakePolicy(args.fake_arms, args.fake_history)
    else:
        from openarm_policy.tinydp_policy import TinyDPPolicy
        log.info(f"loading checkpoint from {args.ckpt_dir} ...")
        policy = TinyDPPolicy(args.ckpt_dir, args.ckpt_id, args.weights, args.device,
                              args.ddim_steps)
    log.info(policy.describe())

    start_path = args.start
    if start_path is None and args.ckpt_dir:
        candidate = os.path.join(args.ckpt_dir, "start_pose.yaml")
        start_path = candidate if os.path.isfile(candidate) else None
    start = load_start(start_path) if start_path else None
    if start is not None:
        log.info(f"start pose: {start.describe()}")

    limits = SafetyLimits(tuple(args.workspace_min), tuple(args.workspace_max), args.max_step)
    ex = StepExecutor(node, args.frame,
                      {"left": args.left_ee_link, "right": args.right_ee_link}, limits)
    topic = args.image_topic or detect_image_topic(node)
    if topic is None:
        return 2
    cam = Camera(node, topic, policy.image_size, lambda: ex.gripper_closed)
    err = ex.wait_ready(SIDES)
    if err:
        log.error(err)
        return 1
    if cam.wait_frame(timeout=10.0) is None:
        log.error(f"no image on {topic}")
        return 1
    h, w, enc = cam.shape
    log.info(f"camera {topic}: {w}x{h} {enc}, stretched to {policy.image_size}x"
             f"{policy.image_size} (the sim render must have the same aspect ratio)")
    return PolicyRunner(node, args, policy, ex, cam, start).run()


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # Plain KeyboardInterrupt on Ctrl+C: rclpy's handler would shut the context down, and the
    # stop command could no longer be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("openarm_policy")
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
