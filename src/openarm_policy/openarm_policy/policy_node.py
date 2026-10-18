"""Run an R2R2R-trained diffusion policy on OpenArm, step by step, via ee_pose_server.

Policies (--policy):
  * OpenArm cup policy bundle (openarm_cup_policy/1): a folder with the .pt and its loader
    (openarm_policy.py, tinydp_vision.py), e.g. policies/2026-10-18_openarm_left_v2. It
    carries the camera calibration, image crops, timing and home pose; preprocessing expects
    the raw head-camera frames (1920x1080, distorted) of openarm_camera's /camera/image_raw
    (or .../compressed).
  * tinydp checkpoint folder: run.yaml, action_statistics.json, checkpoint_<N>.pt.

Episode:
  go to start pose   --start, else <policy dir>/start_pose.yaml, else the bundle's home joints
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

  pixi run -e policy policy -- --policy policies/2026-10-18_openarm_left_v2 \\
      --image-topic /camera/image_raw/compressed --mode dry
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
from sensor_msgs.msg import CameraInfo

from openarm_ee_control.r2r2r_data import load_start
from openarm_ee_control.step_executor import SIDES, SafetyLimits, StepExecutor, StepResult, ask
from openarm_policy.observation import (Camera, History, Observation, pick_history,
                                        resize_like_training)


class FakePolicy:
    """Scripted stand-in with the policy interface: 3 mm steps along a slowly turning
    direction in the y-z plane, gripper closed on every other pair of predictions."""

    frame_dt = action_dt = None
    raw_size = calibration = start_pose = None

    def __init__(self, arms, obs_horizon=4, action_horizon=16, image_size=224):
        self.arms, self.obs_horizon = arms, obs_horizon
        self.action_horizon, self.image_size = action_horizon, image_size
        self.calls = 0

    def describe(self) -> str:
        return (f"FAKE policy: arms={self.arms}, obs history {self.obs_horizon}, "
                f"chunk {self.action_horizon}")

    def preprocess(self, rgb):
        return resize_like_training(rgb, self.image_size)

    def predict(self, observations):
        phi = 2 * np.pi * self.calls / 12
        direction = np.array([0.0, np.cos(phi), np.sin(phi)])
        grip = float((self.calls // 2) % 2)
        self.calls += 1
        out = {}
        for s in self.arms:
            poses = np.repeat(observations[-1].tcp[s][None], self.action_horizon, axis=0)
            poses[:, :3, 3] += 0.003 * np.arange(1, self.action_horizon + 1)[:, None] * direction
            out[s] = (poses, np.full(self.action_horizon, grip))
        return out


def load_policy(args):
    if args.fake:
        return FakePolicy(args.fake_arms, args.fake_history)
    from openarm_policy.cup_bundle_policy import CupBundlePolicy, find_bundle
    bundle = find_bundle(args.policy)
    if bundle:
        return CupBundlePolicy(bundle, args.device, args.ddim_steps)
    from openarm_policy.tinydp_policy import TinyDPPolicy
    folder, ckpt_id = args.policy, args.ckpt_id
    if os.path.isfile(folder):  # .../checkpoint_<N>.pt
        folder, name = os.path.split(os.path.abspath(folder))
        ckpt_id = int(name.removeprefix("checkpoint_").removesuffix(".pt"))
    if not os.path.isfile(os.path.join(folder, "run.yaml")):
        raise FileNotFoundError(f"{args.policy}: neither a policy bundle (.pt next to "
                                "openarm_policy.py) nor a tinydp folder (run.yaml)")
    return TinyDPPolicy(folder, ckpt_id, args.weights, args.device, args.ddim_steps or 0)


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


def camera_info_topic(image_topic: str) -> str:
    base = image_topic.removesuffix("/compressed").rstrip("/")
    return base.rsplit("/", 1)[0] + "/camera_info"


def check_calibration(node: Node, topic: str, calibration, timeout: float = 3.0) -> str:
    """Compare the camera's CameraInfo with the policy's calibration. Returns a log line."""
    got = []
    sub = node.create_subscription(CameraInfo, topic, got.append, 10)
    deadline = time.monotonic() + timeout
    while not got and time.monotonic() < deadline:
        time.sleep(0.1)
    node.destroy_subscription(sub)
    if not got:
        return f"no {topic} (start openarm_camera with calibration_file:=...): calibration not compared"
    K, dist = calibration
    k_cam, d_cam = np.reshape(got[0].k, (3, 3)), np.asarray(got[0].d)
    n = min(len(d_cam), len(dist))
    if np.allclose(k_cam, K, rtol=1e-4, atol=1e-3) and np.allclose(d_cam[:n], dist[:n], atol=1e-6):
        return f"{topic} matches the policy's calibration"
    return (f"WARNING {topic} differs from the policy's calibration: camera K diag "
            f"{np.diag(k_cam).round(1).tolist()} dist {d_cam[:n].round(4).tolist()}, policy K diag "
            f"{np.diag(K).round(1).tolist()} dist {np.asarray(dist)[:n].round(4).tolist()}")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", "--ckpt-dir", dest="policy",
                    help="policy bundle folder / .pt, or tinydp checkpoint folder")
    ap.add_argument("--ckpt-id", type=int, help="tinydp: checkpoint_<N>.pt (default: highest N)")
    ap.add_argument("--weights", choices=["model", "ema"], default="model",
                    help="tinydp: 'model' is what tinydp's DiffusionWrapper loads")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ddim-steps", type=int,
                    help="denoising: N DDIM steps, 0 = the full training DDPM chain "
                         "(default: bundle 10, tinydp 0)")
    ap.add_argument("--start", help="start pose: YAML (traj_replay --save-start) or an R2R2R "
                    "episode (frame 0); default <policy dir>/start_pose.yaml, else the bundle's "
                    "home joints")
    ap.add_argument("--image-topic", help="sensor_msgs/Image, or CompressedImage if it ends in "
                    "/compressed; default: the only Image topic")
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
    if not args.fake and not args.policy:
        ap.error("--policy is required (or --fake)")
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
    def observe(self, frame) -> Observation:
        return Observation(frame.image,
                           {s: self.ex.ee_pose(s, frame.stamp) for s in SIDES},
                           {s: 1.0 if frame.gripper[s] else 0.0 for s in SIDES})

    def fresh(self) -> Observation:
        frame = self.cam.wait_frame(newer_than=self.node.get_clock().now(), timeout=5.0)
        if frame is None:
            raise RuntimeError("camera stopped publishing")
        return self.observe(frame)

    # ------------------------------------------------------------------ episode start
    def check_training_range(self):
        """Warn if the policy arm's TCP is outside the positions the policy was trained on."""
        if not hasattr(self.policy, "train_box"):
            return
        for s in self.policy.arms:
            lo, hi = self.policy.train_box(s)
            p = self.ex.ee_pose(s)[:3, 3]
            off = float(np.linalg.norm(np.maximum(lo - p, 0) + np.maximum(p - hi, 0)))
            msg = (f"{s} TCP ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}); training data: "
                   + ", ".join(f"{a} {lo[i]:.2f}..{hi[i]:.2f}" for i, a in enumerate("xyz")))
            if off > 0.02:
                self.log.warn(f"{msg}: {off * 100:.0f} cm outside")
            else:
                self.log.info(f"{msg}: inside")

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
            self.log.warn("no start pose (--start, <policy dir>/start_pose.yaml or bundle home "
                          "joints): the policy starts from the current pose")
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
        t0 = time.monotonic()
        prediction = self.policy.predict(self.history.observations())
        dt = time.monotonic() - t0
        chunk, problems, lines = {}, [], []
        k = args.steps_per_move
        for s, (poses, grip) in prediction.items():
            k = min(k, len(poses))
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
            ex.stop(list(chunk))
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
            positions = {s: chunk[s][0][:res.executed, :3, 3] for s in chunk}
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
    if not args.fake:
        log.info(f"loading policy {args.policy} ...")
    policy = load_policy(args)
    log.info(policy.describe())
    if policy.frame_dt and policy.action_dt and abs(policy.frame_dt - policy.action_dt) > 1e-6:
        log.warn(f"observation step {policy.frame_dt:.4f} s != action step "
                 f"{policy.action_dt:.4f} s: history entries are taken per executed waypoint")

    policy_dir = None
    if args.policy:
        policy_dir = args.policy if os.path.isdir(args.policy) else os.path.dirname(args.policy)
    start_path = args.start
    if start_path is None and policy_dir:
        candidate = os.path.join(policy_dir, "start_pose.yaml")
        start_path = candidate if os.path.isfile(candidate) else None
    start = load_start(start_path) if start_path else policy.start_pose
    if start is not None:
        log.info(f"start pose: {start.describe()}")

    limits = SafetyLimits(tuple(args.workspace_min), tuple(args.workspace_max), args.max_step)
    ex = StepExecutor(node, args.frame,
                      {"left": args.left_ee_link, "right": args.right_ee_link}, limits)
    topic = args.image_topic or detect_image_topic(node)
    if topic is None:
        return 2
    cam = Camera(node, topic, policy.preprocess, lambda: ex.gripper_closed,
                 record_period=policy.frame_dt or 0.0)
    err = ex.wait_ready(SIDES)
    if err:
        log.error(err)
        return 1
    size = cam.raw_shape(timeout=10.0)
    if size is None:
        log.error(f"no image on {topic}")
        return 1
    if policy.raw_size and size != policy.raw_size:
        log.error(f"camera {topic} is {size[0]}x{size[1]}, the policy needs raw "
                  f"{policy.raw_size[0]}x{policy.raw_size[1]} frames (openarm_camera width:= "
                  "height:=, no resizing or undistortion before the policy)")
        return 1
    log.info(f"camera {topic}: {size[0]}x{size[1]} -> model input {policy.image_size}x"
             f"{policy.image_size}")
    if policy.calibration is not None:
        line = check_calibration(node, camera_info_topic(topic), policy.calibration)
        if line.startswith("WARNING"):
            log.warn(line)
        else:
            log.info(line)
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
