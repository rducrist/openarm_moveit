"""Client of ee_pose_server that executes pose sequences step by step (policy chunks, replays).

One step is a short list of TCP waypoints per arm, plus a gripper command per waypoint. The
executor checks the waypoints against SafetyLimits, shows them in RViz, sends them to
ee_pose_server as one Cartesian path per arm, switches grippers where the command changes,
and blocks until every commanded arm reports a terminal status.

Call the blocking methods from a thread other than the one spinning `node`.

RViz topics (nav_msgs/Path, frame = `frame`):
  /openarm/<arm>/planned_path     waypoints of the next step / predicted chunk
  /openarm/<arm>/reference_path   whole reference trajectory (replay)
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, Header, String
from tf2_ros import Buffer, TransformListener

from openarm_ee_control.geometry import (matrix_from_transform, matrix_from_xyz_wxyz,
                                         pose_from_matrix, rotation_angle)

SIDES = ("left", "right")
TERMINAL = ("succeeded", "failed", "stopped")


@dataclass
class SafetyLimits:
    """Checked before every step. Positions in the executor's frame."""
    workspace_min: tuple = (-0.3, -0.9, 0.6)   # m; set z to the table height + margin
    workspace_max: tuple = (1.0, 0.9, 1.9)
    max_step: float = 0.05       # m between consecutive waypoints, incl. current TCP -> first
    max_rot_step: float = 0.6    # rad, same


@dataclass
class StepResult:
    ok: bool
    message: str
    executed: int = 0            # waypoints reached (same index for all commanded arms)
    statuses: dict = field(default_factory=dict)


class _StatusWatch:
    """Latest status string of one topic, collected only after arm() was called."""

    def __init__(self):
        self._cv = threading.Condition()
        self._latest = None
        self._armed = False

    def arm(self):
        with self._cv:
            self._latest, self._armed = None, True

    def on_msg(self, msg: String):
        with self._cv:
            if self._armed:
                self._latest = msg.data
                self._cv.notify_all()

    def wait(self, timeout: float) -> str | None:
        with self._cv:
            done = self._cv.wait_for(
                lambda: self._latest is not None and self._latest.split(":")[0] in TERMINAL,
                timeout)
            self._armed = False
            return self._latest if done else None


class StepExecutor:
    def __init__(self, node: Node, frame: str = "world", ee_links: dict | None = None,
                 limits: SafetyLimits | None = None, timeout: float = 60.0):
        self.node, self.frame = node, frame
        self.ee_links = {s: f"openarm_{s}_hand_tcp" for s in SIDES}
        self.ee_links.update(ee_links or {})
        self.limits = limits or SafetyLimits()
        self.timeout = timeout
        # Long enough to look up poses at the stamps of all frames recorded during a move.
        self.tf = Buffer(cache_time=Duration(seconds=30.0))
        self._tfl = TransformListener(self.tf, node, spin_thread=False)
        # Commanded gripper state per arm (True = closed). R2R2R logs the gripper *command*,
        # so this is also what the policy gets as gripper proprio.
        self.gripper_closed = {s: False for s in SIDES}

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._path_pub, self._pose_pub, self._grip_pub, self._stop_pub = {}, {}, {}, {}
        self._joints_pub = {}
        self._viz_pub, self._ref_pub = {}, {}
        self._status, self._gstatus = {}, {}
        for s in SIDES:
            ns = f"/openarm/{s}"
            self._path_pub[s] = node.create_publisher(PoseArray, f"{ns}/target_path_cartesian", 10)
            self._pose_pub[s] = node.create_publisher(PoseStamped, f"{ns}/target_pose", 10)
            self._joints_pub[s] = node.create_publisher(JointState, f"{ns}/target_joints", 10)
            self._grip_pub[s] = node.create_publisher(Float64, f"{ns}/gripper", 10)
            self._stop_pub[s] = node.create_publisher(Empty, f"{ns}/stop", 10)
            self._viz_pub[s] = node.create_publisher(Path, f"{ns}/planned_path", latched)
            self._ref_pub[s] = node.create_publisher(Path, f"{ns}/reference_path", latched)
            self._status[s], self._gstatus[s] = _StatusWatch(), _StatusWatch()
            node.create_subscription(String, f"{ns}/status", self._status[s].on_msg, 10)
            node.create_subscription(String, f"{ns}/gripper_status", self._gstatus[s].on_msg, 10)

    # ------------------------------------------------------------------ state
    def wait_ready(self, sides=SIDES, timeout: float = 30.0) -> str | None:
        """Wait for TF of the TCP links and for ee_pose_server. Returns an error or None."""
        deadline = time.monotonic() + timeout
        missing = []
        while time.monotonic() < deadline:
            missing = [self.ee_links[s] for s in sides
                       if not self.tf.can_transform(self.frame, self.ee_links[s], Time())]
            missing += [f"ee_pose_server (/openarm/{s}/status)" for s in sides
                        if self.node.count_publishers(f"/openarm/{s}/status") == 0]
            if not missing:
                # Subscriptions need a moment to match before the first command is sent.
                time.sleep(0.5)
                return None
            time.sleep(0.2)
        return "not available: " + ", ".join(missing)

    def ee_pose(self, side: str, stamp: Time | None = None, timeout: float = 0.5) -> np.ndarray:
        """TCP pose of one arm in `frame` (4x4), at `stamp` if given (falls back to latest,
        also right away for stamps more than 2 s from now: replayed or mis-clocked images)."""
        link = self.ee_links[side]
        if stamp is not None and abs((self.node.get_clock().now() - stamp).nanoseconds) < 2e9:
            try:
                t = self.tf.lookup_transform(self.frame, link, stamp,
                                             timeout=Duration(seconds=timeout))
                return matrix_from_transform(t.transform)
            except Exception:  # stamp outside the TF buffer: use the latest pose instead
                pass
        t = self.tf.lookup_transform(self.frame, link, Time(), timeout=Duration(seconds=timeout))
        return matrix_from_transform(t.transform)

    # ------------------------------------------------------------------ checks / display
    def check(self, side: str, poses: np.ndarray, start: np.ndarray | None = None) -> list[str]:
        """Safety-limit violations of waypoints `poses` (N, 4, 4), starting at `start`."""
        lim = self.limits
        lo, hi = np.asarray(lim.workspace_min), np.asarray(lim.workspace_max)
        prev = self.ee_pose(side) if start is None else start
        problems = []
        for i, T in enumerate(poses):
            p = T[:3, 3]
            if np.any(p < lo) or np.any(p > hi):
                problems.append(f"{side} waypoint {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
                                f"outside workspace {tuple(lo)}..{tuple(hi)}")
            d = float(np.linalg.norm(p - prev[:3, 3]))
            a = rotation_angle(prev[:3, :3], T[:3, :3])
            if d > lim.max_step:
                problems.append(f"{side} waypoint {i}: step {d * 100:.1f} cm > "
                                f"{lim.max_step * 100:.1f} cm")
            if a > lim.max_rot_step:
                problems.append(f"{side} waypoint {i}: rotation step {np.degrees(a):.0f} deg > "
                                f"{np.degrees(lim.max_rot_step):.0f} deg")
            prev = T
        return problems

    def _path_msg(self, poses: np.ndarray) -> Path:
        header = Header(frame_id=self.frame, stamp=self.node.get_clock().now().to_msg())
        return Path(header=header,
                    poses=[PoseStamped(header=header, pose=pose_from_matrix(T)) for T in poses])

    def show(self, side: str, poses: np.ndarray):
        self._viz_pub[side].publish(self._path_msg(poses))

    def show_reference(self, side: str, poses: np.ndarray):
        self._ref_pub[side].publish(self._path_msg(poses))

    # ------------------------------------------------------------------ commands
    def _wait_all(self, watches: dict, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        out = {}
        for s, watch in watches.items():
            st = watch.wait(max(0.0, deadline - time.monotonic()))
            out[s] = st if st is not None else "failed: no status from ee_pose_server (timeout)"
        return out

    def _stationary(self, side: str, poses: np.ndarray, tol=(0.001, np.radians(0.5))) -> bool:
        current = self.ee_pose(side)
        return all(np.linalg.norm(T[:3, 3] - current[:3, 3]) < tol[0]
                   and rotation_angle(T[:3, :3], current[:3, :3]) < tol[1] for T in poses)

    def move(self, waypoints: dict) -> dict:
        """Cartesian path through waypoints {side: (N, 4, 4)}; all arms at once. Blocks.

        Arms whose waypoints all lie within 1 mm / 0.5 deg of their current TCP pose are not
        commanded (reported as succeeded)."""
        sides = [s for s, w in waypoints.items() if len(w) and not self._stationary(s, w)]
        skipped = {s: "succeeded" for s, w in waypoints.items() if s not in sides}
        if not sides:
            return skipped
        header = Header(frame_id=self.frame, stamp=self.node.get_clock().now().to_msg())
        for s in sides:
            self._status[s].arm()
        for s in sides:
            self._path_pub[s].publish(
                PoseArray(header=header, poses=[pose_from_matrix(T) for T in waypoints[s]]))
        return {**skipped, **self._wait_all({s: self._status[s] for s in sides}, self.timeout)}

    def approach(self, side: str, pose: np.ndarray) -> str:
        """Free-space (OMPL) move to one pose, e.g. the first pose of a trajectory. Blocks."""
        self._status[side].arm()
        header = Header(frame_id=self.frame, stamp=self.node.get_clock().now().to_msg())
        self._pose_pub[side].publish(PoseStamped(header=header, pose=pose_from_matrix(pose)))
        return self._wait_all({side: self._status[side]}, self.timeout)[side]

    def move_joints(self, side: str, joints: dict) -> str:
        """Free-space (OMPL) move of one arm to joint positions {name: rad}. Blocks."""
        self._status[side].arm()
        msg = JointState(name=list(joints), position=[float(v) for v in joints.values()])
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self._joints_pub[side].publish(msg)
        return self._wait_all({side: self._status[side]}, self.timeout)[side]

    def go_to_start(self, start, sides=SIDES, grippers_open: bool = True) -> dict:
        """Move to a StartPose (r2r2r_data): joint goal where the start pose has joint values,
        else an OMPL pose goal for the TCP. One arm after the other, so each plan is
        collision-checked against the other arm where it actually is. Stops at the first
        failure. Returns {side: status}."""
        results = {}
        if grippers_open:
            results.update({f"{s} gripper": st for s, st in
                            self.set_grippers({s: False for s in sides}).items()})
        for s in sides:
            joints = start.arm_joints(s)
            if joints:
                results[s] = self.move_joints(s, joints)
            elif s in start.tcp:
                results[s] = self.approach(s, matrix_from_xyz_wxyz(np.asarray(start.tcp[s])))
            else:
                continue
            if results[s] != "succeeded":
                break
        return results

    def set_grippers(self, closed: dict) -> dict:
        """Open/close grippers {side: closed}. Blocks until they report back."""
        for s, c in closed.items():
            self._gstatus[s].arm()
            self._grip_pub[s].publish(Float64(data=1.0 if c else 0.0))
            self.gripper_closed[s] = bool(c)
        return self._wait_all({s: self._gstatus[s] for s in closed}, self.timeout)

    def stop(self, sides=SIDES):
        for s in sides:
            self._stop_pub[s].publish(Empty())

    def execute(self, chunk: dict) -> StepResult:
        """Execute {side: (poses (N, 4, 4), closed (N,) bool)}; N must match across arms.

        The arms move through the waypoints as one Cartesian path each. Where the gripper
        command of an arm switches at waypoint j, the motion stops at j, the gripper switches,
        and the rest follows as another path.
        """
        sides = list(chunk)
        n = len(chunk[sides[0]][0])
        if any(len(chunk[s][0]) != n or len(chunk[s][1]) != n for s in sides):
            return StepResult(False, "all arms need the same number of waypoints and gripper values")
        i = 0
        statuses = {}
        while i < n:
            j = i
            while j < n and all(bool(chunk[s][1][j]) == self.gripper_closed[s] for s in sides):
                j += 1
            end = min(j + 1, n)
            statuses = self.move({s: chunk[s][0][i:end] for s in sides})
            failed = {s: st for s, st in statuses.items() if st != "succeeded"}
            if failed:
                return StepResult(False, "; ".join(f"{s}: {st}" for s, st in failed.items()),
                                  executed=i, statuses=statuses)
            if j < n:
                switch = {s: bool(chunk[s][1][j]) for s in sides
                          if bool(chunk[s][1][j]) != self.gripper_closed[s]}
                gres = self.set_grippers(switch)
                failed = {s: st for s, st in gres.items() if st != "succeeded"}
                if failed:
                    return StepResult(False, "; ".join(f"{s} gripper: {st}"
                                                       for s, st in failed.items()),
                                      executed=end, statuses=statuses)
            i = end
        return StepResult(True, "ok", executed=n, statuses=statuses)


def ask(prompt: str) -> str:
    """input() that returns 'q' on EOF / Ctrl+C, lower-cased and stripped."""
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"
