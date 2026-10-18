"""Plan and execute end-effector pose goals for one OpenArm arm through MoveIt 2.

Planning goes through move_group's services (/plan_kinematic_path, /compute_cartesian_path)
and execution goes straight to the arm's JointTrajectoryController. Each arm therefore runs
independently: both arms can move at once, and a new goal replaces the running trajectory.
"""

import threading
from dataclasses import dataclass

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint, MotionPlanRequest,
                             MoveItErrorCodes, OrientationConstraint, PositionConstraint,
                             RobotState)
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ERROR_NAMES = {v: k for k, v in MoveItErrorCodes.__dict__.items()
               if k.isupper() and isinstance(v, int)}


@dataclass
class ArmConfig:
    name: str            # "left" / "right"
    group: str           # MoveIt planning group
    ee_link: str         # link the pose goal refers to
    controller: str      # JointTrajectoryController name


@dataclass
class PlanOptions:
    cartesian: bool = False
    planner_pipeline: str = "ompl"
    planner_id: str = ""
    planning_time: float = 5.0
    attempts: int = 5
    velocity_scaling: float = 0.2
    acceleration_scaling: float = 0.2
    position_tolerance: float = 0.002     # m
    orientation_tolerance: float = 0.01   # rad
    cartesian_step: float = 0.005         # m between Cartesian waypoints
    cartesian_min_fraction: float = 0.98


class Result:
    def __init__(self, ok: bool, message: str):
        self.ok, self.message = ok, message

    def __repr__(self):
        return f"{'OK' if self.ok else 'FAILED'}: {self.message}"


def _wait(future, timeout: float):
    """Block on an rclpy future from a non-executor thread."""
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        return None
    return future.result()


class ArmCommander:
    """Pose-goal interface for one arm. Call methods from a thread other than the executor."""

    def __init__(self, node: Node, arm: ArmConfig, planning_frame: str = "world",
                 tf_buffer: Buffer | None = None):
        self.node, self.arm, self.planning_frame = node, arm, planning_frame
        self.plan_srv = node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.cart_srv = node.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.fjt = ActionClient(node, FollowJointTrajectory,
                                f"/{arm.controller}/follow_joint_trajectory")
        if tf_buffer is None:
            tf_buffer = Buffer()
            self._tf_listener = TransformListener(tf_buffer, node, spin_thread=False)
        self.tf = tf_buffer
        self._goal_lock = threading.Lock()
        self._active_goal = None

    # ------------------------------------------------------------------ queries
    def wait_for_servers(self, timeout: float = 30.0) -> bool:
        return (self.plan_srv.wait_for_service(timeout)
                and self.cart_srv.wait_for_service(timeout)
                and self.fjt.wait_for_server(timeout))

    def current_pose(self, frame: str | None = None, timeout: float = 2.0) -> PoseStamped:
        frame = frame or self.planning_frame
        t = self.tf.lookup_transform(frame, self.arm.ee_link, Time(),
                                     timeout=Duration(seconds=timeout))
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = t.header.stamp
        ps.pose.position.x = t.transform.translation.x
        ps.pose.position.y = t.transform.translation.y
        ps.pose.position.z = t.transform.translation.z
        ps.pose.orientation = t.transform.rotation
        return ps

    # ----------------------------------------------------------------- planning
    def _pose_constraints(self, target: PoseStamped, opt: PlanOptions) -> Constraints:
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE,
                                dimensions=[opt.position_tolerance])
        pc = PositionConstraint(header=target.header, link_name=self.arm.ee_link,
                                constraint_region=BoundingVolume(
                                    primitives=[sphere],
                                    primitive_poses=[Pose(position=target.pose.position)]),
                                weight=1.0)
        oc = OrientationConstraint(header=target.header, link_name=self.arm.ee_link,
                                   orientation=target.pose.orientation,
                                   absolute_x_axis_tolerance=opt.orientation_tolerance,
                                   absolute_y_axis_tolerance=opt.orientation_tolerance,
                                   absolute_z_axis_tolerance=opt.orientation_tolerance,
                                   weight=1.0)
        return Constraints(name="ee_pose", position_constraints=[pc],
                           orientation_constraints=[oc])

    def plan(self, target: PoseStamped, opt: PlanOptions):
        """Return (JointTrajectory | None, message)."""
        if not target.header.frame_id:
            target.header.frame_id = self.planning_frame
        if opt.cartesian:
            return self.plan_path(target.header, [target.pose], opt)
        return self._motion_plan(self._pose_constraints(target, opt), opt)

    def plan_joints(self, names: list[str], positions: list[float], opt: PlanOptions):
        """Free-space (OMPL) plan to joint positions of this arm's group."""
        goal = Constraints(name="joints", joint_constraints=[
            JointConstraint(joint_name=n, position=float(p), tolerance_above=1e-3,
                            tolerance_below=1e-3, weight=1.0)
            for n, p in zip(names, positions)])
        return self._motion_plan(goal, opt)

    def _motion_plan(self, goal: Constraints, opt: PlanOptions):
        mpr = MotionPlanRequest()
        mpr.group_name = self.arm.group
        mpr.start_state = RobotState(is_diff=True)  # = current state from the planning scene monitor
        mpr.goal_constraints = [goal]
        mpr.pipeline_id = opt.planner_pipeline
        mpr.planner_id = opt.planner_id
        mpr.num_planning_attempts = opt.attempts
        mpr.allowed_planning_time = opt.planning_time
        mpr.max_velocity_scaling_factor = opt.velocity_scaling
        mpr.max_acceleration_scaling_factor = opt.acceleration_scaling
        res = _wait(self.plan_srv.call_async(GetMotionPlan.Request(motion_plan_request=mpr)),
                    opt.planning_time + 30.0)
        if res is None:
            return None, "plan_kinematic_path timed out"
        r = res.motion_plan_response
        if r.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"plan: {ERROR_NAMES.get(r.error_code.val, r.error_code.val)}"
        return r.trajectory.joint_trajectory, f"planned in {r.planning_time:.2f}s"

    def plan_path(self, header: Header, poses: list[Pose], opt: PlanOptions):
        """Straight-line Cartesian path from the current state through all `poses`.

        IK is seeded from the current joint state and each previous waypoint, so the arm
        stays in its current configuration branch. Returns (JointTrajectory | None, message).
        """
        if not header.frame_id:
            header.frame_id = self.planning_frame
        req = GetCartesianPath.Request()
        req.header = header
        req.start_state = RobotState(is_diff=True)
        req.group_name = self.arm.group
        req.link_name = self.arm.ee_link
        req.waypoints = list(poses)
        req.max_step = opt.cartesian_step
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = opt.velocity_scaling
        req.max_acceleration_scaling_factor = opt.acceleration_scaling
        res = _wait(self.cart_srv.call_async(req), 30.0)
        if res is None:
            return None, "compute_cartesian_path timed out"
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"cartesian: {ERROR_NAMES.get(res.error_code.val, res.error_code.val)}"
        if res.fraction < opt.cartesian_min_fraction:
            return None, f"cartesian path only {res.fraction:.0%} feasible"
        return res.solution.joint_trajectory, f"cartesian ({res.fraction:.0%})"

    # ---------------------------------------------------------------- execution
    def execute(self, traj, wait: bool = True, timeout: float = 120.0) -> Result:
        if not traj.points:
            return Result(True, "already at goal")
        traj.header.stamp = Time().to_msg()  # start immediately
        goal = FollowJointTrajectory.Goal(trajectory=traj)
        gh = _wait(self.fjt.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            return Result(False, f"{self.arm.controller} rejected the trajectory")
        with self._goal_lock:
            self._active_goal = gh
        if not wait:
            return Result(True, "executing")
        res = _wait(gh.get_result_async(), timeout)
        if res is None:
            return Result(False, "execution timed out")
        code = res.result.error_code
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            return Result(True, "reached goal")
        return Result(False, f"controller error {code}: {res.result.error_string}")

    def stop(self):
        with self._goal_lock:
            gh, self._active_goal = self._active_goal, None
        if gh is not None:
            gh.cancel_goal_async()

    def move_to_pose(self, target: PoseStamped, opt: PlanOptions | None = None,
                     wait: bool = True) -> Result:
        opt = opt or PlanOptions()
        traj, msg = self.plan(target, opt)
        if traj is None:
            return Result(False, msg)
        res = self.execute(traj, wait=wait)
        return Result(res.ok, f"{msg}; {res.message}")


class GripperCommander:
    """Moves one gripper (a JointTrajectoryController on the finger joint) to a position."""

    def __init__(self, node: Node, controller: str, joint: str):
        self.joint = joint
        self.fjt = ActionClient(node, FollowJointTrajectory,
                                f"/{controller}/follow_joint_trajectory")

    def move(self, position: float, duration: float = 1.0, timeout: float = 10.0) -> Result:
        if not self.fjt.wait_for_server(timeout):
            return Result(False, "gripper controller not available")
        point = JointTrajectoryPoint(positions=[position],
                                     time_from_start=Duration(seconds=duration).to_msg())
        traj = JointTrajectory(joint_names=[self.joint], points=[point])
        goal = FollowJointTrajectory.Goal(trajectory=traj)
        gh = _wait(self.fjt.send_goal_async(goal), timeout)
        if gh is None or not gh.accepted:
            return Result(False, "gripper controller rejected the goal")
        res = _wait(gh.get_result_async(), duration + timeout)
        if res is None:
            return Result(False, "gripper timed out")
        code = res.result.error_code
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            return Result(True, f"gripper at {position:.3f}")
        return Result(False, f"gripper controller error {code}: {res.result.error_string}")
