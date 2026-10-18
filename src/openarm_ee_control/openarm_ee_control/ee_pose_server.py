"""Topic interface for end-effector pose control of both OpenArm arms.

Subscribes (empty frame_id = planning_frame):
  /openarm/{left,right}/target_pose            geometry_msgs/PoseStamped  plan (OMPL) + execute
  /openarm/{left,right}/target_pose_cartesian  geometry_msgs/PoseStamped  straight-line move
  /openarm/{left,right}/target_path_cartesian  geometry_msgs/PoseArray    straight-line moves
                                               through all poses, executed as one trajectory
  /openarm/{left,right}/target_joints          sensor_msgs/JointState     plan (OMPL) + execute
                                               to joint positions (name + position)
  /openarm/{left,right}/gripper                std_msgs/Float64  0 = open .. 1 = closed
  /openarm/{left,right}/stop                   std_msgs/Empty    cancel the running motion

Publishes:
  /openarm/{left,right}/ee_pose         geometry_msgs/PoseStamped, current EE pose (TF), 30 Hz
  /openarm/{left,right}/status          std_msgs/String: planning | executing | succeeded |
                                        stopped | failed: ...
  /openarm/{left,right}/gripper_status  std_msgs/String: executing | succeeded | failed: ...

Each arm keeps only the newest target: targets that arrive while planning are coalesced, and
a new trajectory replaces the one the controller is running. The gripper is independent of
the arm and also keeps only its newest command.
"""

import threading

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, String
from tf2_ros import Buffer, TransformListener

from openarm_ee_control.commander import ArmCommander, ArmConfig, GripperCommander, PlanOptions


def _log_status(node: Node, text: str, failed: bool):
    # Separate call sites: rclpy raises if one call site logs at changing severities.
    if failed:
        node.get_logger().warn(text)
    else:
        node.get_logger().info(text)


class ArmWorker:
    def __init__(self, node: "EEPoseServer", commander: ArmCommander):
        self.node, self.cmd = node, commander
        name = commander.arm.name
        self.status_pub = node.create_publisher(String, f"/openarm/{name}/status", 10)
        self.pose_pub = node.create_publisher(PoseStamped, f"/openarm/{name}/ee_pose", 10)
        cbg = node.cbg
        node.create_subscription(PoseStamped, f"/openarm/{name}/target_pose",
                                 lambda m: self.submit(m, "ompl"), 10, callback_group=cbg)
        node.create_subscription(PoseStamped, f"/openarm/{name}/target_pose_cartesian",
                                 lambda m: self.submit(m, "cartesian"), 10, callback_group=cbg)
        node.create_subscription(PoseArray, f"/openarm/{name}/target_path_cartesian",
                                 lambda m: self.submit(m, "path"), 10, callback_group=cbg)
        node.create_subscription(JointState, f"/openarm/{name}/target_joints",
                                 lambda m: self.submit(m, "joints"), 10, callback_group=cbg)
        node.create_subscription(Empty, f"/openarm/{name}/stop",
                                 lambda _: self.stop(), 10, callback_group=cbg)
        self._pending = None
        self._stopped = False
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, msg, mode: str):
        with self._cv:
            self._pending = (msg, mode)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._pending = None
            self._stopped = True
        self.cmd.stop()

    def _status(self, text: str):
        self.status_pub.publish(String(data=text))
        _log_status(self.node, f"[{self.cmd.arm.name}] {text}", text.startswith("failed"))

    def _plan(self, target, mode: str):
        opt = self.node.plan_options(cartesian=mode in ("cartesian", "path"))
        if mode == "path":
            if not target.poses:
                return None, "empty path"
            return self.cmd.plan_path(target.header, target.poses, opt)
        if mode == "joints":
            if not target.name or len(target.name) != len(target.position):
                return None, "joint goal needs matching name and position lists"
            return self.cmd.plan_joints(list(target.name), list(target.position), opt)
        return self.cmd.plan(target, opt)

    def _run(self):
        while rclpy.ok():
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None)
                target, mode = self._pending
                self._pending = None
                self._stopped = False
            self._status("planning")
            traj, msg = self._plan(target, mode)
            if traj is None:
                self._status(f"failed: {msg}")
                continue
            if self._stopped:
                self._status("stopped")
                continue
            self._status("executing")
            res = self.cmd.execute(traj, wait=True)
            # A newer target preempts the controller goal; don't report that as a failure.
            if not res.ok and self._pending is not None:
                continue
            if self._stopped:
                self._status("stopped")
                continue
            self._status("succeeded" if res.ok else f"failed: {res.message}")

    def publish_pose(self):
        try:
            self.pose_pub.publish(self.cmd.current_pose(timeout=0.0))
        except Exception:  # TF not ready yet
            pass


class GripperWorker:
    def __init__(self, node: "EEPoseServer", name: str, commander: GripperCommander):
        self.node, self.name, self.cmd = node, name, commander
        self.status_pub = node.create_publisher(String, f"/openarm/{name}/gripper_status", 10)
        node.create_subscription(Float64, f"/openarm/{name}/gripper", self.submit, 10,
                                 callback_group=node.cbg)
        self._pending = None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, msg: Float64):
        with self._cv:
            self._pending = min(max(msg.data, 0.0), 1.0)
            self._cv.notify()

    def _status(self, text: str):
        self.status_pub.publish(String(data=text))
        _log_status(self.node, f"[{self.name} gripper] {text}", text.startswith("failed"))

    def _run(self):
        while rclpy.ok():
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None)
                closure, self._pending = self._pending, None
            p = self.node._param
            position = p("gripper_open") + closure * (p("gripper_closed") - p("gripper_open"))
            self._status("executing")
            res = self.cmd.move(position, duration=p("gripper_duration"))
            self._status("succeeded" if res.ok else f"failed: {res.message}")


class EEPoseServer(Node):
    def __init__(self):
        super().__init__("ee_pose_server")
        p = self.declare_parameter
        p("planning_frame", "world")
        p("left_group", "left_arm")
        p("right_group", "right_arm")
        p("left_ee_link", "openarm_left_grasp_frame")
        p("right_ee_link", "openarm_right_grasp_frame")
        p("left_controller", "left_arm_controller")
        p("right_controller", "right_arm_controller")
        p("left_gripper_controller", "left_gripper_controller")
        p("right_gripper_controller", "right_gripper_controller")
        p("left_gripper_joint", "openarm_left_finger_joint1")
        p("right_gripper_joint", "openarm_right_finger_joint1")
        p("gripper_open", 0.044)    # finger joint position [m] for command 0
        p("gripper_closed", 0.0)    # finger joint position [m] for command 1
        p("gripper_duration", 1.0)  # s
        p("velocity_scaling", 0.2)
        p("acceleration_scaling", 0.2)
        p("planning_time", 5.0)
        p("position_tolerance", 0.002)
        p("orientation_tolerance", 0.01)
        p("pose_publish_rate", 30.0)

        self.cbg = ReentrantCallbackGroup()
        self.tf = Buffer()
        self._tfl = TransformListener(self.tf, self, spin_thread=False)
        frame = self.get_parameter("planning_frame").value

        self.workers = []
        self.grippers = []
        for side in ("left", "right"):
            arm = ArmConfig(side, self._param(f"{side}_group"), self._param(f"{side}_ee_link"),
                            self._param(f"{side}_controller"))
            cmd = ArmCommander(self, arm, frame, tf_buffer=self.tf)
            self.workers.append(ArmWorker(self, cmd))
            gripper = GripperCommander(self, self._param(f"{side}_gripper_controller"),
                                       self._param(f"{side}_gripper_joint"))
            self.grippers.append(GripperWorker(self, side, gripper))
        self.create_timer(1.0 / self._param("pose_publish_rate"), self._publish_poses,
                          callback_group=self.cbg)
        self.get_logger().info(
            "Ready: publish geometry_msgs/PoseStamped to /openarm/{left,right}/target_pose"
            " (or .../target_pose_cartesian, PoseArray on .../target_path_cartesian)")

    def _param(self, name):
        return self.get_parameter(name).value

    def plan_options(self, cartesian: bool) -> PlanOptions:
        return PlanOptions(cartesian=cartesian,
                           velocity_scaling=self._param("velocity_scaling"),
                           acceleration_scaling=self._param("acceleration_scaling"),
                           planning_time=self._param("planning_time"),
                           position_tolerance=self._param("position_tolerance"),
                           orientation_tolerance=self._param("orientation_tolerance"))

    def _publish_poses(self):
        for w in self.workers:
            w.publish_pose()


def main():
    rclpy.init()
    node = EEPoseServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
