"""Topic interface for end-effector pose control of both OpenArm arms.

Subscribes (geometry_msgs/PoseStamped, empty frame_id = planning_frame):
  /openarm/left/target_pose            plan (OMPL) + execute
  /openarm/right/target_pose
  /openarm/left/target_pose_cartesian  straight-line Cartesian move
  /openarm/right/target_pose_cartesian

Publishes:
  /openarm/{left,right}/ee_pose   geometry_msgs/PoseStamped, current EE pose (TF), 30 Hz
  /openarm/{left,right}/status    std_msgs/String: planning | executing | succeeded | failed: ...

Each arm keeps only the newest target: targets that arrive while planning are coalesced, and
a new trajectory replaces the one the controller is running.
"""

import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from openarm_ee_control.commander import ArmCommander, ArmConfig, PlanOptions


class ArmWorker:
    def __init__(self, node: "EEPoseServer", commander: ArmCommander):
        self.node, self.cmd = node, commander
        name = commander.arm.name
        self.status_pub = node.create_publisher(String, f"/openarm/{name}/status", 10)
        self.pose_pub = node.create_publisher(PoseStamped, f"/openarm/{name}/ee_pose", 10)
        cbg = node.cbg
        node.create_subscription(PoseStamped, f"/openarm/{name}/target_pose",
                                 lambda m: self.submit(m, False), 10, callback_group=cbg)
        node.create_subscription(PoseStamped, f"/openarm/{name}/target_pose_cartesian",
                                 lambda m: self.submit(m, True), 10, callback_group=cbg)
        self._pending = None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, msg: PoseStamped, cartesian: bool):
        with self._cv:
            self._pending = (msg, cartesian)
            self._cv.notify()

    def _status(self, text: str):
        self.status_pub.publish(String(data=text))
        log = self.node.get_logger()
        (log.warn if text.startswith("failed") else log.info)(f"[{self.cmd.arm.name}] {text}")

    def _run(self):
        while rclpy.ok():
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None)
                target, cartesian = self._pending
                self._pending = None
            opt = self.node.plan_options(cartesian)
            self._status("planning")
            traj, msg = self.cmd.plan(target, opt)
            if traj is None:
                self._status(f"failed: {msg}")
                continue
            self._status("executing")
            res = self.cmd.execute(traj, wait=True)
            # A newer target preempts the controller goal; don't report that as a failure.
            if not res.ok and self._pending is not None:
                continue
            self._status("succeeded" if res.ok else f"failed: {res.message}")

    def publish_pose(self):
        try:
            self.pose_pub.publish(self.cmd.current_pose(timeout=0.0))
        except Exception:  # TF not ready yet
            pass


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
        for side in ("left", "right"):
            arm = ArmConfig(side, self._param(f"{side}_group"), self._param(f"{side}_ee_link"),
                            self._param(f"{side}_controller"))
            cmd = ArmCommander(self, arm, frame, tf_buffer=self.tf)
            self.workers.append(ArmWorker(self, cmd))
        self.create_timer(1.0 / self._param("pose_publish_rate"), self._publish_poses,
                          callback_group=self.cbg)
        self.get_logger().info(
            "Ready: publish geometry_msgs/PoseStamped to /openarm/{left,right}/target_pose"
            " (or .../target_pose_cartesian)")

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
