"""Gravity compensation for the OpenArm arms.

Computes the joint torques that hold the arm against gravity, g(q), from the URDF (Pinocchio)
and the measured joint positions, and streams them to one effort ForwardCommandController per
arm (<side>_gravity_controller). The OpenArm driver adds that effort command to the motor's
MIT loop as feedforward torque, next to the position command from the trajectory controller:

    tau_motor = kp * (q_cmd - q) + kd * (dq_cmd - dq) + scale * g(q)

so kp only has to correct the residual error instead of carrying the arm's weight.

Safety: torques are clamped per joint (max_torque) and ramped in over ramp_time seconds; if
/joint_states goes stale the command fades to zero, and zeros are sent on shutdown.
"""

import signal
import time

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


class GravityComp(Node):
    def __init__(self):
        super().__init__("gravity_comp")
        p = self.declare_parameter
        p("arms", ["left", "right"])
        p("scale", 1.0)              # fraction of the model's gravity torque to apply
        # Per-joint clamp [Nm], joint1..7. Motor peaks: DM8009 54, DM4340 28, DM4310 10.
        p("max_torque", [20.0, 20.0, 10.0, 10.0, 3.0, 3.0, 3.0])
        p("ramp_time", 2.0)          # s, fade torques in after start
        p("stale_timeout", 0.2)      # s without /joint_states -> fade to zero
        p("report_period", 5.0)      # s between model-vs-measured log lines (0 = off)

        self.arms = list(self.get_parameter("arms").value)
        self.joints = {s: [f"openarm_{s}_joint{i}" for i in range(1, 8)] for s in self.arms}
        self.pubs = {s: self.create_publisher(Float64MultiArray, f"/{s}_gravity_controller/commands", 10)
                     for s in self.arms}
        self.model = self.data = None
        self.q = None
        self.last_js = None
        self.measured = {}
        self.t_start = None
        self.last_report = 0.0
        self.fade = 0.0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._on_description, latched)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 50)
        self.create_timer(0.05, self._watchdog)

    # ------------------------------------------------------------------ model
    def _on_description(self, msg: String):
        model = pin.buildModelFromXML(msg.data)  # fixed base at the URDF root ("world")
        missing = [j for js in self.joints.values() for j in js if not model.existJointName(j)]
        if missing:
            self.get_logger().error(f"URDF lacks joints {missing}; gravity compensation disabled")
            return
        self.model, self.data = model, model.createData()
        self.q = pin.neutral(model)
        self.idx = {s: [model.joints[model.getJointId(j)].idx_q for j in js]
                    for s, js in self.joints.items()}
        self.q_index = {model.names[i]: model.joints[i].idx_q for i in range(1, model.njoints)
                        if model.joints[i].nq == 1}
        total = sum(model.inertias[i].mass for i in range(1, model.njoints))
        self.get_logger().info(
            f"Model loaded: {model.nq} dof, {total:.2f} kg moving mass; "
            f"scale={self._p('scale')}, max_torque={self._p('max_torque')}")

    def _p(self, name):
        return self.get_parameter(name).value

    # ---------------------------------------------------------------- control
    def _on_joint_states(self, msg: JointState):
        if self.model is None:
            return
        for name, pos in zip(msg.name, msg.position):
            i = self.q_index.get(name)
            if i is not None and np.isfinite(pos):
                self.q[i] = pos
        for name, eff in zip(msg.name, msg.effort):
            self.measured[name] = eff
        now = time.monotonic()
        self.last_js = now
        if self.t_start is None:
            self.t_start = now
        ramp = self._p("ramp_time")
        self.fade = 1.0 if ramp <= 0 else min(1.0, (now - self.t_start) / ramp)

        g = pin.computeGeneralizedGravity(self.model, self.data, self.q)
        limit = np.asarray(self._p("max_torque"), dtype=float)
        gain = self._p("scale") * self.fade
        for side in self.arms:
            tau = np.clip(gain * g[self.idx[side]], -limit, limit)
            self.pubs[side].publish(Float64MultiArray(data=tau.tolist()))
        self._report(now, g)

    def _watchdog(self):
        """Fade out if joint states stop (driver stalled or stack shutting down)."""
        if self.last_js is None or time.monotonic() - self.last_js < self._p("stale_timeout"):
            return
        if self.fade > 0.0:
            self.get_logger().warn("/joint_states stale: sending zero gravity torque")
        self.fade = 0.0
        self.t_start = None  # ramp in again when joint states return
        self.publish_zero()

    def publish_zero(self):
        for side in self.arms:
            self.pubs[side].publish(Float64MultiArray(data=[0.0] * 7))

    def _report(self, now, g):
        period = self._p("report_period")
        if period <= 0 or now - self.last_report < period:
            return
        self.last_report = now
        for side in self.arms:
            model = " ".join(f"{g[i]:+5.1f}" for i in self.idx[side])
            meas = " ".join(f"{self.measured.get(j, float('nan')):+5.1f}" for j in self.joints[side])
            self.get_logger().info(f"[{side}] g(q) model [Nm]: {model} | measured effort: {meas}")


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Handle SIGINT/SIGTERM ourselves so the context is still valid for the final zero command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    node = GravityComp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        time.sleep(0.1)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
