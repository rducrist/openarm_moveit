"""Add the table to MoveIt's planning scene as a collision box, so every plan avoids it.

The box spans x_min..x_max and y_min..y_max in `frame`, from z_min (the floor) up to z_max
(the table top). Defaults: the cup_table sim (dataset_info.table in the policy bundle: top at
z 1.09 m, centre (0.466, -0.007), radius 0.30 m), with the edge towards the robot at
x = 0.18 m. All motions of ee_pose_server are planned by MoveIt (OMPL, Cartesian paths with
collision checking), so no arm link can be planned into the box; the gripper fingers count
too, so waypoints with the TCP at or below the top are rejected.

Started by the launch files (table:=true). Run it by hand to change or remove the box while
everything is running (same id: the box is replaced):
  ros2 run openarm_ee_control table_collision --ros-args -p z_max:=1.10
  ros2 run openarm_ee_control table_collision --ros-args -p remove:=true
"""

import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


def main(args=None):
    rclpy.init(args=args)
    node = Node("table_collision")
    p = node.declare_parameter
    frame = p("frame", "world").value
    object_id = p("object_id", "table").value
    x_min, x_max = p("x_min", 0.18).value, p("x_max", 0.765).value
    y_min, y_max = p("y_min", -0.306).value, p("y_max", 0.292).value
    z_min, z_max = p("z_min", 0.0).value, p("z_max", 1.09).value
    remove = p("remove", False).value
    timeout = p("timeout", 120.0).value
    log = node.get_logger()

    obj = CollisionObject(id=object_id)
    obj.header.frame_id = frame
    if remove:
        obj.operation = CollisionObject.REMOVE
        what = f"removing {object_id!r}"
    else:
        if not (x_max > x_min and y_max > y_min and z_max > z_min):
            log.error("need x_max > x_min, y_max > y_min, z_max > z_min")
            sys.exit(2)
        obj.operation = CollisionObject.ADD
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                         dimensions=[x_max - x_min, y_max - y_min, z_max - z_min])]
        center = Pose()
        center.position.x, center.position.y, center.position.z = (
            (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2)
        center.orientation.w = 1.0
        obj.primitive_poses = [center]
        what = (f"{object_id!r} box x {x_min:.3f}..{x_max:.3f}, y {y_min:.3f}..{y_max:.3f}, "
                f"z {z_min:.3f}..{z_max:.3f} m in {frame}")

    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not client.wait_for_service(timeout_sec=timeout):
        log.error(f"/apply_planning_scene (move_group) not available after {timeout:.0f} s")
        sys.exit(1)
    scene = PlanningScene(is_diff=True, world=PlanningSceneWorld(collision_objects=[obj]))
    future = client.call_async(ApplyPlanningScene.Request(scene=scene))
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    ok = future.done() and future.result().success
    if ok:
        log.info(f"planning scene: {what}")
    else:
        log.error(f"planning scene update failed: {what}")
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
