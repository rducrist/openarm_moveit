"""Bring up OpenArm (bimanual) for end-effector pose control with MoveIt 2.

Starts:
  * robot_state_publisher + ros2_control (mock or real CAN hardware)
  * joint_state_broadcaster, per-arm JointTrajectoryControllers, gripper controllers
  * move_group (OMPL + Pilz) and RViz with the MotionPlanning panel
  * ee_pose_server: PoseStamped topics -> MoveIt plans -> execution
  * gravity_comp (gravity_comp:=true): URDF gravity torque as motor feedforward
  * table_collision (table:=true): the table as a collision box in the planning scene

Robot model:
  * default: upstream openarm_description v2.0 (or v1.0 with arm_type:=v1.0)
  * custom:  urdf:=/abs/path/robot.urdf (or .xacro). If the file has no <ros2_control>
             block, one is generated with the upstream OpenArm macro, so mock and real
             hardware both work as long as joint names follow openarm_{left,right}_jointN.
"""

import os
import re
import tempfile

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PKG = "openarm_ee_control"


def _arm_version(arm_type: str) -> str:
    return "v1.0" if any(x in arm_type for x in ("1.0", "10", "1_0")) else "v2.0"


def _xacro_mappings(ctx) -> dict:
    lc = lambda n: LaunchConfiguration(n).perform(ctx)  # noqa: E731
    return {
        "arm_type": lc("arm_type"),
        "bimanual": "true",
        "ros2_control": "true",
        "use_fake_hardware": lc("use_fake_hardware"),
        "left_can_interface": lc("left_can_interface"),
        "right_can_interface": lc("right_can_interface"),
        "can_fd": lc("can_fd"),
        "emit_grasp_frame": "true",  # v2.0: adds openarm_<side>_grasp_frame (TCP)
    }


def default_ee_link(side: str, version: str) -> str:
    return f"openarm_{side}_hand_tcp" if version == "v1.0" else f"openarm_{side}_grasp_frame"


def build_robot_description(ctx) -> str:
    """Return the URDF string, from the upstream xacro or the user's custom file."""
    custom = LaunchConfiguration("urdf").perform(ctx).strip()
    version = _arm_version(LaunchConfiguration("arm_type").perform(ctx))
    mappings = _xacro_mappings(ctx)

    if not custom:
        desc = get_package_share_directory("openarm_description")
        name = "openarm_v10.urdf.xacro" if version == "v1.0" else "openarm_v20.urdf.xacro"
        path = os.path.join(desc, "assets", "robot", f"openarm_{version}", "urdf", name)
        return xacro.process_file(path, mappings=mappings).toxml()

    custom = os.path.abspath(os.path.expanduser(custom))
    if not os.path.isfile(custom):
        raise FileNotFoundError(f"urdf:={custom} does not exist")

    if custom.endswith(".xacro"):
        urdf = xacro.process_file(custom, mappings=mappings).toxml()
    else:
        with open(custom, encoding="utf-8") as f:
            urdf = f.read()

    if re.search(r"<ros2_control\s+name=", urdf):
        return urdf

    # No ros2_control block: wrap the model and add the upstream OpenArm one.
    wrapper = os.path.join(get_package_share_directory(PKG), "urdf",
                           f"add_ros2_control_{version.replace('.', '')}.urdf.xacro")
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as tmp:
        tmp.write(urdf)
    return xacro.process_file(wrapper, mappings={**mappings, "model": tmp.name}).toxml()


def check_can_interfaces(ctx):
    """Refuse to start real hardware on a CAN interface that is missing or down.

    The OpenArm driver does not report failed CAN sends: on a down interface it logs a
    successful activation while no frame reaches the motors and joint states stay at zero.
    """
    problems = []
    for arg in ("right_can_interface", "left_can_interface"):
        iface = LaunchConfiguration(arg).perform(ctx)
        try:
            with open(f"/sys/class/net/{iface}/flags", encoding="ascii") as f:
                up = int(f.read().strip(), 16) & 0x1  # IFF_UP
        except FileNotFoundError:
            problems.append(f"{iface} ({arg}) does not exist: is the CAN adapter plugged in?")
            continue
        if not up:
            problems.append(f"{iface} ({arg}) is down: run `pixi run can-up` (needed after every replug)")
    if problems:
        raise RuntimeError("use_fake_hardware:=false but:\n  " + "\n  ".join(problems))


def launch_setup(ctx):
    if LaunchConfiguration("use_fake_hardware").perform(ctx).lower() != "true":
        check_can_interfaces(ctx)
    share = get_package_share_directory(PKG)
    cfg = lambda f: os.path.join(share, "config", f)  # noqa: E731
    version = _arm_version(LaunchConfiguration("arm_type").perform(ctx))
    srdf = LaunchConfiguration("srdf").perform(ctx).strip() or cfg(
        f"openarm_bimanual_{version.replace('.', '')}.srdf")

    robot_description = build_robot_description(ctx)
    ee_links = {s: LaunchConfiguration(f"{s}_ee_link").perform(ctx) or default_ee_link(s, version)
                for s in ("left", "right")}

    moveit_config = (
        MoveItConfigsBuilder("openarm", package_name=PKG)
        .robot_description_semantic(file_path=srdf)
        .robot_description_kinematics(file_path=cfg("kinematics.yaml"))
        .joint_limits(file_path=cfg("joint_limits.yaml"))
        .trajectory_execution(file_path=cfg("moveit_controllers.yaml"))
        .pilz_cartesian_limits(file_path=cfg("pilz_cartesian_limits.yaml"))
        .planning_scene_monitor(publish_robot_description=True,
                                publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"],
                            default_planning_pipeline="ompl")
        .to_moveit_configs()
    )
    moveit_config.robot_description = {"robot_description": robot_description}
    moveit_params = moveit_config.to_dict()
    use_sim_time = {"use_sim_time": False}

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[moveit_config.robot_description],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="both",
            parameters=[cfg("ros2_controllers.yaml")],
            # Jazzy reads the URDF from this topic (published by robot_state_publisher).
            remappings=[("~/robot_description", "/robot_description")],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[moveit_params, use_sim_time],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="log",
            arguments=["-d", cfg("moveit.rviz")],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
            ],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
        Node(
            package=PKG,
            executable="ee_pose_server",
            output="screen",
            parameters=[{
                "left_ee_link": ee_links["left"],
                "right_ee_link": ee_links["right"],
                "planning_frame": LaunchConfiguration("planning_frame"),
            }],
            condition=IfCondition(LaunchConfiguration("pose_server")),
        ),
    ]

    spawn = lambda *names: Node(  # noqa: E731
        package="controller_manager",
        executable="spawner",
        arguments=[*names, "--controller-manager", "/controller_manager"],
        output="screen",
    )
    controllers = [
        spawn("joint_state_broadcaster"),
        spawn("left_arm_controller", "right_arm_controller",
              "left_gripper_controller", "right_gripper_controller"),
    ]
    if LaunchConfiguration("gravity_comp").perform(ctx).lower() == "true":
        controllers.append(spawn("left_gravity_controller", "right_gravity_controller"))
        nodes.append(Node(
            package=PKG,
            executable="gravity_comp",
            output="screen",
            parameters=[{"scale": float(LaunchConfiguration("gravity_comp_scale").perform(ctx))}],
        ))
    if LaunchConfiguration("table").perform(ctx).lower() == "true":
        num = lambda n: float(LaunchConfiguration(n).perform(ctx))  # noqa: E731
        nodes.append(Node(  # one-shot: adds the box once move_group is up, then exits
            package=PKG,
            executable="table_collision",
            output="screen",
            parameters=[{
                "frame": LaunchConfiguration("planning_frame").perform(ctx),
                "x_min": num("table_x_min"), "x_max": num("table_x_max"),
                "y_min": num("table_y_min"), "y_max": num("table_y_max"),
                "z_min": 0.0, "z_max": num("table_top_z"),
            }],
        ))
    return nodes + [TimerAction(period=1.0, actions=controllers)]


def generate_launch_description():
    args = [
        DeclareLaunchArgument("urdf", default_value="",
                              description="Custom URDF/xacro path. Empty = upstream OpenArm."),
        DeclareLaunchArgument("srdf", default_value="",
                              description="Custom SRDF path. Empty = config/openarm_bimanual_<ver>.srdf."),
        DeclareLaunchArgument("arm_type", default_value="v2.0",
                              description="OpenArm hardware revision: v1.0 or v2.0."),
        DeclareLaunchArgument("use_fake_hardware", default_value="true",
                              description="true = mock_components, false = OpenArm over CAN."),
        DeclareLaunchArgument("right_can_interface", default_value="can0"),
        DeclareLaunchArgument("left_can_interface", default_value="can1"),
        DeclareLaunchArgument("can_fd", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("pose_server", default_value="true"),
        DeclareLaunchArgument("gravity_comp", default_value="true",
                              description="Feedforward gravity torque from the URDF (gravity_comp node)."),
        DeclareLaunchArgument("gravity_comp_scale", default_value="1.0",
                              description="Fraction of the model gravity torque applied (0..1)."),
        DeclareLaunchArgument("left_ee_link", default_value="",
                              description="Link left pose goals refer to. Empty = TCP of arm_type."),
        DeclareLaunchArgument("right_ee_link", default_value="",
                              description="Link right pose goals refer to. Empty = TCP of arm_type."),
        DeclareLaunchArgument("planning_frame", default_value="world",
                              description="Frame assumed when a pose has no frame_id."),
        DeclareLaunchArgument("table", default_value="false",
                              description="Add the table as a collision box (floor to table_top_z)."),
        DeclareLaunchArgument("table_x_min", default_value="0.18",
                              description="Table edge towards the robot [m, planning_frame]."),
        DeclareLaunchArgument("table_x_max", default_value="0.765"),
        DeclareLaunchArgument("table_y_min", default_value="-0.306"),
        DeclareLaunchArgument("table_y_max", default_value="0.292"),
        DeclareLaunchArgument("table_top_z", default_value="1.09",
                              description="Height of the table top [m]."),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
