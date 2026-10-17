"""MoveIt demo for the SagresOverdose OpenArm (v1.0 bimanual, raised body, head camera).

  ros2 launch openarm_ee_control demo.launch.py                          # mock hardware + RViz
  ros2 launch openarm_ee_control demo.launch.py use_fake_hardware:=false # real arms over CAN

Same stack as bringup.launch.py, with urdf:= defaulting to urdf/sagres_openarm_v10.urdf.xacro
and arm_type:=v1.0 (SRDF, TCP links and the ros2_control block follow from that).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

DEFAULTS = {
    "urdf": os.path.join(get_package_share_directory("openarm_ee_control"), "urdf",
                         "sagres_openarm_v10.urdf.xacro"),
    "srdf": "",
    "arm_type": "v1.0",
    "use_fake_hardware": "true",
    "right_can_interface": "can0",
    "left_can_interface": "can1",
    "can_fd": "true",
    "rviz": "true",
    "pose_server": "true",
    "gravity_comp": "true",
    "gravity_comp_scale": "1.0",
    "left_ee_link": "",
    "right_ee_link": "",
    "planning_frame": "world",
}


def generate_launch_description():
    bringup = os.path.join(get_package_share_directory("openarm_ee_control"), "launch",
                           "bringup.launch.py")
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in DEFAULTS.items()]
        + [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={k: LaunchConfiguration(k) for k in DEFAULTS}.items())]
    )
