"""Head camera: /camera/image_raw, /camera/image_raw/compressed, /camera/camera_info.

  ros2 launch openarm_camera camera.launch.py
  ros2 launch openarm_camera camera.launch.py calibration_file:=$PWD/calib/head_cam.npz
  ros2 launch openarm_camera camera.launch.py device:=/dev/video2 width:=1280 height:=720

camera_info is only published with calibration_file, for the calibrated resolution. The
frame (camera_optical_frame) comes from the robot description (sagres_openarm_v10.urdf.xacro).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = {  # name: (default, type)
    "device": ("/dev/v4l/by-id/usb-HBV_HD_CAMERA_HBV_HD_CAMERA-video-index0", str),
    "width": ("1920", int),
    "height": ("1080", int),
    "fps": ("30.0", float),
    "fourcc": ("MJPG", str),
    "frame_id": ("camera_optical_frame", str),
    "calibration_file": ("", str),
    "autofocus": ("false", bool),
    "focus": ("-1", int),
    "exposure": ("-1", int),
}


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="camera"),
        *(DeclareLaunchArgument(name, default_value=default)
          for name, (default, _) in ARGS.items()),
        Node(
            package="openarm_camera",
            executable="camera_publisher",
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=[{name: ParameterValue(LaunchConfiguration(name), value_type=t)
                         for name, (_, t) in ARGS.items()}],
        ),
    ])
