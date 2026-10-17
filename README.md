# openarm_moveit

End-effector pose control for the bimanual OpenArm with MoveIt 2 (ROS 2 Jazzy), in a
self-contained pixi environment (RoboStack). No system ROS install is needed.

## Quick start

```bash
git clone --recurse-submodules <this repo> && cd openarm_moveit
pixi install
pixi run demo        # build + MoveIt/RViz demo with the SagresOverdose model, mock hardware
pixi run smoke       # headless check: pose goals on both arms + topic interface
```

In a second terminal (`pixi shell`, or prefix with `pixi run`):

```bash
ros2 run openarm_ee_control send_pose --arm right --print                      # current TCP pose
ros2 run openarm_ee_control send_pose --arm right --xyz 0.2 0 0.15 --relative  # plan + execute
ros2 run openarm_ee_control send_pose --arm left --xyz 0 0 0.05 --relative --cartesian
ros2 run openarm_ee_control send_pose --arm left --xyz 0.3 0.2 1.0 --rpy 0 1.57 0  # absolute, world frame
```

## Robot model

| Launch | Model |
|---|---|
| `demo.launch.py` (default for `pixi run demo`) | `urdf/sagres_openarm_v10.urdf.xacro`: upstream v1.0 bimanual with the body raised to z = 0.628 m (shoulders at 1.325 m) and `camera_link` at the head-camera pose, as in SagresOverdose branch `urdf` |
| `bringup.launch.py` | upstream `openarm_description` v2.0 (or `arm_type:=v1.0`) |

Both accept `urdf:=/path/to/model.urdf[.xacro]` and `srdf:=...`. If a custom URDF has no
`<ros2_control>` block, the upstream OpenArm one is added automatically (`urdf/add_ros2_control_*`).
Joint names must stay `openarm_{left,right}_joint{1..7}` / `openarm_{left,right}_finger_joint1`,
otherwise provide a matching SRDF and controller config.

The world offset and camera pose are xacro args (`xyz`, `rpy`, `camera`, `camera_xyz`,
`camera_rpy`). Set `camera:=false` in the xacro if another node publishes `camera_link`.

## Control interfaces

`ee_pose_server` (started by both launch files) gives a topic API, e.g. for a learned policy:

| Topic | Type | |
|---|---|---|
| `/openarm/{left,right}/target_pose` | `geometry_msgs/PoseStamped` (in) | OMPL plan + execute |
| `/openarm/{left,right}/target_pose_cartesian` | `geometry_msgs/PoseStamped` (in) | straight-line move |
| `/openarm/{left,right}/ee_pose` | `geometry_msgs/PoseStamped` (out, 30 Hz) | current TCP pose |
| `/openarm/{left,right}/status` | `std_msgs/String` (out) | `planning`, `executing`, `succeeded`, `failed: ...` |

Poses refer to the TCP link: `openarm_<side>_hand_tcp` (v1.0) or `openarm_<side>_grasp_frame`
(v2.0); override with `left_ee_link:=` / `right_ee_link:=`. Each arm plans via move_group and
executes on its own `*_arm_controller`, so both arms move concurrently and a new target replaces
the running one. The RViz MotionPlanning panel works as usual (groups `left_arm`, `right_arm`).

## Real hardware

```bash
pixi run can-up                                         # can0 = right, can1 = left, CAN-FD
ros2 launch openarm_ee_control demo.launch.py use_fake_hardware:=false
```

Gravity compensation is on by default (`gravity_comp:=true`, `gravity_comp_scale:=1.0`): the
`gravity_comp` node computes the holding torque per joint from the URDF (Pinocchio) and the
measured joint positions and sends it through `<side>_gravity_controller` (effort interface) as
motor feedforward, next to the trajectory controller's position command. Torques are clamped per
joint, ramped in over 2 s, faded to zero if `/joint_states` stops for 0.2 s, and zeroed on
shutdown. Every 5 s it logs model torque vs. measured effort per arm. Turn it off with
`gravity_comp:=false`, or start lower with e.g. `gravity_comp_scale:=0.5`.

Motor gains (kp/kd per joint and gripper) are in `src/openarm_ee_control/config/control_gains_v10.yaml`;
rebuild and relaunch after editing.

Velocities are capped at 2 rad/s in `config/joint_limits.yaml` and scaled by 0.2 by default
(the URDF allows ~21 rad/s). Keep the e-stop in reach for the first run.

## Layout

```
src/openarm_description   upstream, submodule (pinned)
src/openarm_ros2          upstream, submodule: openarm_hardware (CAN ros2_control plugin), bringup
src/openarm_can           upstream, submodule: CAN library used by openarm_hardware
src/openarm_ee_control    this repo: launch, MoveIt/controller config, pose server, send_pose
```

## Packages in the pixi env

ROS 2 Jazzy base, `xacro`, `robot_state_publisher`, `rviz2`, `ros2_control` + `ros2_controllers`,
`moveit` (+ `moveit_configs_utils`, `moveit_servo`, `moveit_py`, Pilz), `trac_ik_kinematics_plugin`,
Pinocchio (gravity compensation), colcon and a C++ toolchain. `ros_gz` and `zed_description` are only there because
`openarm_description` declares them.

## Notes

- IK uses TRAC-IK. KDL failed even 5 mm Cartesian steps on this arm.
- Upstream `openarm_bimanual_moveit_config` declares the grippers as `GripperCommand` but runs
  them as trajectory controllers; `config/moveit_controllers.yaml` uses `FollowJointTrajectory`.
- `setuptools` is pinned below 80: newer versions break `colcon build --symlink-install`.
