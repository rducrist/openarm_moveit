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

More topics, used by the replay tool and the policy node:

| Topic | Type | |
|---|---|---|
| `/openarm/{left,right}/target_path_cartesian` | `geometry_msgs/PoseArray` (in) | straight-line moves through all poses, one trajectory |
| `/openarm/{left,right}/target_joints` | `sensor_msgs/JointState` (in) | OMPL plan + execute to joint positions (name + position) |
| `/openarm/{left,right}/gripper` | `std_msgs/Float64` (in) | 0 = open (finger 0.044 m) .. 1 = closed (0.0 m) |
| `/openarm/{left,right}/stop` | `std_msgs/Empty` (in) | cancel the running motion |
| `/openarm/{left,right}/gripper_status` | `std_msgs/String` (out) | `executing`, `succeeded`, `failed: ...` |

## Running an R2R2R diffusion policy

`openarm_policy` runs a [tinydp](https://github.com/Max-Fu/tinydp) diffusion policy trained on
[Real2Render2Real](https://github.com/uynitsuj/real2render2real) data, step by step. Each episode
starts with **go to start pose → start policy** (both confirmed with Enter); then each step
goes observe (camera + TCP poses + commanded grippers) → predict a chunk of absolute TCP poses → execute the
first `--steps-per-move` of them as one Cartesian move (MoveIt, collision-checked) → wait until
the arm stops → observe again. The images the policy was trained on show the rendered arm, so
it needs the real arm in the loop; mock hardware only tests the plumbing.

```bash
git submodule update --init third_party/tinydp   # model code (pinned to R2R2R's commit)
pixi install -e policy                           # torch 2.5.1 (CUDA 12.4), diffusers, timm

# terminal 1: robot
ros2 launch openarm_ee_control demo.launch.py use_fake_hardware:=false
# terminal 2: camera driver publishing sensor_msgs/Image (rgb8/bgr8), same pose and aspect
#             ratio as the camera the policy was rendered with, e.g. `pixi run camera`
#             (/camera/image_raw)
# terminal 3: replay a generated episode first (checks frame, TCP, gripper)
pixi run replay -- <output_data/.../successes/env_X> --mode dry   # shows it in RViz only
pixi run replay -- <output_data/.../successes/env_X> --arms right  # Enter per move
# store the sim's start pose next to the checkpoint (once; no robot needed)
pixi run replay -- <output_data/.../successes/env_X> --save-start <folder>/start_pose.yaml
# then the policy
pixi run -e policy policy -- --ckpt-dir <folder> --image-topic <topic> --mode dry
pixi run -e policy policy -- --ckpt-dir <folder> --image-topic <topic>             # step mode
```

The checkpoint folder holds `run.yaml`, `action_statistics.json`, `checkpoint_<N>.pt` and,
optionally, `start_pose.yaml`. Everything else is read from `run.yaml`: which arm(s) the policy
drives, history length, chunk length, image size.

- Start pose: the posture the sim episodes start from, i.e. the joint angles of an episode's
  first frame (`joint_angles` + `joint_names.txt`), reached with an OMPL joint goal, one arm
  after the other. Joint angles reproduce the sim's arm posture (elbow) in the images; a TCP
  pose alone would not on a 7-DoF arm, so `tcp` is only the fallback for episodes without
  joint data. Taken from `--start` (YAML or episode) or `<ckpt-dir>/start_pose.yaml`. After
  reaching it, the node warns if the policy arm's TCP is outside the positions in the
  training data (`action_statistics.json`). `h` at any prompt goes back to it for a new
  episode; an episode also ends after `--max-steps`.

- Modes: `dry` predicts and shows the chunk (`/openarm/<arm>/planned_path` in RViz) without
  moving; `step` (default) executes a step on Enter (`r` re-samples, `a` switches to auto);
  `auto` runs until `--max-steps`. Ctrl+C stops the arm and returns to step mode.
- Before every step the waypoints are checked against a workspace box (`--workspace-min/max`,
  set z above the table) and a maximum step of `--max-step` m per waypoint (incl. the current
  TCP to the first waypoint). A violation stops auto mode.
- Poses are in `world` (`--frame`), the root frame of the Sagres URDF that R2R2R rendered.
  Gripper proprio is the *commanded* state, as R2R2R logs `gripper_binary_cmd`.
- Observation history: with `seq_length` 1 the policy sees the frame taken after the arm
  settled. With longer histories each executed waypoint adds one entry (the frame recorded
  during the move where the arm was closest to it), matching tinydp's spacing of one action
  step per history step.
- `--ddim-steps 10` denoises in 10 DDIM steps instead of the full 100-step DDPM chain that
  tinydp's wrapper runs; `--weights ema` uses the EMA weights.
- `--fake` replaces the network by scripted 3 mm steps (no torch needed) to test the loop.

`traj_replay` reads `robot_data/robot_data.h5` of an R2R2R episode (`ee_poses` as
`[x y z qw qx qy qz]` left then right, `gripper_binary_cmd` with 1 = closed), subsamples it like
the policy's actions (`--subsample 2`), goes to the episode's start pose (as above) and then replays it in
Cartesian moves of `--steps-per-move` waypoints, with the same modes and checks.

## Head camera and calibration

`openarm_camera` publishes the HBV USB head camera (1920x1080 MJPEG at 30 fps by default):

```bash
pixi run camera                                                  # no camera_info yet
ros2 launch openarm_camera camera.launch.py calibration_file:=$PWD/calib/head_cam.npz
```

| Topic | Type | |
|---|---|---|
| `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | the camera's own JPEGs, passed through (record this one) |
| `/camera/image_raw` | `sensor_msgs/Image` (bgr8) | decoded only while someone subscribes |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | only with `calibration_file`, at the calibrated resolution |

Images carry `frame_id` `camera_optical_frame`, which the Sagres URDF attaches to `camera_link`
(REP 103 optical rotation). Other launch args: `device`, `width`, `height`, `fps`, `fourcc`,
`autofocus`, `focus`, `exposure`.

Calibration uses the ChArUco board in `SagresOverdose/charuco_board_settings.json` (5 x 4,
32 mm squares). If the board as built differs from the nominal size, measure it and pass
`--square-length` / `--marker-length` to both scripts.

1. **Intrinsics.** With the camera running, record the hand-held board at 40-60 still poses
   (tilted up to 45 deg, covering the image corners), then calibrate:
   ```bash
   ros2 bag record -o calib_intrinsics /camera/image_raw/compressed
   pixi run calib-intrinsics -- calib_intrinsics --board ../SagresOverdose/charuco_board_settings.json \
       --out calib/head_cam.npz --debug-dir calib/intrinsics_debug
   ```
2. **Camera pose (eye-to-hand).** Fix the board rigidly to one gripper (where exactly does not
   matter), bring up the robot, and record 15-30 arm poses held ~2 s each. Rotate the gripper
   about at least two axes (30 deg or more); translations alone leave the pose undetermined.
   ```bash
   ros2 bag record -o calib_handeye /camera/image_raw/compressed /tf /tf_static
   pixi run calib-handeye -- calib_handeye --board ../SagresOverdose/charuco_board_settings.json \
       --intrinsics calib/head_cam.npz --ee-frame openarm_right_hand --out calib/handeye.json \
       --debug-dir calib/handeye_debug
   ```
   It prints `camera_xyz:=... camera_rpy:=...` for the URDF (`camera_link` in `world`) and
   writes `world -> camera_optical_frame` plus the board's pose on the gripper to the JSON.
   Check the reprojection RMS (should be a few px at most) and the debug images: red crosses
   (predicted from the robot) on the green circles (detected).

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
src/openarm_ee_control    this repo: launch, MoveIt/controller config, pose server, send_pose,
                          step executor + traj_replay
src/openarm_policy        this repo: R2R2R / tinydp policy node (pixi env `policy`)
src/openarm_camera        this repo: head camera publisher
scripts/calibrate_*.py    camera intrinsics and eye-to-hand calibration from rosbags
third_party/tinydp        upstream, submodule: diffusion policy model code (not built by colcon)
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
