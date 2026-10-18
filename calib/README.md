# Head camera calibration (HBV USB camera, 1920x1080)

Board: `charuco_board_settings.json` from SagresOverdose (branch `calibration`): 5 x 4
ChArUco, DICT_4X4_50, 32 mm squares, 24 mm markers, nominal size (the hand-drawn board was
not measured).
Made with `scripts/calibrate_intrinsics.py` and `scripts/calibrate_handeye.py` in
this repo, from rosbags of `/camera/image_raw/compressed` (+ `/tf`, `/tf_static`).

## head_cam.npz: intrinsics, usable

`mtx` (3x3), `dist` (plumb_bob k1 k2 p1 p2 k3), `image_width`/`image_height`; loadable by
openarm_camera (`calibration_file:=`) and SagresOverdose's openarm_camera_info node.

| | value | 1 sigma |
|---|---|---|
| fx, fy | 1049.3, 1057.4 px | 36 px |
| cx, cy | 940.0, 539.7 px | 22 px |
| dist | -0.373, 0.232, -0.0007, -0.0026, -0.076 | |
| RMS | 0.35 px, 25 views | |

The fit is good, but the focal length is only known to ~3%: more strongly tilted views
near the image corners would tighten it.

## handeye.json: camera pose, NOT usable yet

`world -> camera_optical_frame` from the board on `openarm_left_hand` (bag `calib_handeye_2`):
camera at (0.163, -0.010, 1.643) m, looking almost straight down.

Only 9 arm poses, reprojection RMS 11 px, median board mismatch 7 cm, camera position
uncertain by ~4 cm / 5 deg. Likely causes: too few poses or too little gripper rotation,
the board shifting on the gripper, or the nominal board size. Re-record with 15-30 poses
rotated about two axes before using it.
