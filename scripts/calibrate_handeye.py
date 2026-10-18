#!/usr/bin/env python3
"""Hand-eye (eye-to-hand) calibration of the static head camera from a rosbag.

The ChArUco board is fixed rigidly to one gripper; the robot moves it to many poses in front
of the camera and holds each one. From the bag the script takes the board pose seen by the
camera (solvePnP, with the intrinsics from calibrate_intrinsics.py) and the gripper pose from
/tf, then solves

    T_base_cam * T_cam_board(i) = T_base_ee(i) * T_ee_board        for every view i

for the two unknowns T_base_cam (where the camera is) and T_ee_board (where the board sits on
the gripper): a closed-form AX = XB solution for a start value, then a joint least-squares
refinement on the reprojection error of all board corners. Only frames where the arm has been still for
--settle seconds are used, one per arm pose, so image/TF time offsets don't matter.

Output: JSON with T_base_cam (camera_optical_frame: z forward, x right, y down), the matching
camera_link pose (REP 103, the URDF's camera_xyz / camera_rpy) and T_ee_board.

Recording (robot bringup and camera running, board taped/clamped flat to the gripper so it
cannot shift; it doesn't matter where exactly):
  ros2 bag record -o calib_handeye /camera/image_raw/compressed /tf /tf_static
Move the arm through 15-30 poses with the board fully in view (send_pose or RViz), pausing
~2 s at each. Vary the gripper orientation as much as possible: rotations about at least two
different axes, 30 deg or more. Translations alone leave the calibration undetermined.

Example:
  pixi run calib-handeye -- calib_handeye/ --board ../SagresOverdose/charuco_board_settings.json \\
      --intrinsics calib/head_cam.npz --ee-frame openarm_right_hand --out calib/handeye.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrate_intrinsics import load_board, mean_shift, ros_image_to_gray

# camera_link (REP 103: x forward, z up) -> camera_optical_frame, as in the URDF
T_LINK_OPTICAL = np.eye(4)
T_LINK_OPTICAL[:3, :3] = Rotation.from_euler("xyz", [-np.pi / 2, 0, -np.pi / 2]).as_matrix()


@dataclass
class View:
    index: int            # message index on the image topic
    stamp: float          # header stamp, s
    T_base_ee: np.ndarray
    T_cam_board: np.ndarray
    obj: np.ndarray       # (N, 3) board points
    img: np.ndarray       # (N, 2) detected pixels
    pnp_rms: float
    jpeg: bytes | None


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="rosbag2 folder")
    ap.add_argument("--board", type=Path, required=True, help="charuco_board_settings.json")
    ap.add_argument("--intrinsics", type=Path,
                    help="NPZ from calibrate_intrinsics.py; default: camera_info in the bag")
    ap.add_argument("--topic", help="image topic; default: the only image topic in the bag")
    ap.add_argument("--base-frame", default="world", help="frame the camera pose is given in")
    ap.add_argument("--ee-frame", default="openarm_right_hand",
                    help="link the board is fixed to (not a finger)")
    ap.add_argument("--out", type=Path, default=Path("handeye.json"))
    ap.add_argument("--square-length", type=float, help="measured square side, m")
    ap.add_argument("--marker-length", type=float, help="measured marker side incl. border, m")
    ap.add_argument("--min-corners", type=int, default=8, help="ChArUco corners per view")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="s the arm must have been still before a frame is used")
    ap.add_argument("--still-trans", type=float, default=0.001,
                    help="m: gripper moving more than this within --settle is not still")
    ap.add_argument("--still-rot", type=float, default=0.3, help="deg, same for rotation")
    ap.add_argument("--novelty-trans", type=float, default=0.01,
                    help="m: a view is a new arm pose if it is this far from all kept views ...")
    ap.add_argument("--novelty-rot", type=float, default=3.0, help="... or rotated this much")
    ap.add_argument("--max-pnp-error", type=float, default=2.0,
                    help="px: drop views whose board PnP fits worse (misdetections)")
    ap.add_argument("--outlier-factor", type=float, default=3.0,
                    help="drop views whose final RMS error exceeds this times the median ...")
    ap.add_argument("--outlier-min", type=float, default=1.5, help="... and this many px")
    ap.add_argument("--debug-dir", type=Path,
                    help="write every view with detected (green) and predicted (red) corners")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- transforms

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def inv(T):
    return make_T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def T_from_vec(v):
    return make_T(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:])


def vec_from_T(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def T_from_msg(tf):
    q, p = tf.rotation, tf.translation
    return make_T(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [p.x, p.y, p.z])


def pose_diff(A, B):
    """(translation m, rotation deg) between two poses."""
    D = inv(A) @ B
    return np.linalg.norm(D[:3, 3]), np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude())


# ---------------------------------------------------------------- bag

def open_reader(path: Path, topics):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path)), rosbag2_py.ConverterOptions("cdr", "cdr"))
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def read_tf(path: Path, types):
    """All /tf and /tf_static of the bag in a tf2 buffer."""
    from rclpy.duration import Duration
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage
    from tf2_ros import Buffer

    tf_topics = [t for t in ("/tf", "/tf_static") if types.get(t) == "tf2_msgs/msg/TFMessage"]
    if "/tf" not in tf_topics:
        sys.exit("no /tf in the bag: record /tf and /tf_static with the robot running")
    buf = Buffer(cache_time=Duration(seconds=1e7))
    n = 0
    reader = open_reader(path, tf_topics)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        for tf in deserialize_message(data, TFMessage).transforms:
            if topic == "/tf_static":
                buf.set_transform_static(tf, "bag")
            else:
                buf.set_transform(tf, "bag")
            n += 1
    print(f"read {n} transforms from {', '.join(tf_topics)}")
    return buf


def make_lookup(buf, base, ee):
    from rclpy.time import Time

    def lookup(t: float):
        try:
            tf = buf.lookup_transform(base, ee, Time(nanoseconds=int(t * 1e9)))
        except Exception:  # extrapolation / not connected
            return None
        return T_from_msg(tf.transform)

    return lookup


def check_frames(buf, base, ee, t0):
    from rclpy.time import Time
    if buf.can_transform(base, ee, Time(nanoseconds=int(t0 * 1e9))):
        return
    try:
        buf.lookup_transform(base, ee, Time(nanoseconds=int(t0 * 1e9)))
    except Exception as e:  # noqa: BLE001
        reason = str(e).splitlines()[0]
    sys.exit(f"cannot look up {base} -> {ee} at the first image ({reason}).\nFrames in the "
             f"bag:\n{buf.all_frames_as_string()}")


def read_intrinsics(args, path, types):
    if args.intrinsics:
        with np.load(args.intrinsics, allow_pickle=False) as npz:
            K, dist = np.asarray(npz["mtx"], float), np.asarray(npz["dist"], float).ravel()
            size = (int(npz["image_width"]), int(npz["image_height"])) \
                if "image_width" in npz else None
        return K, dist, size, str(args.intrinsics)
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo
    info_topics = [t for t, ty in types.items() if ty == "sensor_msgs/msg/CameraInfo"]
    if len(info_topics) != 1:
        sys.exit(f"pass --intrinsics (CameraInfo topics in the bag: {info_topics or 'none'})")
    reader = open_reader(path, info_topics)
    if not reader.has_next():
        sys.exit(f"{info_topics[0]} is empty: pass --intrinsics")
    msg = deserialize_message(reader.read_next()[1], CameraInfo)
    if not any(msg.k):
        sys.exit(f"{info_topics[0]} has no intrinsics: pass --intrinsics")
    return np.reshape(msg.k, (3, 3)), np.asarray(msg.d, float), (msg.width, msg.height), \
        info_topics[0]


def image_messages(path, topic, types):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, Image

    msg_types = {"sensor_msgs/msg/Image": Image, "sensor_msgs/msg/CompressedImage": CompressedImage}
    image_topics = sorted(t for t, ty in types.items() if ty in msg_types)
    if topic is None:
        if len(image_topics) != 1:
            sys.exit(f"pass --topic, image topics in the bag: {image_topics or 'none'}")
        topic = image_topics[0]
    elif topic not in image_topics:
        sys.exit(f"{topic} is not an image topic in the bag; image topics: {image_topics}")
    print(f"images from {topic}")
    reader = open_reader(path, [topic])
    i = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, msg_types[types[topic]])
        yield i, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg
        i += 1


# ---------------------------------------------------------------- view selection

def collect_views(args, board, detector, K, dist, size, lookup, images):
    views, prev = [], None
    counts = dict(frames=0, no_tf=0, moving=0, no_board=0, few_corners=0, similar=0,
                  bad_pnp=0)
    for index, stamp, msg in images:
        counts["frames"] += 1
        if counts["frames"] % 100 == 0:
            print(f"\r  image {index}: {len(views)} views", end="", file=sys.stderr, flush=True)
        T = lookup(stamp)
        past = [lookup(stamp - args.settle * f) for f in (1.0, 0.5)]
        if T is None or any(p is None for p in past):
            counts["no_tf"] += 1
            continue
        if any(d[0] > args.still_trans or d[1] > args.still_rot
               for d in (pose_diff(p, T) for p in past)):
            counts["moving"] += 1
            prev = None
            continue
        if any(pose_diff(v.T_base_ee, T)[0] < args.novelty_trans and
               pose_diff(v.T_base_ee, T)[1] < args.novelty_rot for v in views):
            counts["similar"] += 1
            continue

        gray = ros_image_to_gray(msg)
        if size and (gray.shape[1], gray.shape[0]) != tuple(size):
            sys.exit(f"images are {gray.shape[1]}x{gray.shape[0]}, intrinsics are for "
                     f"{size[0]}x{size[1]}")
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            counts["no_board"] += 1
            prev = None
            continue
        current = dict(zip(ids.ravel().tolist(), corners.reshape(-1, 2).astype(np.float64)))
        # The board must also be still in the image (the arm may sway after the joints stop).
        shift = mean_shift(prev, current, np.median) if prev else None
        prev = current
        if len(ids) < args.min_corners:
            counts["few_corners"] += 1
            continue
        if shift is None or shift > 0.5:
            counts["moving"] += 1
            continue
        obj, img = board.matchImagePoints(corners, ids)
        obj, img = obj.reshape(-1, 3), img.reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            counts["bad_pnp"] += 1
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        proj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        if rms > args.max_pnp_error:
            counts["bad_pnp"] += 1
            continue
        jpeg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() \
            if args.debug_dir else None
        views.append(View(index, stamp, T, make_T(cv2.Rodrigues(rvec)[0], tvec), obj, img,
                          rms, jpeg))
    print("", file=sys.stderr)
    print(f"looked at {counts['frames']} images: {len(views)} views (arm poses) kept; skipped "
          f"{counts['moving']} moving, {counts['similar']} same arm pose as a kept view, "
          f"{counts['no_tf']} without TF, {counts['no_board']} without board, "
          f"{counts['few_corners']} with < {args.min_corners} corners, "
          f"{counts['bad_pnp']} with PnP error > {args.max_pnp_error} px")
    return views


# ---------------------------------------------------------------- solve

def initial_guess(views):
    """Closed-form start value. Eliminating T_ee_board between views i and j gives
    A X = X B with A = T_base_ee(j) T_base_ee(i)^-1, B = T_cam_board(j) T_cam_board(i)^-1
    and X = T_base_cam. Rotation: rotvec(A) = R_X rotvec(B) for every pair (Kabsch, as in
    Park & Martin); translation: (R_A - I) t_X = R_X t_B - t_A, linear least squares.
    (OpenCV 5's Python bindings don't export cv2.calibrateHandEye.)"""
    A, B = [], []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            A.append(views[j].T_base_ee @ inv(views[i].T_base_ee))
            B.append(views[j].T_cam_board @ inv(views[i].T_cam_board))
    alpha = Rotation.from_matrix([a[:3, :3] for a in A]).as_rotvec()
    beta = Rotation.from_matrix([b[:3, :3] for b in B]).as_rotvec()
    U, _, Vt = np.linalg.svd(alpha.T @ beta)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    M = np.concatenate([a[:3, :3] - np.eye(3) for a in A])
    rhs = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    X = make_T(R, np.linalg.lstsq(M, rhs, rcond=None)[0])
    Y = board_on_ee(views, X)
    err = np.median([pose_diff(v.T_base_ee @ Y, X @ v.T_cam_board)[0] for v in views])
    print(f"initial guess: median board position mismatch {err * 1000:.1f} mm")
    return X, Y


def board_on_ee(views, X):
    """Mean T_ee_board implied by camera pose X."""
    Ys = [inv(v.T_base_ee) @ X @ v.T_cam_board for v in views]
    R = Rotation.from_matrix([Y[:3, :3] for Y in Ys]).mean().as_matrix()
    return make_T(R, np.mean([Y[:3, 3] for Y in Ys], axis=0))


def residuals(x, views, K, dist):
    X, Y = T_from_vec(x[:6]), T_from_vec(x[6:])
    X_inv = inv(X)
    out = []
    for v in views:
        T = X_inv @ v.T_base_ee @ Y
        proj = cv2.projectPoints(v.obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0]
        out.append((proj.reshape(-1, 2) - v.img).ravel())
    return np.concatenate(out)


def refine(views, X, Y, K, dist):
    x0 = np.concatenate([vec_from_T(X), vec_from_T(Y)])
    res = least_squares(residuals, x0, args=(views, K, dist), loss="huber", f_scale=1.0,
                        x_scale="jac")
    X, Y = T_from_vec(res.x[:6]), T_from_vec(res.x[6:])
    r = residuals(res.x, views, K, dist).reshape(-1, 2)
    sizes = np.cumsum([len(v.obj) for v in views])[:-1]
    per_view = np.array([np.sqrt(np.mean(np.sum(e ** 2, axis=1))) for e in np.split(r, sizes)])
    # Covariance from the Jacobian, scaled by the residual variance. Optimistic: it assumes
    # exact robot kinematics and independent corner errors.
    dof = max(r.size - x0.size, 1)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * np.sum(r ** 2) / dof
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    return X, Y, per_view, std


def solve(args, views, K, dist):
    outliers = []
    while True:
        if len(views) < 4:
            sys.exit(f"only {len(views)} usable arm poses, need at least 4 (better 15+)")
        X, Y = initial_guess(views)
        X, Y, per_view, std = refine(views, X, Y, K, dist)
        limit = max(args.outlier_factor * np.median(per_view), args.outlier_min)
        bad = per_view > limit
        if not bad.any():
            return views, outliers, X, Y, per_view, std
        outliers += [(v, e) for v, e, b in zip(views, per_view, bad) if b]
        views = [v for v, b in zip(views, bad) if not b]
        print(f"dropped {bad.sum()} outlier view(s) above {limit:.2f} px, solving again")


# ---------------------------------------------------------------- report

def motion_diversity(views):
    """Largest rotation between any two gripper poses (deg) and the rotation axes' spread."""
    Rs = Rotation.from_matrix([v.T_base_ee[:3, :3] for v in views])
    rel = [(Rs[i].inv() * Rs[j]).as_rotvec() for i in range(len(views))
           for j in range(i + 1, len(views))]
    rel = np.array(rel)
    max_angle = np.degrees(np.linalg.norm(rel, axis=1).max())
    s = np.linalg.svd(rel, compute_uv=False)
    return max_angle, np.degrees(s / np.sqrt(len(rel)))


def report(args, views, outliers, X, Y, per_view, std):
    pos_err = [pose_diff(v.T_base_ee @ Y, X @ v.T_cam_board) for v in views]
    T_link = X @ inv(T_LINK_OPTICAL)
    xyz = T_link[:3, 3]
    rpy = Rotation.from_matrix(T_link[:3, :3]).as_euler("xyz")
    max_angle, spread = motion_diversity(views)
    tilt = np.degrees(np.arccos(np.clip(-X[2, 2], -1, 1)))  # optical axis vs straight down

    print(f"\nviews            {len(views)} used, {len(outliers)} dropped as outliers")
    print(f"reprojection     RMS {np.sqrt(np.mean(per_view ** 2)):.2f} px (per view: median "
          f"{np.median(per_view):.2f}, max {per_view.max():.2f}); PnP alone: median "
          f"{np.median([v.pnp_rms for v in views]):.2f} px")
    print(f"board mismatch   median {np.median([e[0] for e in pos_err]) * 1000:.1f} mm, "
          f"{np.median([e[1] for e in pos_err]):.2f} deg (robot chain vs camera, per view)")
    print(f"uncertainty      camera position {np.linalg.norm(std[3:6]) * 1000:.1f} mm, "
          f"rotation {np.degrees(np.linalg.norm(std[:3])):.2f} deg (1 sigma, optimistic)")
    print(f"gripper motion   max rotation between views {max_angle:.0f} deg, axis spread "
          f"{' / '.join(f'{s:.0f}' for s in spread)} deg")
    print(f"\n{args.base_frame} -> camera_optical_frame:\n{np.array2string(X, precision=5)}")
    print(f"camera position  {xyz.round(4).tolist()} m, optical axis {tilt:.1f} deg from "
          f"straight down")
    print(f"{args.ee_frame} -> board origin: {(Y[:3, 3] * 1000).round(1).tolist()} mm")
    print(f'\nURDF (camera_link, REP 103): camera_xyz:="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" '
          f'camera_rpy:="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"')
    if args.base_frame != "world":
        print(f"  (relative to {args.base_frame}; the URDF's camera_link is a child of world)")

    warnings = []
    if len(views) < 12:
        warnings.append(f"only {len(views)} arm poses; 15-30 give a stabler result")
    if max_angle < 30:
        warnings.append("gripper rotated less than 30 deg between views: rotate it more")
    if spread[1] < 5:
        warnings.append("gripper rotations are (nearly) about one axis only: the camera "
                        "position along that axis is poorly determined; rotate about a "
                        "second axis too")
    if np.sqrt(np.mean(per_view ** 2)) > 3.0:
        warnings.append("reprojection error above 3 px: the board may have slipped on the "
                        "gripper, wrong --ee-frame, or images/TF out of sync (--settle)")
    for msg in warnings:
        print(f"WARNING: {msg}")
    return dict(xyz=xyz, rpy=rpy, pos_err=pos_err)


def write_debug(debug_dir, views, outliers, X, Y, per_view, K, dist):
    debug_dir.mkdir(parents=True, exist_ok=True)
    X_inv = inv(X)
    for v, err, tag in [*((v, e, "view") for v, e in zip(views, per_view)),
                        *((v, e, "outlier") for v, e in outliers)]:
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(v.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE),
                           cv2.COLOR_GRAY2BGR)
        T = X_inv @ v.T_base_ee @ Y  # board pose predicted from the robot
        rvec, tvec = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3]
        for x, y in v.img:
            cv2.circle(img, (round(x), round(y)), 8, (0, 255, 0), 2)
        for x, y in cv2.projectPoints(v.obj, rvec, tvec, K, dist)[0].reshape(-1, 2):
            cv2.drawMarker(img, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.05, 2)
        cv2.imwrite(str(debug_dir / f"{tag}_{v.index:06d}_{err:.2f}px.jpg"), img)
    print(f"debug images in {debug_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.bag.is_dir():
        sys.exit(f"{args.bag} is not a rosbag2 folder")
    board, detector = load_board(args.board, args.square_length, args.marker_length)
    reader = open_reader(args.bag, None)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    del reader
    K, dist, size, intrinsics_src = read_intrinsics(args, args.bag, types)
    buf = read_tf(args.bag, types)
    lookup = make_lookup(buf, args.base_frame, args.ee_frame)

    images = image_messages(args.bag, args.topic, types)
    first = next(images, None)
    if first is None:
        sys.exit("no images in the bag")
    check_frames(buf, args.base_frame, args.ee_frame, first[1] + args.settle)

    def all_images():
        yield first
        yield from images

    views = collect_views(args, board, detector, K, dist, size, lookup, all_images())
    views, outliers, X, Y, per_view, std = solve(args, views, K, dist)
    r = report(args, views, outliers, X, Y, per_view, std)

    result = {
        "bag": str(args.bag.resolve()),
        "intrinsics": intrinsics_src,
        "base_frame": args.base_frame,
        "ee_frame": args.ee_frame,
        "camera_frame": "camera_optical_frame",
        "T_base_camera_optical": X.tolist(),
        "camera_link": {"parent": args.base_frame, "xyz": r["xyz"].tolist(),
                        "rpy": r["rpy"].tolist()},
        "T_ee_board": Y.tolist(),
        "n_views": len(views),
        "rms_px": float(np.sqrt(np.mean(per_view ** 2))),
        "median_board_mismatch_m": float(np.median([e[0] for e in r["pos_err"]])),
        "std_camera_position_m": float(np.linalg.norm(std[3:6])),
        "std_camera_rotation_deg": float(np.degrees(np.linalg.norm(std[:3]))),
        "board": {"square_length": board.getSquareLength(),
                  "marker_length": board.getMarkerLength()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve()}")
    if args.debug_dir:
        write_debug(args.debug_dir, views, outliers, X, Y, per_view, K, dist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
