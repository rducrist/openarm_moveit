#!/usr/bin/env python3
"""Intrinsic camera calibration from a video (or rosbag) of a ChArUco board.

Detects the board in every frame, keeps frames where it is still and that differ from the
views already kept, runs cv2.calibrateCamera (plumb_bob: k1 k2 p1 p2 k3), drops outlier views
and recalibrates. Writes an NPZ with `mtx` (3x3) and `dist` (1x5) plus the image size: the
calibration_file of openarm_camera (camera.launch.py) and the --intrinsics of
calibrate_handeye.py. SagresOverdose's openarm_camera_info node (branch urdf) loads it too.

The board comes from a charuco_board_settings.json (SagresOverdose). If the board was printed
or drawn at a slightly different size, pass the measured --square-length / --marker-length:
the intrinsics don't depend on the board's scale, but the reported distances do.

Recording (same camera and resolution as in deployment, autofocus off):
  ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 \\
      -i /dev/v4l/by-id/usb-HBV_HD_CAMERA_HBV_HD_CAMERA-video-index0 -c:v copy calib.mkv
(-c:v copy stores the camera's own MJPEG frames, no re-encoding), or with the camera node
running (`pixi run camera`; its compressed topic passes the MJPEG frames through as well):
  ros2 bag record -o calib_intrinsics /camera/image_raw/compressed
Hold the board 0.3-0.6 m from the camera and pause ~1 s at each pose: frames where the board
moves are skipped (motion blur, rolling shutter). Tilt it up to ~45 deg in all directions and
bring it into every part of the image, the corners especially. 40-60 poses are plenty.

Examples:
  pixi run calib-intrinsics -- calib.mkv --board ../SagresOverdose/charuco_board_settings.json
  pixi run calib-intrinsics -- calib_intrinsics/ --topic /camera/image_raw/compressed --board BOARD \\
      --square-length 0.0318 --marker-length 0.0239 --out head_cam.npz --debug-dir calib_debug
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COVERAGE_GRID = (6, 4)  # columns, rows of the image-coverage report


@dataclass
class View:
    index: int              # frame index in the video / message index in the bag
    corners: dict           # ChArUco id -> (x, y) pixel
    obj: np.ndarray         # (N, 1, 3) board points
    img: np.ndarray         # (N, 1, 2) detected image points
    jpeg: bytes | None      # frame, kept only for --debug-dir


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="video file, or rosbag2 folder")
    ap.add_argument("--board", type=Path, required=True, help="charuco_board_settings.json")
    ap.add_argument("--topic", help="image topic in the bag (sensor_msgs/Image or "
                                    "CompressedImage); default: the only image topic")
    ap.add_argument("--out", type=Path, default=Path("intrinsics.npz"))
    ap.add_argument("--square-length", type=float, help="measured square side, m")
    ap.add_argument("--marker-length", type=float, help="measured marker side incl. border, m")
    ap.add_argument("--step", type=int, default=1, help="look at every n-th frame")
    ap.add_argument("--min-corners", type=int, default=6, help="ChArUco corners per view")
    ap.add_argument("--max-motion", type=float, default=1.0,
                    help="px per frame: board moving faster than this is skipped")
    ap.add_argument("--novelty", type=float, default=0.02,
                    help="a view is kept only if its corners are at least this fraction of "
                         "the image diagonal (mean) away from every kept view")
    ap.add_argument("--max-views", type=int, default=100,
                    help="subsample evenly in time if more views qualify")
    ap.add_argument("--outlier-factor", type=float, default=3.0,
                    help="drop views whose RMS error exceeds this times the median ...")
    ap.add_argument("--outlier-min", type=float, default=0.5,
                    help="... and this many px")
    ap.add_argument("--fix-k3", action="store_true",
                    help="keep k3 = 0 (try this if k3 comes out large with poor coverage)")
    ap.add_argument("--debug-dir", type=Path,
                    help="write every view with detected (green) and reprojected (red) "
                         "corners, and one undistorted frame")
    args = ap.parse_args(argv)
    if args.min_corners < 6:
        ap.error("--min-corners must be >= 6 (fewer can be collinear on a 4 x 3 corner grid)")
    return args


def load_board(path: Path, square_length: float | None, marker_length: float | None):
    cfg = json.loads(path.read_text())["board"]
    if cfg.get("type", "ChArUco") != "ChArUco" or cfg.get("length_unit", "metre") != "metre":
        sys.exit(f"{path}: expected a ChArUco board in metres")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (cfg["squares_x"], cfg["squares_y"]),
        square_length or cfg["square_length"], marker_length or cfg["marker_length"],
        dictionary, np.asarray(cfg["marker_ids"], dtype=np.int32))
    board.setLegacyPattern(bool(cfg["legacy_pattern"]))
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = int(cfg.get("marker_border_bits", 1))
    detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)
    return board, detector


# ---------------------------------------------------------------- frame sources (gray images)

def video_frames(path: Path, step: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{path}: {total} frames at {cap.get(cv2.CAP_PROP_FPS):.1f} fps")
    i = 0
    while True:
        if i % step:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()


GRAY_CONVERSIONS = {  # ROS image encoding -> (channels, cv2 conversion to gray)
    "mono8": (1, None), "8UC1": (1, None),
    "bgr8": (3, cv2.COLOR_BGR2GRAY), "rgb8": (3, cv2.COLOR_RGB2GRAY),
    "bgra8": (4, cv2.COLOR_BGRA2GRAY), "rgba8": (4, cv2.COLOR_RGBA2GRAY),
    "yuv422_yuy2": (2, cv2.COLOR_YUV2GRAY_YUY2), "yuyv": (2, cv2.COLOR_YUV2GRAY_YUY2),
    "yuv422": (2, cv2.COLOR_YUV2GRAY_UYVY), "uyvy": (2, cv2.COLOR_YUV2GRAY_UYVY),
}


def ros_image_to_gray(msg) -> np.ndarray:
    if hasattr(msg, "format"):  # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if msg.encoding not in GRAY_CONVERSIONS:
        sys.exit(f"unsupported image encoding {msg.encoding!r}")
    ch, conversion = GRAY_CONVERSIONS[msg.encoding]
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = rows[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    return img[:, :, 0].copy() if conversion is None else cv2.cvtColor(img, conversion)


def bag_frames(path: Path, topic: str | None, step: int):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, Image

    msg_types = {"sensor_msgs/msg/Image": Image, "sensor_msgs/msg/CompressedImage": CompressedImage}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path)), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    image_topics = sorted(t for t, ty in types.items() if ty in msg_types)
    if topic is None:
        if len(image_topics) != 1:
            sys.exit(f"pass --topic, image topics in the bag: {image_topics or 'none'}")
        topic = image_topics[0]
    elif topic not in image_topics:
        sys.exit(f"{topic} is not an image topic in the bag; image topics: {image_topics}")
    msg_type = msg_types[types[topic]]
    count = next((t.message_count for t in reader.get_metadata().topics_with_message_count
                  if t.topic_metadata.name == topic), "?")
    print(f"{path}: {count} messages on {topic} ({types[topic]})")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    i = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        if i % step == 0:
            yield i, ros_image_to_gray(deserialize_message(data, msg_type))
        i += 1


# ---------------------------------------------------------------- view selection

def mean_shift(a: dict, b: dict, stat=np.mean) -> float | None:
    """Mean (or other stat) pixel distance between corners seen in both views, None if < 4."""
    common = a.keys() & b.keys()
    if len(common) < 4:
        return None
    return float(stat([np.hypot(*(a[k] - b[k])) for k in common]))


def collect_views(frames, board, detector, args):
    views, size, prev = [], None, None
    counts = dict(frames=0, no_board=0, few_corners=0, moving=0, similar=0)
    min_shift = None
    for index, gray in frames:
        counts["frames"] += 1
        if size is None:
            size = (gray.shape[1], gray.shape[0])
            min_shift = args.novelty * np.hypot(*size)
        elif (gray.shape[1], gray.shape[0]) != size:
            sys.exit(f"frame {index} is {gray.shape[1]}x{gray.shape[0]}, earlier frames {size}")
        if counts["frames"] % 50 == 0:
            print(f"\r  frame {index}: {len(views)} views", end="", file=sys.stderr, flush=True)

        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            counts["no_board"] += 1
            prev = None
            continue
        current = dict(zip(ids.ravel().tolist(), corners.reshape(-1, 2).astype(np.float64)))
        # Speed from the previous looked-at frame; unknown (first frame of a pause) = moving.
        speed = None
        if prev is not None:
            shift = mean_shift(prev[1], current, np.median)
            speed = None if shift is None else shift / (index - prev[0])
        prev = (index, current)
        if len(ids) < args.min_corners:
            counts["few_corners"] += 1
            continue
        if speed is None or speed > args.max_motion:
            counts["moving"] += 1
            continue
        shifts = (mean_shift(v.corners, current) for v in views)
        if any(s is not None and s < min_shift for s in shifts):
            counts["similar"] += 1
            continue
        obj, img = board.matchImagePoints(corners, ids)
        jpeg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() \
            if args.debug_dir else None
        views.append(View(index, current, obj, img, jpeg))
    print("", file=sys.stderr)
    if size is None:
        sys.exit("no frames read")
    print(f"looked at {counts['frames']} frames: {len(views)} views kept; skipped "
          f"{counts['no_board']} without board, {counts['few_corners']} with < "
          f"{args.min_corners} corners, {counts['moving']} moving, {counts['similar']} "
          f"similar to a kept view")
    if len(views) > args.max_views:
        keep = np.linspace(0, len(views) - 1, args.max_views).round().astype(int)
        views = [views[i] for i in keep]
        print(f"subsampled to {len(views)} views (--max-views)")
    return views, size


# ---------------------------------------------------------------- calibration

def calibrate(views, size, flags):
    rms, K, dist, rvecs, tvecs, std, _, per_view = cv2.calibrateCameraExtended(
        [v.obj for v in views], [v.img for v in views], size, None, None, flags=flags)
    return dict(rms=rms, K=K, dist=dist.reshape(1, -1)[:, :5], rvecs=rvecs, tvecs=tvecs,
                std=std.ravel(), per_view=per_view.ravel())


def calibrate_robust(views, size, args):
    flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    outliers = []
    for _ in range(5):
        if len(views) < 10:
            sys.exit(f"only {len(views)} usable views, need at least 10: record more poses or "
                     f"relax --max-motion / --novelty")
        res = calibrate(views, size, flags)
        limit = max(args.outlier_factor * np.median(res["per_view"]), args.outlier_min)
        bad = res["per_view"] > limit
        if not bad.any():
            break
        outliers += [(v, e) for v, e, b in zip(views, res["per_view"], bad) if b]
        views = [v for v, b in zip(views, bad) if not b]
        print(f"dropped {bad.sum()} outlier view(s) above {limit:.2f} px, recalibrating")
    else:
        res = calibrate(views, size, flags)
    return views, outliers, res


def report(views, outliers, res, size):
    K, d, std = res["K"], res["dist"].ravel(), res["std"]
    w, h = size
    tilts, dists = [], []
    for rvec, tvec in zip(res["rvecs"], res["tvecs"]):
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        tilts.append(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
        dists.append(np.linalg.norm(tvec))
    cols, rows = COVERAGE_GRID
    grid = np.zeros((rows, cols), int)
    for v in views:
        p = v.img.reshape(-1, 2)
        np.add.at(grid, (np.clip((p[:, 1] * rows / h).astype(int), 0, rows - 1),
                         np.clip((p[:, 0] * cols / w).astype(int), 0, cols - 1)), 1)
    hfov, vfov = (np.degrees(2 * np.arctan(n / (2 * f))) for n, f in ((w, K[0, 0]), (h, K[1, 1])))

    print(f"\nimage size       {w} x {h}")
    print(f"views            {len(views)} used, {len(outliers)} dropped as outliers")
    print(f"RMS reprojection {res['rms']:.3f} px (per view: median {np.median(res['per_view']):.3f},"
          f" max {res['per_view'].max():.3f})")
    print(f"fx, fy           {K[0, 0]:.2f} +- {std[0]:.2f}, {K[1, 1]:.2f} +- {std[1]:.2f} px")
    print(f"cx, cy           {K[0, 2]:.2f} +- {std[2]:.2f}, {K[1, 2]:.2f} +- {std[3]:.2f} px")
    print("dist             " + "  ".join(f"{n} {v:+.5f} +- {s:.5f}" for n, v, s in
                                          zip(("k1", "k2", "p1", "p2", "k3"), d, std[4:9])))
    print(f"field of view    {hfov:.1f} x {vfov:.1f} deg")
    print(f"board tilt       {min(tilts):.0f} - {max(tilts):.0f} deg (median "
          f"{np.median(tilts):.0f}), distance {min(dists):.2f} - {max(dists):.2f} m")
    print(f"coverage         corners per image cell ({cols} x {rows}):")
    for row in grid:
        print("                 " + " ".join(f"{n:5d}" for n in row))

    warnings = []
    if len(views) < 25:
        warnings.append(f"only {len(views)} views; 40-60 distinct poses give stabler results")
    if res["rms"] > 1.0:
        warnings.append("RMS above 1 px: check the board (flat? corners accurate?) and the "
                        "--debug-dir images for misdetections")
    if max(tilts) < 30:
        warnings.append("board never tilted beyond 30 deg: focal length is poorly constrained")
    if (grid == 0).any():
        warnings.append(f"{(grid == 0).sum()} image cell(s) without any corner: distortion is "
                        "extrapolated there, record views near those edges/corners")
    if abs(K[0, 0] / K[1, 1] - 1) > 0.02:
        warnings.append("fx and fy differ by more than 2%: unusual for square pixels")
    if abs(K[0, 2] - w / 2) > 0.1 * w or abs(K[1, 2] - h / 2) > 0.1 * h:
        warnings.append("principal point is more than 10% off the image centre")
    if std[0] > 0.01 * K[0, 0]:
        warnings.append("fx uncertainty above 1%: add more tilted views")
    for msg in warnings:
        print(f"WARNING: {msg}")


def write_debug(debug_dir: Path, views, outliers, res):
    debug_dir.mkdir(parents=True, exist_ok=True)
    K, dist = res["K"], res["dist"]
    poses = {id(v): (r, t) for v, r, t in zip(views, res["rvecs"], res["tvecs"])}
    for v, err, tag in [*((v, e, "view") for v, e in zip(views, res["per_view"])),
                        *((v, e, "outlier") for v, e in outliers)]:
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(v.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE),
                           cv2.COLOR_GRAY2BGR)
        for x, y in v.img.reshape(-1, 2):
            cv2.circle(img, (round(x), round(y)), 8, (0, 255, 0), 2)
        if id(v) in poses:  # outliers have no pose in the final solution
            proj = cv2.projectPoints(v.obj, *poses[id(v)], K, dist)[0].reshape(-1, 2)
            for x, y in proj:
                cv2.drawMarker(img, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.imwrite(str(debug_dir / f"{tag}_{v.index:06d}_{err:.2f}px.jpg"), img)
    # The view reaching furthest from the image centre shows distortion best.
    centre = K[:2, 2]
    far = max(views, key=lambda v: np.linalg.norm(v.img.reshape(-1, 2) - centre, axis=1).max())
    img = cv2.imdecode(np.frombuffer(far.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(debug_dir / f"undistorted_{far.index:06d}.jpg"), cv2.undistort(img, K, dist))
    cv2.imwrite(str(debug_dir / f"original_{far.index:06d}.jpg"), img)
    print(f"debug images in {debug_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    board, detector = load_board(args.board, args.square_length, args.marker_length)
    if args.input.is_dir():
        frames = bag_frames(args.input, args.topic, args.step)
    elif args.input.is_file():
        frames = video_frames(args.input, args.step)
    else:
        sys.exit(f"{args.input} does not exist")

    views, size = collect_views(frames, board, detector, args)
    views, outliers, res = calibrate_robust(views, size, args)
    report(views, outliers, res, size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, mtx=res["K"], dist=res["dist"], image_width=size[0],
             image_height=size[1], rms=res["rms"], std_intrinsics=res["std"][:9],
             n_views=len(views), square_length=board.getSquareLength(),
             marker_length=board.getMarkerLength())
    print(f"\nwrote {args.out.resolve()} (valid for {size[0]}x{size[1]} only)")
    if args.debug_dir:
        write_debug(args.debug_dir, views, outliers, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
