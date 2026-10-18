"""Publish a V4L2 (USB) camera as sensor_msgs/Image, CompressedImage and CameraInfo.

Topics (relative, so the launch file's namespace applies, e.g. /camera/...):
  image_raw             sensor_msgs/Image, bgr8; decoded only while someone subscribes
  image_raw/compressed  sensor_msgs/CompressedImage, jpeg; with fourcc MJPG these are the
                        camera's own frames, passed through without re-encoding (record this
                        one for calibration bags: ~10x smaller than image_raw, no extra loss)
  camera_info           sensor_msgs/CameraInfo, only with calibration_file and only while the
                        stream has the calibrated resolution

The calibration file is the NPZ written by scripts/calibrate_intrinsics.py (`mtx`, `dist`,
`image_width`, `image_height`); NPZs from the SagresOverdose openarm_camera_info node (only
`mtx`, `dist`) work too and are assumed to match the stream's resolution.

Every message of one frame carries the same header: stamp = time the frame was read,
frame_id = `frame_id` (default camera_optical_frame: z forward, x right, y down).
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

DEFAULT_DEVICE = "/dev/v4l/by-id/usb-HBV_HD_CAMERA_HBV_HD_CAMERA-video-index0"


class CameraPublisher(Node):
    def __init__(self):
        super().__init__("camera_publisher")
        p = self.declare_parameter
        self.device = p("device", DEFAULT_DEVICE).value
        self.width = int(p("width", 1920).value)
        self.height = int(p("height", 1080).value)
        self.fps = float(p("fps", 30.0).value)
        self.fourcc = str(p("fourcc", "MJPG").value)
        self.frame_id = p("frame_id", "camera_optical_frame").value
        self.jpeg_quality = int(p("jpeg_quality", 95).value)  # only when re-encoding
        self.autofocus = bool(p("autofocus", False).value)
        self.focus = int(p("focus", -1).value)        # -1: leave as is
        self.exposure = int(p("exposure", -1).value)  # -1: auto
        calibration_file = p("calibration_file", "").value

        self.info = self.load_calibration(calibration_file) if calibration_file else None

        # Reliable publishers serve both reliable (ros2 bag) and best-effort subscribers.
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_raw = self.create_publisher(Image, "image_raw", qos)
        self.pub_jpeg = self.create_publisher(CompressedImage, "image_raw/compressed", qos)
        self.pub_info = self.create_publisher(CameraInfo, "camera_info", qos)

        self.cap = None
        self.passthrough = False
        self.stream_size = None
        self.size_warned = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ setup

    def load_calibration(self, path):
        path = Path(path).expanduser()
        with np.load(path, allow_pickle=False) as npz:
            K = np.asarray(npz["mtx"], np.float64).reshape(3, 3)
            dist = np.asarray(npz["dist"], np.float64).ravel()
            size = (int(npz["image_width"]), int(npz["image_height"])) \
                if "image_width" in npz else None
        info = CameraInfo()
        info.width, info.height = size if size else (0, 0)
        info.distortion_model = "plumb_bob" if dist.size <= 5 else "rational_polynomial"
        info.d = dist.tolist()
        info.k = K.ravel().tolist()
        info.r = np.eye(3).ravel().tolist()
        info.p = np.hstack([K, np.zeros((3, 1))]).ravel().tolist()
        self.get_logger().info(
            f"intrinsics from {path}: fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} "
            f"cy {K[1, 2]:.1f}" + (f", for {size[0]}x{size[1]}" if size else ""))
        return info

    def open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0) and not self.autofocus:
            self.get_logger().info("camera has no autofocus control (fixed focus?)")
        if self.focus >= 0:
            cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        if self.exposure >= 0:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual, 3 = aperture priority
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        # With MJPG and CONVERT_RGB off, read() returns the camera's JPEG bytes as one row.
        self.passthrough = self.fourcc == "MJPG" and cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        decoded = frame
        if self.passthrough:
            decoded = cv2.imdecode(frame, cv2.IMREAD_COLOR) \
                if frame.ndim == 2 and frame.shape[0] == 1 else None
            if decoded is None:
                self.get_logger().warning("MJPEG passthrough unavailable, re-encoding JPEGs")
                self.passthrough = False
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                ok, decoded = cap.read()
                if not ok:
                    cap.release()
                    return None
        h, w = decoded.shape[:2]
        self.stream_size = (w, h)
        got = cap.get(cv2.CAP_PROP_FOURCC)
        got = int(got).to_bytes(4, "little").decode(errors="replace")
        self.get_logger().info(
            f"opened {self.device}: {w}x{h} {got} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
            f"{', JPEG passthrough' if self.passthrough else ''}")
        if (w, h) != (self.width, self.height):
            self.get_logger().warning(f"asked for {self.width}x{self.height}, got {w}x{h}")
        return cap

    # ------------------------------------------------------------ capture loop

    def run(self):
        frames, t_report = 0, time.monotonic()
        while not self.stop.is_set() and rclpy.ok():
            if self.cap is None:
                self.cap = self.open()
                if self.cap is None:
                    self.get_logger().error(f"cannot open {self.device}, retrying",
                                            throttle_duration_sec=10.0)
                    time.sleep(1.0)
                    continue
            ok, frame = self.cap.read()
            stamp = self.get_clock().now().to_msg()
            if not ok:
                self.get_logger().error("frame grab failed, reopening the camera")
                self.cap.release()
                self.cap = None
                continue
            self.publish(frame, stamp)
            frames += 1
            if time.monotonic() - t_report > 30.0:
                rate = frames / (time.monotonic() - t_report)
                self.get_logger().info(f"{rate:.1f} fps")
                frames, t_report = 0, time.monotonic()
        if self.cap is not None:
            self.cap.release()

    def publish(self, frame, stamp):
        jpeg = frame.tobytes() if self.passthrough else None
        bgr = None if self.passthrough else frame
        want_raw = self.pub_raw.get_subscription_count() > 0
        want_jpeg = self.pub_jpeg.get_subscription_count() > 0
        if bgr is None and want_raw:
            bgr = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warning("undecodable JPEG frame", throttle_duration_sec=5.0)
                return
        if jpeg is None and want_jpeg:
            jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1]
            jpeg = jpeg.tobytes()

        if want_jpeg:
            msg = CompressedImage()
            msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
            msg.format = "jpeg"
            msg.data = jpeg
            self.pub_jpeg.publish(msg)
        if want_raw:
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
            msg.height, msg.width = bgr.shape[:2]
            msg.encoding, msg.step = "bgr8", bgr.shape[1] * 3
            msg.data = np.ascontiguousarray(bgr).tobytes()
            self.pub_raw.publish(msg)
        if self.info is not None:
            self.publish_info(stamp)

    def publish_info(self, stamp):
        w, h = self.stream_size
        if self.info.width and (w, h) != (self.info.width, self.info.height):
            if not self.size_warned:
                self.get_logger().error(
                    f"stream is {w}x{h} but the calibration is for {self.info.width}x"
                    f"{self.info.height}: not publishing camera_info")
                self.size_warned = True
            return
        self.info.header.stamp, self.info.header.frame_id = stamp, self.frame_id
        self.info.width, self.info.height = w, h
        self.pub_info.publish(self.info)

    def destroy_node(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
