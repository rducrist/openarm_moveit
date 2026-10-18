"""Camera input and observation history for step-wise policy execution.

Each policy backend turns a camera frame into its model input (`policy.preprocess(rgb)`) and
builds its proprio from the TCP poses and gripper commands in an Observation.

The policies train on histories of `seq_length` frames spaced like the predicted actions:
one history step == one action step. When the robot executes k waypoints per step and then
stops, the history therefore gets one entry per executed waypoint, taken from the frames
recorded during the move (the frame where the arm was closest to that waypoint), and the
last entry is a fresh frame after the arm settled. With seq_length = 1 only that fresh frame
is used.
"""

import threading
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image

_CHANNELS = {"rgb8": (3, False), "bgr8": (3, True), "rgba8": (4, False), "bgra8": (4, True)}


def image_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (rgb8/bgr8/rgba8/bgra8) -> (H, W, 3) uint8 RGB."""
    if msg.encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    ch, bgr = _CHANNELS[msg.encoding]
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    a = a[:, :msg.width * ch].reshape(msg.height, msg.width, ch)[..., :3]
    return np.ascontiguousarray(a[..., ::-1] if bgr else a)


def compressed_to_rgb(msg: CompressedImage) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("undecodable compressed image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_like_training(rgb: np.ndarray, size: int) -> np.ndarray:
    """Stretch to size x size with PIL bilinear, as tinydp's Resize((S, S)) on the JPGs."""
    return np.asarray(PILImage.fromarray(rgb).resize((size, size), PILImage.BILINEAR))


@dataclass
class Frame:
    stamp: Time              # header stamp (for TF lookups)
    image: object            # policy.preprocess() output
    gripper: dict            # commanded gripper state per arm at capture time (True = closed)


@dataclass
class Observation:
    image: object            # policy.preprocess() output
    tcp: dict                # {side: 4x4 TCP pose in the policy frame}
    gripper: dict            # {side: 1.0 closed / 0.0 open}, the last command


class Camera:
    """Keeps the newest image message; preprocesses frames on demand, and records
    preprocessed frames (at most one per `record_period` s) between start/stop_recording().

    Topics ending in /compressed are read as sensor_msgs/CompressedImage, others as Image.
    Freshness uses the arrival time, so replayed or offset camera clocks work too."""

    def __init__(self, node: Node, topic: str, preprocess, gripper_state,
                 record_period: float = 0.0):
        self.node = node
        self._preprocess = preprocess          # rgb (H, W, 3) uint8 -> model input
        self._gripper_state = gripper_state    # callable -> {side: bool}
        self._record_period = record_period
        self._cv = threading.Condition()
        self._latest = None                    # (msg, arrival Time)
        self._recording = None
        self._last_recorded = None
        self.shape = None                      # (height, width, encoding/format)
        if topic.endswith("/compressed"):
            self._to_rgb = compressed_to_rgb
            node.create_subscription(CompressedImage, topic, self._on_image,
                                     qos_profile_sensor_data)
        else:
            self._to_rgb = image_to_rgb
            node.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)

    def _frame(self, msg) -> Frame:
        rgb = self._to_rgb(msg)
        return Frame(Time.from_msg(msg.header.stamp), self._preprocess(rgb),
                     dict(self._gripper_state()))

    def _on_image(self, msg):
        arrival = self.node.get_clock().now()
        with self._cv:
            self._latest = (msg, arrival)
            self._cv.notify_all()
            record = self._recording is not None and (
                self._last_recorded is None
                or (arrival - self._last_recorded).nanoseconds * 1e-9 >= self._record_period)
            if record:
                self._last_recorded = arrival
        if record:
            frame = self._frame(msg)
            with self._cv:
                if self._recording is not None:
                    self._recording.append(frame)

    def raw_shape(self, timeout: float = 10.0):
        """(width, height) of the newest frame, or None if nothing arrived."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._latest is not None, timeout):
                return None
            msg = self._latest[0]
        h, w = self._to_rgb(msg).shape[:2]
        return w, h

    def wait_frame(self, newer_than: Time | None = None, timeout: float = 5.0) -> Frame | None:
        """Newest frame, waiting for one that arrived after `newer_than` if given."""
        def ready():
            return self._latest is not None and (newer_than is None or self._latest[1] > newer_than)
        with self._cv:
            if not self._cv.wait_for(ready, timeout):
                return None
            msg = self._latest[0]
        return self._frame(msg)

    def start_recording(self):
        with self._cv:
            self._recording = deque(maxlen=600)
            self._last_recorded = None

    def stop_recording(self) -> list:
        with self._cv:
            frames, self._recording = list(self._recording or []), None
        return frames


class History:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.buf = deque(maxlen=horizon)

    def reset(self, obs: Observation):
        """Start of an episode: the robot stood still, so the history is the same frame."""
        self.buf.clear()
        self.buf.extend([obs] * self.horizon)

    def extend(self, observations):
        self.buf.extend(observations)

    def observations(self) -> list:
        """Oldest first."""
        return list(self.buf)


def pick_history(recorded: list, positions: dict, final: Observation, horizon: int,
                 to_observation) -> list:
    """History entries for the waypoints just executed.

    recorded:  Frames captured during the move (time order)
    positions: {side: (k, 3)} executed waypoint positions of the commanded arms
    final:     Observation after the arm settled (stands for the last waypoint)
    to_observation: Frame -> Observation
    Returns min(k, horizon) Observations, oldest first. Without recorded frames the
    intermediate entries fall back to `final`.
    """
    k = len(next(iter(positions.values())))
    n = min(k, horizon)
    if n <= 1 or not recorded:
        return [final] * n
    candidates = [to_observation(f) for f in recorded]
    entries, start = [], 0
    for j in range(k - n, k - 1):        # waypoints whose entries come before `final`
        dist = [sum(float(np.linalg.norm(c.tcp[s][:3, 3] - positions[s][j])) for s in positions)
                for c in candidates[start:]]
        start += int(np.argmin(dist))    # monotonic in time
        entries.append(candidates[start])
    return entries + [final]
