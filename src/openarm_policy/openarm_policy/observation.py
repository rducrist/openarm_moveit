"""Camera input and observation history for step-wise policy execution.

tinydp trains on histories of `seq_length` frames spaced `subsample_steps` logged frames apart,
which is also the spacing of the predicted actions: one history step == one action step. When
the robot executes k waypoints per step and then stops, the history therefore gets one entry
per executed waypoint, taken from the frames recorded during the move (the frame where the arm
was closest to that waypoint), and the last entry is a fresh frame after the arm settled.
With seq_length = 1 only that fresh frame is used.
"""

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image

_CHANNELS = {"rgb8": (3, False), "bgr8": (3, True), "rgba8": (4, False), "bgra8": (4, True)}


def image_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (rgb8/bgr8/rgba8/bgra8) -> (H, W, 3) uint8 RGB."""
    if msg.encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    ch, bgr = _CHANNELS[msg.encoding]
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    a = a[:, :msg.width * ch].reshape(msg.height, msg.width, ch)[..., :3]
    return np.ascontiguousarray(a[..., ::-1] if bgr else a)


def resize_like_training(rgb: np.ndarray, size: int) -> np.ndarray:
    """Stretch to size x size with PIL bilinear, as tinydp's Resize((S, S)) on the JPGs."""
    return np.asarray(PILImage.fromarray(rgb).resize((size, size), PILImage.BILINEAR))


@dataclass
class Frame:
    stamp: Time
    image: np.ndarray        # (S, S, 3) uint8
    gripper: dict            # commanded gripper state per arm at capture time


@dataclass
class Observation:
    image: np.ndarray        # (S, S, 3) uint8
    proprio: np.ndarray      # (20,)


class Camera:
    """Keeps the newest image; records resized frames between start/stop_recording()."""

    def __init__(self, node: Node, topic: str, size: int, gripper_state):
        self.size = size
        self._gripper_state = gripper_state   # callable -> {side: bool}
        self._cv = threading.Condition()
        self._latest = None
        self._recording = None
        self.shape = None
        node.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)

    def _on_image(self, msg: Image):
        with self._cv:
            self._latest = msg
            self.shape = (msg.height, msg.width, msg.encoding)
            recording = self._recording is not None
            self._cv.notify_all()
        if recording:
            frame = Frame(Time.from_msg(msg.header.stamp),
                          resize_like_training(image_to_rgb(msg), self.size),
                          dict(self._gripper_state()))
            with self._cv:
                if self._recording is not None:
                    self._recording.append(frame)

    def wait_frame(self, newer_than: Time | None = None, timeout: float = 5.0) -> Frame | None:
        """Newest frame, waiting for one stamped after `newer_than` if given."""
        def ready():
            if self._latest is None:
                return False
            return newer_than is None or Time.from_msg(self._latest.header.stamp) > newer_than
        with self._cv:
            if not self._cv.wait_for(ready, timeout):
                return None
            msg = self._latest
        return Frame(Time.from_msg(msg.header.stamp),
                     resize_like_training(image_to_rgb(msg), self.size),
                     dict(self._gripper_state()))

    def start_recording(self):
        with self._cv:
            self._recording = deque(maxlen=600)

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

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(images (T, S, S, 3), proprio (T, 20)), oldest first."""
        return (np.stack([o.image for o in self.buf]),
                np.stack([o.proprio for o in self.buf]).astype(np.float32))


def pick_history(recorded: list, positions: dict, final: Observation, horizon: int,
                 to_observation) -> list:
    """History entries for the waypoints just executed.

    recorded:  Frames captured during the move (time order)
    positions: {side: (k, 3)} executed waypoint positions of the commanded arms
    final:     Observation after the arm settled (stands for the last waypoint)
    to_observation: Frame -> RecordedObservation
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
        dist = [sum(float(np.linalg.norm(c.tcp[s] - positions[s][j])) for s in positions)
                for c in candidates[start:]]
        start += int(np.argmin(dist))    # monotonic in time
        entries.append(candidates[start].obs)
    return entries + [final]


@dataclass
class RecordedObservation:
    obs: Observation
    tcp: dict                # {side: (3,)} TCP positions at the frame's stamp
