"""OpenArm cup policy bundles ("openarm_cup_policy/1", written by the training repo's
export_policy.py).

A bundle is one .pt file with the weights, training config, action statistics, camera
calibration and data-generation settings, shipped with its loader (openarm_policy.py +
tinydp_vision.py) in the same folder. That loader (OpenArmCupPolicy) reproduces the training
data path exactly, so this adapter only maps it onto the policy node's interface:

  preprocess  raw head-camera frame (raw_size, distorted) -> undistort with the bundled
              calibration -> crop -> the policy's image crop -> image_size, ImageNet norm
  predict     obs_horizon Observations (frame_dt apart, oldest first)
              -> {arm: (poses (H, 4, 4), gripper (H,))}, action_dt apart
  start_pose  the data's home joints (dataset_info.robot.home_joints)

The loader module is imported under its own name: the bundle's file is called
openarm_policy.py, like this ROS package.
"""

import glob
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from openarm_ee_control.geometry import matrix_from_xyz_wxyz
from openarm_ee_control.r2r2r_data import ARM_JOINT, StartPose

COLS = {"left": slice(0, 10), "right": slice(10, 20)}
LOADER = "openarm_policy.py"


def find_bundle(path: str) -> str | None:
    """The bundle file for `path` (a bundle .pt, or a folder holding one .pt next to the
    loader), or None if `path` is not a bundle."""
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    if not os.path.isfile(os.path.join(folder, LOADER)):
        return None
    if os.path.isfile(path):
        return path if path.endswith(".pt") else None
    files = sorted(glob.glob(os.path.join(path, "*.pt")))
    if len(files) != 1:
        raise ValueError(f"{path}: {len(files)} .pt files next to {LOADER}; pass the file")
    return files[0]


def _import_loader(folder: str):
    spec = importlib.util.spec_from_file_location("openarm_cup_policy_loader",
                                                  os.path.join(folder, LOADER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _xyz_wxyz(T: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([*T[:3, 3], w, x, y, z])


class CupBundlePolicy:
    def __init__(self, path: str, device: str = "cuda", ddim_steps: int | None = None):
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, running the policy on the CPU (slow)")
            device = "cpu"
        loader = _import_loader(os.path.dirname(os.path.abspath(path)))
        # None: the bundle's deployment default (DDIM, 10 steps); 0: training's DDPM chain
        steps = 10 if ddim_steps is None else (ddim_steps or None)
        self.p = loader.OpenArmCupPolicy(Path(path), device=device, inference_steps=steps)
        spec = self.p.spec()
        self.path = path
        self.arms = [self.p.arm]
        self.obs_horizon = self.p.obs_horizon
        self.action_horizon = self.p.horizon
        self.image_size = self.p.image_size
        self.frame_dt, self.action_dt = self.p.frame_dt, self.p.action_dt
        self.raw_size = tuple(int(v) for v in self.p.raw_size)
        self.calibration = (np.asarray(self.p.K_raw), np.asarray(self.p.dist))
        self._undistort_maps = cv2.initUndistortRectifyMap(
            self.calibration[0], self.calibration[1], None, self.calibration[0], self.raw_size,
            cv2.CV_16SC2)
        self.action_min = self.p.stats["min"].numpy()
        self.action_max = self.p.stats["max"].numpy()
        self.provenance = spec["provenance"]
        robot = spec["dataset_info"].get("robot", {})
        home = {n: float(v) for n, v in robot.get("home_joints", {}).items() if ARM_JOINT.match(n)}
        self.start_pose = (StartPose(joints=home, source=f"{os.path.basename(path)} home_joints")
                           if home else None)
        self.device = device

    def describe(self) -> str:
        prov = self.provenance
        return (f"{os.path.basename(self.path)} (epoch {prov.get('epoch')}, "
                f"{prov.get('training_rollouts')} rollouts, commit {prov.get('git_commit', '')[:8]}): "
                f"arm {self.arms[0]}, obs {self.obs_horizon} x {self.frame_dt * 1000:.1f} ms, "
                f"chunk {self.action_horizon} x {self.action_dt * 1000:.1f} ms, camera "
                f"{self.raw_size[0]}x{self.raw_size[1]} raw -> {self.image_size}px, "
                f"{self.p.inference_steps} denoising steps on {self.device}")

    def preprocess(self, rgb: np.ndarray):
        """The loader's preprocess(raw=True) with the undistortion maps built once: cv2.undistort
        rebuilds them on every call (0.15-1.1 s at 1920x1080); remap with the same CV_16SC2
        maps is bit-identical (cv2.undistort uses them internally) and takes a few ms."""
        if (rgb.shape[1], rgb.shape[0]) != self.raw_size:
            raise ValueError(f"frame is {rgb.shape[1]}x{rgb.shape[0]}, the raw camera frames "
                             f"are {self.raw_size[0]}x{self.raw_size[1]}")
        frame = cv2.remap(rgb, *self._undistort_maps, cv2.INTER_LINEAR)
        if self.p.crop:
            x0, y0, w, h = self.p.crop
            frame = frame[y0:y0 + h, x0:x0 + w]
        return self.p.transform(PILImage.fromarray(np.ascontiguousarray(frame)))

    def train_box(self, side: str):
        c = COLS[side]
        return self.action_min[c][:3], self.action_max[c][:3]

    def predict(self, observations: list) -> dict:
        left = np.stack([_xyz_wxyz(o.tcp["left"]) for o in observations])
        right = np.stack([_xyz_wxyz(o.tcp["right"]) for o in observations])
        grippers = np.array([[o.gripper["left"], o.gripper["right"]] for o in observations])
        out = self.p.predict([o.image for o in observations], left, right, grippers)
        poses = matrix_from_xyz_wxyz(np.concatenate([out["tcp_xyz"], out["tcp_wxyz"]], axis=1))
        return {out["arm"]: (poses, np.asarray(out["gripper"]))}
