"""Read R2R2R episodes and policy start poses.

Episode, as written by R2R2R's BatchDataLogger (env_*/robot_data/):
  robot_data.h5     ee_poses (N, 14)  [x y z qw qx qy qz] left, then right, robot root frame
                    joint_angles (N, J), gripper_binary_cmd (N, 2) with 1 = closed
  joint_names.txt   names of the joint_angles columns

Start pose file (YAML), e.g. <ckpt-dir>/start_pose.yaml:
  joints: {openarm_right_joint1: 0.12, ...}     # preferred: reproduces the sim's arm posture
  tcp: {right: [x, y, z, qw, qx, qy, qz], ...}  # used for arms without joint values
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
import yaml

from openarm_ee_control.geometry import matrix_from_xyz_wxyz

SIDES = ("left", "right")
ARM_JOINT = re.compile(r"^openarm_(left|right)_joint[1-7]$")


@dataclass
class StartPose:
    joints: dict = field(default_factory=dict)   # arm joint name -> position [rad]
    tcp: dict = field(default_factory=dict)      # side -> [x, y, z, qw, qx, qy, qz]
    source: str = ""

    def arm_joints(self, side: str) -> dict:
        return {n: v for n, v in self.joints.items() if n.startswith(f"openarm_{side}_")}

    def describe(self) -> str:
        parts = []
        for s in SIDES:
            if self.arm_joints(s):
                parts.append(f"{s}: joints")
            elif s in self.tcp:
                parts.append(f"{s}: TCP pose")
        return f"{self.source} ({', '.join(parts) or 'empty'})"


def find_h5(path: str) -> str:
    if os.path.isfile(path):
        return path
    for candidate in (os.path.join(path, "robot_data", "robot_data.h5"),
                      os.path.join(path, "robot_data.h5")):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"no robot_data.h5 under {path}")


def load_episode(path: str, gripper_cols=(0, 1)) -> dict:
    """{side: (poses (N, 4, 4), closed (N,) bool)} in the episode's root frame."""
    import h5py
    with h5py.File(find_h5(path), "r") as f:
        ee = np.asarray(f["ee_poses"])
        grip = np.asarray(f["gripper_binary_cmd"]) if "gripper_binary_cmd" in f else None
    if ee.ndim != 2 or ee.shape[1] != 14:
        raise ValueError(f"ee_poses has shape {ee.shape}, expected (N, 14)")
    if grip is None:
        grip = np.zeros((len(ee), 2))
    return {side: (matrix_from_xyz_wxyz(ee[:, 7 * k:7 * k + 7]),
                   grip[:, gripper_cols[k]].astype(float) > 0.5)
            for k, side in enumerate(SIDES)}


def start_from_episode(path: str, index: int = 0) -> StartPose:
    """Arm joint angles (by joint_names.txt) and TCP poses of one logged frame."""
    import h5py
    h5 = find_h5(path)
    start = StartPose(source=f"{path} [frame {index}]")
    with h5py.File(h5, "r") as f:
        ee = np.asarray(f["ee_poses"][index])
        angles = np.asarray(f["joint_angles"][index]) if "joint_angles" in f else None
    start.tcp = {s: [float(v) for v in ee[7 * k:7 * k + 7]] for k, s in enumerate(SIDES)}
    names_file = os.path.join(os.path.dirname(h5), "joint_names.txt")
    if angles is not None and os.path.isfile(names_file):
        with open(names_file, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        start.joints = {n: float(a) for n, a in zip(names, angles) if ARM_JOINT.match(n)}
    return start


def load_start(path: str) -> StartPose:
    """Start pose from a YAML file or an episode (frame 0)."""
    if path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        joints = {str(n): float(v) for n, v in (d.get("joints") or {}).items()}
        bad = [n for n in joints if not ARM_JOINT.match(n)]
        if bad:
            raise ValueError(f"{path}: not OpenArm arm joints: {bad}")
        tcp = {s: [float(v) for v in p] for s, p in (d.get("tcp") or {}).items()}
        return StartPose(joints, tcp, path)
    return start_from_episode(path)


def save_start(start: StartPose, path: str):
    d = {"source": start.source,
         "joints": {n: round(v, 6) for n, v in sorted(start.joints.items())},
         "tcp": {s: [round(v, 6) for v in p] for s, p in start.tcp.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Policy start pose (world frame), written by traj_replay --save-start.\n"
                "# joints [rad] are used where present; tcp [x y z qw qx qy qz] otherwise.\n")
        yaml.safe_dump(d, f, sort_keys=False)
