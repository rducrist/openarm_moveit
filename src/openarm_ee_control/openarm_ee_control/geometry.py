"""Pose conversions shared by the replay tool and the policy node.

Conventions:
  * poses are 4x4 homogeneous matrices (numpy, float64)
  * R2R2R / IsaacLab store poses as [x, y, z, qw, qx, qy, qz] (scalar first)
  * ROS and scipy use quaternions [qx, qy, qz, qw] (scalar last)
  * rot6d is the first two ROWS of the rotation matrix, flattened (tinydp / yumi_realtime)
"""

import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation


def matrix_from_xyz_wxyz(xyz_wxyz: np.ndarray) -> np.ndarray:
    """(..., 7) [x, y, z, qw, qx, qy, qz] -> (..., 4, 4)."""
    a = np.asarray(xyz_wxyz, dtype=np.float64)
    T = np.zeros(a.shape[:-1] + (4, 4))
    T[..., :3, :3] = Rotation.from_quat(a[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(
        a.shape[:-1] + (3, 3))
    T[..., :3, 3] = a[..., :3]
    T[..., 3, 3] = 1.0
    return T


def matrix_from_pose(pose: Pose) -> np.ndarray:
    T = np.eye(4)
    q = pose.orientation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def matrix_from_transform(transform) -> np.ndarray:
    """geometry_msgs/Transform -> 4x4."""
    T = np.eye(4)
    q, t = transform.rotation, transform.translation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def pose_from_matrix(T: np.ndarray) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
        float(x), float(y), float(z), float(w))
    return p


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two rows."""
    R = np.asarray(R)
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def matrix_from_rot6d(r6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3), re-orthonormalising the two rows (Gram-Schmidt)."""
    r6 = np.asarray(r6, dtype=np.float64).reshape(np.shape(r6)[:-1] + (2, 3))
    a, b = r6[..., 0, :], r6[..., 1, :]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-2)


def vec10_from_pose(T: np.ndarray, gripper: float) -> np.ndarray:
    """4x4 + gripper -> [xyz, rot6d, gripper] (one arm of the 20-D policy vector)."""
    return np.concatenate([T[:3, 3], rot6d_from_matrix(T[:3, :3]), [gripper]])


def poses_from_vec10(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 10) [xyz, rot6d, gripper] -> ((N, 4, 4) poses, (N,) gripper values)."""
    v = np.asarray(v, dtype=np.float64)
    T = np.zeros(v.shape[:-1] + (4, 4))
    T[..., :3, :3] = matrix_from_rot6d(v[..., 3:9])
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T, v[..., 9]


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle [rad] of the relative rotation between two 3x3 rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))
