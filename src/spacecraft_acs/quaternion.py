"""Quaternion utilities.

Convention: scalar-first, q = [w, x, y, z]. A quaternion q_AB rotates frame B
into frame A, i.e. it represents the attitude of frame B with respect to frame
A, and ``dcm(q_AB) @ v_A`` expresses an A-frame vector in B axes.
Composition: q_AC = multiply(q_AB, q_BC).
"""

from __future__ import annotations

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("cannot normalize zero quaternion")
    return q / n


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def inverse(q: np.ndarray) -> np.ndarray:
    return conjugate(q) / np.dot(q, q)


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        raise ValueError("rotation axis must be nonzero")
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], np.sin(half) * axis / n))


def from_rotation_vector(rv: np.ndarray) -> np.ndarray:
    """Quaternion from a rotation vector (axis * angle, radians)."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return normalize(np.concatenate(([1.0], 0.5 * np.asarray(rv, dtype=float))))
    return from_axis_angle(rv, angle)


def dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix: for q_AB, maps A-frame vectors to B axes."""
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), w * w - x * x + y * y - z * z, 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), w * w - x * x - y * y + z * z],
        ]
    )


def derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Kinematic rate q̇ = ½ q ⊗ [0, ω], ω in the rotated (body) frame."""
    return 0.5 * multiply(q, np.concatenate(([0.0], omega)))


def error(q_cmd: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Error quaternion q_err = q_cmd⁻¹ ⊗ q, forced to the shortest path."""
    q_err = multiply(conjugate(normalize(q_cmd)), normalize(q))
    if q_err[0] < 0.0:
        q_err = -q_err
    return q_err


def rotation_angle(q: np.ndarray) -> float:
    """Total rotation angle of the quaternion, in radians (always >= 0)."""
    q = normalize(q)
    return 2.0 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0]))


def to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """3-2-1 (yaw-pitch-roll) Euler angles [roll, pitch, yaw] in radians."""
    w, x, y, z = normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])
