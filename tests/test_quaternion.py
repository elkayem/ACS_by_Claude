import numpy as np
import pytest

from spacecraft_acs import quaternion as qt


def random_quat(rng):
    return qt.normalize(rng.standard_normal(4))


def test_multiply_identity():
    rng = np.random.default_rng(0)
    q = random_quat(rng)
    assert np.allclose(qt.multiply(qt.IDENTITY, q), q)
    assert np.allclose(qt.multiply(q, qt.IDENTITY), q)


def test_inverse_composition():
    rng = np.random.default_rng(1)
    q = random_quat(rng)
    assert np.allclose(qt.multiply(q, qt.inverse(q)), qt.IDENTITY, atol=1e-12)


def test_dcm_orthonormal_and_composition():
    rng = np.random.default_rng(2)
    q1, q2 = random_quat(rng), random_quat(rng)
    R1, R2 = qt.dcm(q1), qt.dcm(q2)
    assert np.allclose(R1 @ R1.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R1), 1.0)
    # q_AC = q_AB ⊗ q_BC maps A to C: dcm(q_AC) = dcm(q_BC) @ dcm(q_AB)
    q12 = qt.multiply(q1, q2)
    assert np.allclose(qt.dcm(q12), R2 @ R1, atol=1e-12)


def test_axis_angle_round_trip():
    axis = np.array([1.0, 2.0, -1.0])
    angle = 0.7
    q = qt.from_axis_angle(axis, angle)
    assert np.isclose(qt.rotation_angle(q), angle)
    # Rotation about z by 90 deg maps x-axis vector to [0, -1, 0] in body axes
    qz = qt.from_axis_angle([0, 0, 1], np.pi / 2)
    v_body = qt.dcm(qz) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v_body, [0.0, -1.0, 0.0], atol=1e-12)


def test_derivative_matches_finite_difference():
    rng = np.random.default_rng(3)
    q = random_quat(rng)
    omega = np.array([0.01, -0.02, 0.005])
    dt = 1e-6
    dq = qt.from_rotation_vector(omega * dt)
    q_next = qt.multiply(q, dq)
    qdot_fd = (q_next - q) / dt
    assert np.allclose(qt.derivative(q, omega), qdot_fd, atol=1e-8)


def test_error_shortest_path():
    q_cmd = qt.from_axis_angle([1, 0, 0], 0.1)
    q = qt.from_axis_angle([1, 0, 0], 0.3)
    q_err = qt.error(q_cmd, q)
    assert q_err[0] >= 0.0
    assert np.isclose(qt.rotation_angle(q_err), 0.2)
    # Sign flip on the input quaternion must not change the error
    q_err2 = qt.error(q_cmd, -q)
    assert np.allclose(q_err, q_err2)


def test_euler_round_trip():
    angles = np.array([0.1, -0.2, 0.3])  # roll, pitch, yaw
    q = qt.multiply(
        qt.multiply(
            qt.from_axis_angle([0, 0, 1], angles[2]),
            qt.from_axis_angle([0, 1, 0], angles[1]),
        ),
        qt.from_axis_angle([1, 0, 0], angles[0]),
    )
    assert np.allclose(qt.to_euler_zyx(q), angles, atol=1e-12)


def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        qt.normalize(np.zeros(4))
