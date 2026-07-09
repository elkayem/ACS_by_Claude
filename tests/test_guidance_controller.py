import numpy as np

from spacecraft_acs import quaternion as qt
from spacecraft_acs.config import (
    ControllerConfig,
    FilterConfig,
    GainDesignConfig,
    GuidanceConfig,
    StepConfig,
    GEO_ORBIT_RATE,
)
from spacecraft_acs.controller import QuaternionPID, design_gains
from spacecraft_acs.guidance import Guidance

J_DIAG = np.array([8000.0, 4500.0, 6500.0])


def make_pid(kp=None, ki=None, kd=None, filters=(), rate_hz=4.0):
    cfg = ControllerConfig(rate_hz=rate_hz, kp=kp, ki=ki, kd=kd, filters=list(filters))
    return QuaternionPID(cfg, J_DIAG)


def assert_kinematically_consistent(g, t, dt=0.5, atol=1e-12):
    """Central-difference q̇_cmd must equal ½ q_cmd ⊗ [0, ω_cmd], and
    central-difference ω̇_cmd must equal α_cmd."""
    q0, w0, a0 = g.command(t)
    q_m, w_m, _ = g.command(t - dt)
    q_p, w_p, _ = g.command(t + dt)
    assert np.allclose((q_p - q_m) / (2.0 * dt), qt.derivative(q0, w0), atol=atol)
    assert np.allclose((w_p - w_m) / (2.0 * dt), a0, atol=atol)


def test_nadir_command_kinematic_consistency():
    g = Guidance(GuidanceConfig(step=StepConfig(time_s=1e9)), GEO_ORBIT_RATE)
    assert_kinematically_consistent(g, 5000.0)
    # After a quarter GEO orbit the pitch attitude has advanced by 90 deg
    quarter = 0.5 * np.pi / GEO_ORBIT_RATE
    q_quarter, _, _ = g.command(quarter)
    assert np.isclose(qt.rotation_angle(q_quarter), np.pi / 2)


def test_step_command_offset():
    step = StepConfig(axis=[1.0, 0, 0], angle_deg=10.0, time_s=100.0)
    g = Guidance(GuidanceConfig(step=step), GEO_ORBIT_RATE)
    q_before, _, _ = g.command(99.9)
    q_after, _, _ = g.command(100.0)
    q_delta = qt.error(q_before, q_after)
    assert np.isclose(np.rad2deg(qt.rotation_angle(q_delta)), 10.0)
    # Kinematic consistency must also hold after the step
    assert_kinematically_consistent(g, 101.0)


def test_proportional_action():
    pid = make_pid(kp=[100.0, 100, 100], ki=[0.0, 0, 0], kd=[0.0, 0, 0])
    angle = 0.01
    q = qt.from_axis_angle([1, 0, 0], angle)
    u = pid.step(q, np.zeros(3), qt.IDENTITY, np.zeros(3))
    # theta ≈ 2*sin(angle/2) ≈ angle; torque opposes the error
    assert np.isclose(u[0], -100.0 * angle, rtol=1e-4)
    assert np.allclose(u[1:], 0.0)


def test_integral_action_and_freeze():
    pid = make_pid(kp=[0.0, 0, 0], ki=[10.0, 10, 10], kd=[0.0, 0, 0], rate_hz=4.0)
    q = qt.from_axis_angle([1, 0, 0], 0.01)
    u1 = pid.step(q, np.zeros(3), qt.IDENTITY, np.zeros(3))
    u2 = pid.step(q, np.zeros(3), qt.IDENTITY, np.zeros(3))
    theta = 2.0 * np.sin(0.005)
    assert np.isclose(u1[0], -10.0 * theta * 0.25, rtol=1e-6)
    assert np.isclose(u2[0], -10.0 * theta * 0.5, rtol=1e-6)
    u3 = pid.step(q, np.zeros(3), qt.IDENTITY, np.zeros(3), freeze_integrator=True)
    assert np.isclose(u3[0], u2[0])  # integrator held


def test_derivative_action_tracks_command_rate():
    pid = make_pid(kp=[0.0, 0, 0], ki=[0.0, 0, 0], kd=[50.0, 50, 50])
    w_cmd = np.array([0.0, 1e-3, 0.0])
    # Body rotating exactly at the command rate -> zero torque
    u = pid.step(qt.IDENTITY, w_cmd, qt.IDENTITY, w_cmd)
    assert np.allclose(u, 0.0, atol=1e-15)
    u = pid.step(qt.IDENTITY, w_cmd + [1e-4, 0, 0], qt.IDENTITY, w_cmd)
    assert np.isclose(u[0], -50.0 * 1e-4)


def test_lowpass_filter_dc_gain_and_attenuation():
    lp = FilterConfig(type="lowpass", freq_hz=0.05, damping=0.7)
    pid = make_pid(kp=[1.0, 1, 1], ki=[0.0, 0, 0], kd=[0.0, 0, 0], filters=[lp], rate_hz=4.0)
    q = qt.from_axis_angle([1, 0, 0], 0.01)
    u = None
    for _ in range(2000):  # 500 s, settle to DC
        u = pid.step(q, np.zeros(3), qt.IDENTITY, np.zeros(3))
    assert np.isclose(u[0], -2.0 * np.sin(0.005), rtol=1e-3)  # DC gain of 1


def test_notch_attenuates_center_frequency():
    f0 = 0.1
    notch = FilterConfig(type="notch", freq_hz=f0, damping=0.5, depth_db=20.0)
    pid = make_pid(kp=[1.0, 1, 1], ki=[0.0, 0, 0], kd=[0.0, 0, 0], filters=[notch], rate_hz=4.0)
    dt = 0.25
    out = []
    amp = 0.01
    for k in range(4000):
        angle = amp * np.sin(2 * np.pi * f0 * k * dt)
        q = qt.from_axis_angle([1, 0, 0], angle)
        u = pid.step(q, np.zeros(3), qt.IDENTITY, np.zeros(3))
        out.append(u[0])
    steady = np.array(out[2000:])
    attenuation = np.max(np.abs(steady)) / amp
    assert attenuation < 10 ** (-20.0 / 20.0) * 1.5  # ~ -20 dB at center


def test_design_gains_rule():
    d = GainDesignConfig(bandwidth_hz=0.02, damping=0.7, integral_time_factor=10.0)
    kp, ki, kd = design_gains(J_DIAG, d)
    wn = 2 * np.pi * 0.02
    assert np.allclose(kp, J_DIAG * wn**2)
    assert np.allclose(kd, 2 * 0.7 * J_DIAG * wn)
    assert np.allclose(ki, kp * wn / 10.0)
