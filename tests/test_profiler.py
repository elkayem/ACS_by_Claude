import numpy as np
import pytest

from spacecraft_acs import quaternion as qt
from spacecraft_acs.config import (
    GuidanceConfig,
    ProfilerConfig,
    StepConfig,
    GEO_ORBIT_RATE,
)
from spacecraft_acs.guidance import Guidance
from spacecraft_acs.profiler import SlewProfile

from test_guidance_controller import assert_kinematically_consistent


def test_profile_reaches_target_and_respects_limits():
    theta_f, vmax, amax = np.deg2rad(10.0), np.deg2rad(0.05), np.deg2rad(0.001)
    p = SlewProfile(theta_f, vmax, amax)
    t = np.linspace(-10.0, p.duration + 10.0, 20000)
    vals = np.array([p.evaluate(ti) for ti in t])
    theta, rate, accel = vals.T
    assert np.isclose(theta[-1], theta_f)
    assert np.all(rate <= vmax * (1 + 1e-9))
    assert np.all(np.abs(accel) <= amax * (1 + 1e-9))
    assert np.all(np.diff(theta) >= -1e-15)  # monotonic
    # 10 deg at these limits needs a cruise segment
    assert p.t_cruise > 0.0
    assert np.isclose(np.max(rate), vmax)


def test_profile_smoothness():
    """Rate and acceleration are continuous: finite differences of theta and
    rate converge to the analytic rate and acceleration everywhere."""
    p = SlewProfile(np.deg2rad(1.0), np.deg2rad(0.05), np.deg2rad(0.001))
    dt = 1e-4
    for t in np.linspace(-1.0, p.duration + 1.0, 400):
        th_m, r_m, _ = p.evaluate(t - dt)
        th_0, r_0, a_0 = p.evaluate(t)
        th_p, r_p, _ = p.evaluate(t + dt)
        assert np.isclose((th_p - th_m) / (2 * dt), r_0, atol=1e-8)
        assert np.isclose((r_p - r_m) / (2 * dt), a_0, atol=1e-8)


def test_short_slew_has_no_cruise():
    theta_f, vmax, amax = np.deg2rad(1.0), np.deg2rad(0.05), np.deg2rad(0.001)
    p = SlewProfile(theta_f, vmax, amax)
    assert p.t_cruise == 0.0
    assert p.v_peak < vmax
    assert np.isclose(p.v_peak, np.sqrt(0.5 * theta_f * amax))
    # duration = 2*Ta = 4*v_peak/amax
    assert np.isclose(p.duration, 4.0 * p.v_peak / amax)


def test_zero_angle_profile():
    p = SlewProfile(0.0, 1.0, 1.0)
    assert p.duration == 0.0
    assert p.evaluate(5.0) == (0.0, 0.0, 0.0)


def make_profiled_guidance(angle_deg=1.0, axis=(1.0, 0.0, 0.0), time_s=100.0):
    return Guidance(
        GuidanceConfig(
            step=StepConfig(axis=list(axis), angle_deg=angle_deg, time_s=time_s),
            profiler=ProfilerConfig(enabled=True, max_rate_dps=0.05, max_accel_dps2=0.001),
        ),
        GEO_ORBIT_RATE,
    )


def test_profiled_guidance_kinematic_consistency():
    g = make_profiled_guidance()
    # Before, during accel, during decel, and after the slew. atol sits just
    # above the O(dt^2) central-difference truncation floor set by the
    # profile's curvature.
    for t in [50.0, 120.0, 160.0, 100.0 + g.slew_duration + 20.0]:
        assert_kinematically_consistent(g, t, dt=0.05, atol=1e-9)


def test_profiled_guidance_reaches_step_attitude():
    """After the slew completes, the profiled command equals the raw-step
    command exactly."""
    g_prof = make_profiled_guidance()
    g_step = Guidance(
        GuidanceConfig(step=StepConfig(axis=[1.0, 0, 0], angle_deg=1.0, time_s=100.0)),
        GEO_ORBIT_RATE,
    )
    t_end = 100.0 + g_prof.slew_duration + 1.0
    q_p, w_p, a_p = g_prof.command(t_end)
    q_s, w_s, a_s = g_step.command(t_end)
    assert np.allclose(q_p, q_s, atol=1e-12)
    assert np.allclose(w_p, w_s, atol=1e-15)
    assert np.allclose(a_p, a_s, atol=1e-15)


def test_negative_angle_slew():
    g = make_profiled_guidance(angle_deg=-2.0)
    t_end = 100.0 + g.slew_duration + 1.0
    q_cmd, _, _ = g.command(t_end)
    q_base = g.lvlh_attitude(t_end)
    q_off = qt.error(q_base, q_cmd)
    # error() forces the scalar part positive; the -2 deg rotation about +x
    # then shows up as a negative x vector component
    assert np.isclose(np.rad2deg(qt.rotation_angle(q_off)), 2.0)
    assert q_off[1] < 0.0


def test_profiler_config_validation():
    with pytest.raises(ValueError):
        ProfilerConfig(enabled=True, max_rate_dps=-1.0)
