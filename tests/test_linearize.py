import control
import numpy as np

from spacecraft_acs import linearize
from spacecraft_acs.config import Config, ModeConfig, SpacecraftConfig


def make_config(modes):
    return Config(
        spacecraft=SpacecraftConfig(inertia=np.diag([8000.0, 4500.0, 6500.0]), modes=modes)
    )


def test_rigid_plant_is_double_integrator():
    cfg = make_config([])
    g = linearize.plant_ss(cfg, 0)
    w = np.array([0.01, 0.1, 1.0])
    resp = control.tf(g)(1j * w)
    expected = 1.0 / (8000.0 * (1j * w) ** 2)
    assert np.allclose(resp, expected, rtol=1e-10)


def test_plant_poles_match_coupled_mode_frequency():
    """Flexible plant poles: rigid double pole + coupled (free-free) mode."""
    l, f = 45.0, 0.10
    cfg = make_config([ModeConfig(freq_hz=f, damping=0.0, participation=[l, 0, 0])])
    g = linearize.plant_ss(cfg, 0)
    poles = g.poles()
    flex = sorted(abs(p.imag) for p in poles if abs(p.imag) > 1e-9)
    expected = 2 * np.pi * f * np.sqrt(8000.0 / (8000.0 - l**2))
    assert np.isclose(flex[-1], expected, rtol=1e-9)


def test_plant_zeros_at_cantilever_frequency():
    """Collocated torque->attitude zeros sit at the cantilever mode frequency,
    below the coupled pole (alternating pole-zero pattern)."""
    l, f = 45.0, 0.10
    cfg = make_config([ModeConfig(freq_hz=f, damping=0.0, participation=[l, 0, 0])])
    g = linearize.plant_ss(cfg, 0)
    zeros = control.zeros(g)
    assert np.isclose(max(abs(z.imag) for z in zeros), 2 * np.pi * f, rtol=1e-9)


def test_default_config_margins_positive_and_stable():
    from spacecraft_acs import config as cfg_mod
    from pathlib import Path

    c = cfg_mod.load(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    for d in linearize.analyze(c):
        assert d.pm_deg is not None and d.pm_deg > 20.0
        assert d.gm_db is None or d.gm_db > 4.0
        assert all(p.real < 1e-9 for p in d.cl_poles), f"axis {d.axis} unstable"


def test_unfiltered_rigid_pd_margins_match_theory():
    """PD on 1/(Js²) with no filters/delay: PM = atan(2ζ·r)... verified via
    the known analytic phase of L(jw) = (Kd s + Kp)/(J s²)."""
    cfg = make_config([])
    cfg.controller.rate_hz = 1000.0  # make ZOH delay negligible
    cfg.controller.design.integral_time_factor = 0.0  # pure PD
    d = linearize.analyze_axis(cfg, 0)
    # L(jw) = (Kp + j w Kd) / (-J w^2); |L|=1 at wc; PM = atan(wc Kd / Kp)
    kp = 8000.0 * (2 * np.pi * 0.02) ** 2
    kd = 2 * 0.7 * 8000.0 * (2 * np.pi * 0.02)
    wc_pred = None
    # solve |Kp + j w Kd| = J w^2 numerically
    from scipy.optimize import brentq

    f = lambda w: np.hypot(kp, w * kd) - 8000.0 * w**2
    wc_pred = brentq(f, 1e-3, 10.0)
    pm_pred = np.rad2deg(np.arctan2(wc_pred * kd, kp))
    assert np.isclose(d.gain_crossover_hz, wc_pred / (2 * np.pi), rtol=1e-3)
    assert np.isclose(d.pm_deg, pm_pred, atol=0.5)
