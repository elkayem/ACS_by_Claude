"""End-to-end simulation tests, including the cross-domain consistency check:
the nonlinear time-domain simulation must agree with the stability boundary
predicted by the linear frequency-domain analysis."""

from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import linearize, simulate
from spacecraft_acs.controller import resolve_gains

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_default():
    return cfg_mod.load(DEFAULT_YAML)


def test_step_response_end_to_end():
    cfg = load_default()
    # The all-angle robust design runs ~7 mHz roll/yaw bandwidth and a raw
    # step rings the slosh, so settling takes several hundred seconds
    cfg.simulation.duration_s = 1500.0
    cfg.guidance.step.time_s = 50.0
    result = simulate.run(cfg)

    metrics = simulate.step_metrics(result)
    assert metrics.settling_time_s is not None, "step did not settle"
    # A raw 1-deg step saturates the wheels hard with the default gains, so
    # substantial saturation-driven overshoot (~50%) is expected physics; the
    # profiled slew (acs compare) is the operational alternative. This bound
    # only guards against gross regression.
    assert metrics.overshoot_pct < 70.0
    # Wheel torque respects the saturation limit
    assert np.max(np.abs(result.torque_applied)) <= cfg.wheels.max_torque + 1e-12
    # Momentum bookkeeping: dh_w/dt = -T_applied under ZOH
    dt = result.t[1] - result.t[0]
    h_pred = result.h_wheel[0] - np.cumsum(result.torque_applied[:-1], axis=0) * dt
    assert np.allclose(result.h_wheel[1:], h_pred, atol=1e-6)
    # Flexible modes were excited by the maneuver
    assert np.max(np.abs(result.eta[:, 0])) > 1e-4


def _sim_max_error(cfg) -> float:
    """Max attitude error over the run, with a small initial-condition kick
    delivered through a brief command offset (no step, clean environment)."""
    result = simulate.run(cfg)
    err = result.att_err_deg
    if not np.all(np.isfinite(err)):
        return float("inf")  # diverged past floating-point range
    return float(np.max(np.abs(err)))


def _clean_config(gain_scale: float):
    """Idealized config (no noise/saturation/disturbances) with all controller
    gains scaled by gain_scale, which scales the open loop L(s) exactly."""
    cfg = load_default()
    cfg.wheels.ideal = True
    cfg.sensors.perfect = True
    cfg.environment.gravity_gradient = False
    cfg.environment.srp.enabled = False
    kp, ki, kd = resolve_gains(cfg.controller, np.diag(cfg.spacecraft.inertia))
    cfg.controller.kp = kp * gain_scale
    cfg.controller.ki = ki * gain_scale
    cfg.controller.kd = kd * gain_scale
    cfg.guidance.step.angle_deg = 0.01  # small kick to excite the loop
    cfg.guidance.step.time_s = 10.0
    cfg.simulation.duration_s = 900.0
    return cfg


@pytest.mark.slow
def test_gain_margin_consistency_with_time_domain():
    """The linear analysis predicts ~7-8 dB worst-axis gain margin. The
    nonlinear sim must be stable well below it and unstable above."""
    gm_db = min(d.gm_db for d in linearize.analyze(load_default()))
    gm_factor = 10.0 ** (gm_db / 20.0)
    assert 1.8 < gm_factor < 3.2  # sanity: all-angle robust design ~6-10 dB

    err_stable = _sim_max_error(_clean_config(0.6 * gm_factor))
    err_unstable = _sim_max_error(_clean_config(1.4 * gm_factor))
    assert err_stable < 0.05, f"stable case diverged: {err_stable} deg"
    assert err_unstable > 0.5, f"unstable case did not diverge: {err_unstable} deg"
