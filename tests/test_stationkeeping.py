from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import simulate
from spacecraft_acs.config import PhasePlaneConfig
from spacecraft_acs.stationkeeping import BurnController, PhasePlane

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_default():
    return cfg_mod.load(DEFAULT_YAML)


def test_group_geometry_balanced():
    """North burn group: pure +Y net force, zero net torque about the
    nominal CM, and per-axis differential authority."""
    cfg = load_default()
    cfg.rcs.cm_offset = np.zeros(3)  # nominal CM for the design check
    cfg.stationkeeping.burn.delta_v = 1.0
    burn = BurnController(cfg.rcs, cfg.stationkeeping, 0.0625)
    f_net = burn.forces.sum(axis=0)
    tau_net = burn.torques.sum(axis=0)
    # direction vectors are re-normalized at load, so compare loosely
    assert np.allclose(f_net, [0.0, 4 * 10 * 0.940, 0.0], atol=0.05)
    assert np.allclose(tau_net, 0.0, atol=1e-6)
    assert np.allclose(burn.burn_dir, [0.0, 1.0, 0.0])
    # Differential authority: each axis has thrusters of both torque signs
    for axis in range(3):
        assert np.any(burn.torques[:, axis] > 0.1)
        assert np.any(burn.torques[:, axis] < -0.1)


def test_off_pulse_allocation_signs():
    cfg = load_default()
    cfg.rcs.cm_offset = np.zeros(3)
    cfg.stationkeeping.burn.delta_v = 1.0
    burn = BurnController(cfg.rcs, cfg.stationkeeping, 0.0625)
    for axis in range(3):
        for sign in (1, -1):
            u = np.zeros(3, dtype=int)
            u[axis] = sign
            duty, force, torque = burn.step(u)
            assert np.sign(torque[axis]) == sign
            assert np.all(duty >= 0.0) and np.all(duty <= 1.0)
            # Off-pulsing keeps most of the delta-V flowing
            assert force @ burn.burn_dir > 0.5 * 37.6


def test_phase_plane_logic():
    pp = PhasePlane(PhasePlaneConfig(deadband_deg=0.1, rate_lead_s=10.0,
                                     min_drift_rate_dps=0.01,
                                     rate_limit_dps=0.05))
    db = np.deg2rad(0.1)
    zero = np.zeros(3)
    w_dr = np.deg2rad(0.01)
    # inside the hold channel: coast
    assert np.all(pp.step(zero + 0.5 * db, zero) == 0)
    # beyond +deadband at zero rate: fire negative
    u = pp.step(np.array([2 * db, 0, 0]), zero)
    assert u[0] == -1 and u[1] == 0 and u[2] == 0
    # DRIFT CHANNEL: beyond +deadband but drifting back >= min drift rate ->
    # thrusters stay off
    u = pp.step(np.array([2 * db, 0, 0]), np.array([-2 * w_dr, 0, 0]))
    assert u[0] == 0
    # favorable drift below the minimum drift rate: keep firing
    u = pp.step(np.array([2 * db, 0, 0]), np.array([-0.5 * w_dr, 0, 0]))
    assert u[0] == -1
    # mirror side
    u = pp.step(np.array([-2 * db, 0, 0]), np.array([2 * w_dr, 0, 0]))
    assert u[0] == 0
    u = pp.step(np.array([-2 * db, 0, 0]), zero)
    assert u[0] == +1
    # rate limit fires against rate regardless of attitude
    u = pp.step(zero, np.array([0.0, np.deg2rad(0.1), 0.0]))
    assert u[1] == -1


def test_burn_end_to_end():
    """1 m/s north burn: delta-V delivered, attitude held near the deadband,
    wheels untouched, burn terminates."""
    cfg = load_default()
    cfg.stationkeeping.burn.delta_v = 1.0
    cfg.guidance.step.angle_deg = 0.0
    cfg.simulation.duration_s = 100.0 + 130.0 + 120.0
    result = simulate.run(cfg)

    assert np.any(result.burning)
    assert not result.burning[-1], "burn did not terminate"
    dv = result.delta_v[-1]
    assert dv[1] >= 1.0  # target reached along +Y
    assert abs(dv[0]) < 0.05 and abs(dv[2]) < 0.05  # balanced geometry
    # Attitude and rate stayed near the phase-plane limits during the burn
    att = np.abs(result.att_err_deg[result.burning])
    assert np.max(att) < 3.0 * cfg.stationkeeping.phase_plane.deadband_deg
    # Wheels held through the burn
    h = result.h_wheel[result.burning]
    assert np.allclose(h[0], h[-1], atol=1e-9)


def test_burn_group_validation():
    cfg = load_default()
    cfg.stationkeeping.burn.group = "nonexistent"
    cfg.stationkeeping.burn.delta_v = 1.0
    with pytest.raises(ValueError, match="burn group"):
        BurnController(cfg.rcs, cfg.stationkeeping, 0.0625)
