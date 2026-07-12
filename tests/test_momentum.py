from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import simulate
from spacecraft_acs.config import UnloadConfig
from spacecraft_acs.momentum import MomentumManager

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def make_manager(dt=0.0625, **unload_kw):
    """Manager built on the default RCS geometry with an unload policy."""
    cfg = cfg_mod.load(DEFAULT_YAML)
    kw = {"enabled": True, "trigger": 10.0, "target": 1.0, "rate_gain": 0.02}
    kw.update(unload_kw)
    return MomentumManager(UnloadConfig(**kw), cfg.rcs, dt)


def test_idle_below_trigger():
    mgr = make_manager()
    assert np.allclose(mgr.update(np.array([9.0, -5.0, 0.0])), 0.0)
    assert not mgr.unloading


def test_trigger_and_hysteresis():
    mgr = make_manager()
    mgr.update(np.array([10.5, 0.0, 0.0]))
    assert mgr.unloading
    mgr.update(np.array([5.0, 0.0, 0.0]))  # below trigger, above target
    assert mgr.unloading
    mgr.update(np.array([0.9, 0.0, 0.0]))  # below target
    assert not mgr.unloading


def test_disabled_and_unavailable_return_zero():
    cfg = cfg_mod.load(DEFAULT_YAML)
    off = MomentumManager(UnloadConfig(enabled=False), cfg.rcs, 0.0625)
    assert np.allclose(off.update(np.array([100.0, 0.0, 0.0])), 0.0)
    # enabled but no couples -> unavailable, no-op
    cfg.rcs.couples = {}
    na = MomentumManager(UnloadConfig(enabled=True), cfg.rcs, 0.0625)
    assert not na.available
    assert np.allclose(na.update(np.array([100.0, 0.0, 0.0])), 0.0)


def test_couple_torque_from_geometry_opposes_momentum():
    """The fired couple torque is real force x moment arm and opposes the
    stored wheel momentum on the triggered axis."""
    mgr = make_manager(rate_gain=0.05)
    h = np.array([30.0, 0.0, 0.0])
    torque = np.zeros(3)
    for _ in range(50):  # let a pulse fire
        t = mgr.update(h)
        if np.any(t):
            torque = t
            break
    assert torque[0] < 0.0  # opposes +roll momentum
    # magnitude is couple-scale (tens of N*m cycle-average), not the old
    # sub-N*m abstract torque
    assert abs(torque[0]) > 1.0


def test_unload_couples_are_force_free():
    """Each unload couple's member thrust forces sum to zero, so firing any
    couple imparts torque but no net force."""
    cfg = cfg_mod.load(DEFAULT_YAML)
    mgr = MomentumManager(
        UnloadConfig(enabled=True, trigger=10.0, target=1.0), cfg.rcs, 0.0625
    )
    for (axis, sign), mask in mgr._mask.items():
        net_force = mgr.forces[mask].sum(axis=0)
        assert np.allclose(net_force, 0.0, atol=1e-6), (axis, sign, net_force)


def test_propellant_accumulates():
    mgr = make_manager(rate_gain=0.05)
    h = np.array([30.0, 0.0, 0.0])
    for _ in range(200):
        mgr.update(h)
    assert mgr.propellant_kg > 0.0
    assert mgr.min_impulse > 0.1  # real couple bit, N*m*s (10 N * ~1.2 m arm)


def _unload_config():
    cfg = cfg_mod.load(DEFAULT_YAML)
    cfg.unload.enabled = True
    cfg.unload.trigger = 10.0
    cfg.unload.target = 1.0
    cfg.unload.rate_gain = 0.01  # faster dump for a short test window
    cfg.simulation.initial_wheel_momentum = np.array([11.0, -11.0, 6.0])
    cfg.simulation.duration_s = 700.0
    cfg.guidance.step.angle_deg = 0.0
    return cfg


def test_unload_end_to_end():
    cfg = _unload_config()
    result = simulate.run(cfg)
    dt = result.t[1] - result.t[0]

    # Momentum unloaded to (near) target on the triggered axes
    assert np.all(np.abs(result.h_wheel[-1]) < cfg.unload.target + 1.5)
    # Bookkeeping: wheel momentum change matches the external couple impulse
    # the attitude loop absorbed (atol covers residual in-flight body/modal
    # momentum at the snapshot)
    dh = result.h_wheel[-1] - result.h_wheel[0]
    impulse = np.sum(result.torque_thruster[:-1], axis=0) * dt
    assert np.allclose(dh, impulse, atol=2.5)
    # Force-free couples -> negligible parasitic delta-V
    assert np.all(np.abs(result.delta_v[-1]) < 1e-3)
    # Real propellant was spent
    assert result.unload_propellant_kg > 0.0


def test_unload_disturbs_pointing_and_is_bounded():
    """Honest physics: a min-impulse couple pulse far exceeds wheel torque
    authority, so unloads visibly disturb pointing (unlike the old abstract
    model where feedforward could cancel each pulse) — but the average
    unload torque stays within wheel authority so it remains bounded."""
    result = simulate.run(_unload_config())
    peak_deg = np.max(np.abs(result.att_err_deg))
    assert peak_deg * 3600.0 > 30.0  # not hidden by the wheels
    assert peak_deg < 5.0  # bounded, not a runaway


def test_unload_config_validation():
    with pytest.raises(ValueError):
        UnloadConfig(trigger=1.0, target=5.0)
