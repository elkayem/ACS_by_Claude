from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import simulate
from spacecraft_acs.config import ThrusterConfig, UnloadConfig
from spacecraft_acs.momentum import MomentumManager

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def make_manager(dt=0.25, **unload_kw):
    unload = UnloadConfig(**{"trigger": 10.0, "target": 1.0, "rate_gain": 0.02, **unload_kw})
    cfg = ThrusterConfig(enabled=True, torque=5.0, min_on_time_s=0.02, unload=unload)
    return MomentumManager(cfg, dt)


def test_idle_below_trigger():
    mgr = make_manager()
    assert np.allclose(mgr.update(np.array([9.0, -5.0, 0.0])), 0.0)
    assert not mgr.unloading


def test_trigger_and_hysteresis():
    mgr = make_manager()
    mgr.update(np.array([10.5, 0.0, 0.0]))
    assert mgr.unloading
    # Stays active below trigger until reaching target
    mgr.update(np.array([5.0, 0.0, 0.0]))
    assert mgr.unloading
    mgr.update(np.array([0.9, 0.0, 0.0]))
    assert not mgr.unloading


def test_min_impulse_bit_quantization():
    """Every fired pulse delivers at least the minimum impulse bit, torque
    opposes the stored momentum, and the average tracks the request."""
    # Floor regime: proportional request (rate_gain*h = 0.02 N*m) is far
    # below the min-impulse floor (min_impulse/dt = 0.4 N*m) -> exactly one
    # minimum-impulse pulse per cycle
    mgr = make_manager(rate_gain=0.002)
    h = np.array([10.5, 0.0, 0.0])
    mgr.update(np.array([10.5, 0.0, 0.0]))  # arm (above trigger)
    torques = np.array([mgr.update(h)[0] for _ in range(100)])
    assert np.all(torques < 0.0)  # opposes +h
    impulses = np.abs(torques) * mgr.dt
    assert np.allclose(impulses, mgr.min_impulse)

    # Proportional regime: request 0.02*30 = 0.6 N*m exceeds the floor ->
    # per-cycle impulse above the minimum bit, average tracks the request
    mgr = make_manager(rate_gain=0.02, trigger=25.0)
    h = np.array([30.0, 0.0, 0.0])
    torques = np.array([mgr.update(h)[0] for _ in range(100)])
    impulses = np.abs(torques) * mgr.dt
    assert np.all(impulses >= mgr.min_impulse - 1e-12)
    assert np.isclose(np.mean(np.abs(torques)), 0.6, rtol=0.05)


def test_disabled_returns_zero():
    cfg = ThrusterConfig(enabled=False)
    mgr = MomentumManager(cfg, 0.25)
    assert np.allclose(mgr.update(np.array([100.0, 0.0, 0.0])), 0.0)


def _unload_config(compensation=True):
    cfg = cfg_mod.load(DEFAULT_YAML)
    cfg.thrusters.enabled = True
    cfg.thrusters.unload.trigger = 10.0
    cfg.thrusters.unload.target = 1.0
    cfg.thrusters.unload.rate_gain = 0.02
    cfg.thrusters.unload.feedforward_compensation = compensation
    cfg.simulation.initial_wheel_momentum = np.array([11.0, -11.0, 6.0])
    cfg.simulation.duration_s = 300.0
    cfg.guidance.step.angle_deg = 0.0
    return cfg


def test_unload_end_to_end():
    cfg = _unload_config()
    result = simulate.run(cfg)
    dt = result.t[1] - result.t[0]

    # Momentum unloaded to target on the triggered axes
    assert np.all(np.abs(result.h_wheel[-1]) < cfg.thrusters.unload.target + 0.5)
    # Momentum bookkeeping: wheel momentum change matches the thruster
    # impulse absorbed by the attitude loop (small residual for pointing)
    dh = result.h_wheel[-1] - result.h_wheel[0]
    impulse = np.sum(result.torque_thruster[:-1], axis=0) * dt
    assert np.allclose(dh, impulse, atol=0.2)
    # Pointing held through the unload with feedforward compensation
    assert np.max(np.abs(result.att_err_deg)) * 3600.0 < 30.0  # arcsec


def test_unload_compensation_helps():
    err_comp = np.max(np.abs(simulate.run(_unload_config(True)).att_err_deg))
    err_none = np.max(np.abs(simulate.run(_unload_config(False)).att_err_deg))
    assert err_comp < err_none


def test_unload_config_validation():
    with pytest.raises(ValueError):
        UnloadConfig(trigger=1.0, target=5.0)
