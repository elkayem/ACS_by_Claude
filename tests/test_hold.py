from pathlib import Path

import numpy as np

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import montecarlo, simulate
from spacecraft_acs.stationkeeping import HoldController

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def hold_config():
    cfg = cfg_mod.load(DEFAULT_YAML)
    cfg.stationkeeping.hold = True
    cfg.guidance.mode = "inertial"
    cfg.guidance.step.angle_deg = 0.0
    cfg.environment.gravity_gradient = False
    cfg.simulation.initial_att_err_deg = np.ones(3) / np.sqrt(3.0)  # 1 deg
    cfg.simulation.duration_s = 600.0
    return cfg


def test_couples_are_pure_torque():
    cfg = hold_config()
    cfg.rcs.cm_offset = np.zeros(3)
    hold = HoldController(cfg.rcs, cfg.stationkeeping, 0.0625)
    for axis in range(3):
        for sign in (1, -1):
            u = np.zeros(3, dtype=int)
            u[axis] = sign
            duty, force, torque = hold.step(u)
            assert np.allclose(force, 0.0, atol=1e-9), "couple must be force-free"
            assert np.sign(torque[axis]) == sign
            others = [a for a in range(3) if a != axis]
            assert np.allclose(torque[others], 0.0, atol=1e-6)


def test_hold_acquires_and_stays():
    cfg = hold_config()
    res = simulate.run(cfg)
    db = cfg.stationkeeping.phase_plane.deadband_deg
    # per-axis worst error: what the per-axis relay and drift channel bound
    # (the norm of three independent limit cycles legitimately exceeds it)
    err = np.max(np.abs(res.att_err_deg), axis=1)
    assert np.linalg.norm(res.att_err_deg[0]) > 0.9  # started ~1 deg off
    half = len(res.t) // 2
    assert np.all(err[half:] < 1.3 * db)  # captured and bounded per axis
    # zero net delta-V from force-free couples
    assert np.all(np.abs(res.delta_v[-1]) < 1e-3)
    # wheels untouched
    assert np.allclose(res.h_wheel[0], res.h_wheel[-1], atol=1e-9)


def test_hold_campaign_small():
    cfg = hold_config()
    cfg.monte_carlo.n_runs = 2
    rows = montecarlo.run_hold_campaign(cfg)
    assert len(rows) == 2
    for r in rows:
        assert r["acquired"]
        assert np.isfinite(r["prop_rate_g_hr"])
        assert np.isfinite(r["slosh_pk"])
