from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import montecarlo
from spacecraft_acs.config import ModeConfig, SpacecraftConfig

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"

J_NS = np.diag([15000.0, 3000.0, 14500.0])


def make_sc(angle):
    return SpacecraftConfig(
        inertia=J_NS,
        modes=[
            ModeConfig(freq_hz=0.10, damping=0.005,
                       participation=[55.0, 0.0, 0.0], rotates_with_array=True),
            ModeConfig(freq_hz=0.35, damping=0.005,
                       participation=[0.0, 25.0, 0.0]),  # torsion, body-fixed
        ],
        array_angle_deg=angle,
    )


def test_participation_rotates_about_pitch():
    base = make_sc(0.0).all_modes[0].participation
    assert np.allclose(base, [55.0, 0.0, 0.0])

    p90 = make_sc(90.0).all_modes[0].participation
    assert np.isclose(abs(p90[2]), 55.0)  # fully in yaw
    assert np.isclose(p90[0], 0.0, atol=1e-12)
    assert np.isclose(p90[1], 0.0, atol=1e-12)  # never leaks into pitch

    p45 = make_sc(45.0).all_modes[0].participation
    assert np.isclose(abs(p45[0]), 55.0 / np.sqrt(2))
    assert np.isclose(abs(p45[2]), 55.0 / np.sqrt(2))
    # Norm is preserved at every angle
    for a in (17.0, 133.0, 289.0):
        assert np.isclose(np.linalg.norm(make_sc(a).all_modes[0].participation), 55.0)


def test_body_fixed_modes_do_not_rotate():
    for a in (0.0, 45.0, 90.0, 270.0):
        torsion = make_sc(a).all_modes[1].participation
        assert np.allclose(torsion, [0.0, 25.0, 0.0])


def test_slosh_modes_unaffected_by_array_angle():
    cfg0 = cfg_mod.load(DEFAULT_YAML)
    cfg90 = cfg_mod.load(DEFAULT_YAML)
    cfg90.spacecraft.array_angle_deg = 90.0
    n = len(cfg0.spacecraft.modes)
    s0 = cfg0.spacecraft.all_modes[n:]
    s90 = cfg90.spacecraft.all_modes[n:]
    for a, b in zip(s0, s90):
        assert np.allclose(a.participation, b.participation)


def test_pd_validation_covers_all_angles():
    """Participation legal at 0 deg but oversized for the smaller yaw
    inertia must be rejected (the mode sweeps into yaw during the day)."""
    with pytest.raises(ValueError, match="array angle"):
        SpacecraftConfig(
            inertia=np.diag([20000.0, 3000.0, 5000.0]),
            modes=[ModeConfig(freq_hz=0.1, damping=0.005,
                              participation=[75.0, 0.0, 0.0],
                              rotates_with_array=True)],
        )


def test_mc_disperses_array_angle():
    cfg = cfg_mod.load(DEFAULT_YAML)
    rng = np.random.default_rng(3)
    angles = [
        montecarlo.disperse(cfg, cfg.monte_carlo.dispersions, rng)
        .spacecraft.array_angle_deg
        for _ in range(20)
    ]
    assert np.all((np.array(angles) >= 0.0) & (np.array(angles) < 360.0))
    assert np.std(angles) > 30.0  # actually spread over the circle

    cfg.monte_carlo.dispersions.array_angle = False
    d = montecarlo.disperse(cfg, cfg.monte_carlo.dispersions, rng)
    assert d.spacecraft.array_angle_deg == cfg.spacecraft.array_angle_deg
