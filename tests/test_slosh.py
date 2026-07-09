from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import simulate, slosh
from spacecraft_acs.config import SpacecraftConfig, TankConfig

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_slosh_mass_fraction_curve():
    """Monotonically decreasing in fill; sane limits (shallow pool sloshes
    almost entirely, full tank barely at all)."""
    fills = np.linspace(0.05, 0.95, 19)
    fracs = np.array([slosh.slosh_mass_fraction(f) for f in fills])
    assert np.all(np.diff(fracs) < 0.0)
    assert 0.9 < fracs[0] <= 1.0
    assert 0.0 < fracs[-1] < 0.15
    with pytest.raises(ValueError):
        slosh.slosh_mass_fraction(1.0)


def test_tank_reduction_geometry_and_scaling():
    """Two modes per tank, participation = (r x e)*sqrt(m_s/(1-m_s/M)),
    orthogonal lateral directions, frequency raised by the CM-shift factor."""
    tank = TankConfig(
        propellant_mass=700.0, fill_fraction=0.5, location=[0.0, 0.0, 0.9],
        axis=[0.0, 0.0, 1.0], freq_hz=0.007, damping=0.003,
    )
    total_mass = 3000.0
    modes = slosh.tank_equivalent_modes(tank, total_mass)
    assert len(modes) == 2

    m_s = 700.0 * slosh.slosh_mass_fraction(0.5)
    cm_factor = 1.0 - m_s / total_mass
    expected_mag = 0.9 * np.sqrt(m_s / cm_factor)  # |r x e| = 0.9 for e ⊥ z
    for m in modes:
        assert np.isclose(np.linalg.norm(m.participation), expected_mag)
        assert np.isclose(m.freq_hz, 0.007 / np.sqrt(cm_factor))
        assert m.participation[2] == pytest.approx(0.0, abs=1e-12)  # no yaw
        assert m.damping == 0.003
    # The two lateral modes are orthogonal
    assert np.isclose(modes[0].participation @ modes[1].participation, 0.0)


def test_explicit_slosh_mass_override():
    tank = TankConfig(propellant_mass=700.0, location=[0, 0, 1.0], slosh_mass=100.0)
    modes = slosh.tank_equivalent_modes(tank, 3000.0)
    expected = 1.0 * np.sqrt(100.0 / (1.0 - 100.0 / 3000.0))
    assert np.isclose(np.linalg.norm(modes[0].participation), expected)


def test_spacecraft_config_appends_slosh_modes():
    cfg = cfg_mod.load(DEFAULT_YAML)
    sc = cfg.spacecraft
    assert len(sc.tanks) == 2
    assert len(sc.all_modes) == len(sc.modes) + 4  # two lateral modes per tank
    assert sc.participation_matrix.shape == (3, len(sc.all_modes))
    # Positive-definiteness still enforced with slosh included
    L = sc.participation_matrix
    assert np.all(np.linalg.eigvalsh(sc.inertia - L @ L.T) > 0.0)


def test_oversized_slosh_participation_rejected():
    with pytest.raises(ValueError, match="participation too large"):
        SpacecraftConfig(
            inertia=np.diag([100.0, 100.0, 100.0]),
            mass=3000.0,
            tanks=[TankConfig(propellant_mass=500.0, location=[0.0, 0.0, 3.0])],
        )


def test_closed_loop_sim_with_slosh():
    """Default config (tanks included): quiet nadir hold stays stable and the
    slosh coordinates are populated and bounded."""
    cfg = cfg_mod.load(DEFAULT_YAML)
    cfg.simulation.duration_s = 600.0
    cfg.guidance.step.angle_deg = 0.5  # excite the loop
    cfg.guidance.profiler.enabled = True
    result = simulate.run(cfg)
    assert np.all(np.isfinite(result.att_err_deg))
    n_struct = len(cfg.spacecraft.modes)
    slosh_eta = result.eta[:, n_struct:]
    assert slosh_eta.shape[1] == 4
    assert np.max(np.abs(slosh_eta)) > 0.0  # excited
    assert np.max(np.abs(result.att_err_deg)) < 0.1  # tracking maintained
