from pathlib import Path

import numpy as np

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import montecarlo

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_default():
    return cfg_mod.load(DEFAULT_YAML)


def test_disperse_deterministic_and_bounded():
    cfg = load_default()
    disp = cfg.monte_carlo.dispersions
    a = montecarlo.disperse(cfg, disp, np.random.default_rng(5))
    b = montecarlo.disperse(cfg, disp, np.random.default_rng(5))
    assert np.allclose(a.spacecraft.inertia, b.spacecraft.inertia)
    assert a.spacecraft.modes[0].freq_hz == b.spacecraft.modes[0].freq_hz

    rng = np.random.default_rng(6)
    for _ in range(20):
        d = montecarlo.disperse(cfg, disp, rng)
        j_ratio = np.diag(d.spacecraft.inertia) / np.diag(cfg.spacecraft.inertia)
        assert np.all(j_ratio >= 1.0 - disp.inertia_pct / 100.0 - 1e-12)
        assert np.all(j_ratio <= 1.0 + disp.inertia_pct / 100.0 + 1e-12)
        for m_d, m_n in zip(d.spacecraft.modes, cfg.spacecraft.modes):
            assert abs(m_d.freq_hz / m_n.freq_hz - 1.0) <= disp.mode_freq_pct / 100.0 + 1e-12
            lo, hi = disp.mode_damping_range
            assert lo <= m_d.damping <= hi
        # Physical validity enforced by construction
        L = d.spacecraft.participation_matrix
        assert np.all(
            np.linalg.eigvalsh(d.spacecraft.inertia - L @ L.T) > 0.0
        )
    # Controller untouched
    assert np.allclose(
        d.controller.design.bandwidth_hz, cfg.controller.design.bandwidth_hz
    )


def test_zero_dispersion_matches_nominal():
    cfg = load_default()
    disp = type(cfg.monte_carlo.dispersions)(
        inertia_pct=0.0,
        mode_freq_pct=0.0,
        mode_damping_range=(0.005, 0.005),
        participation_pct=0.0,
    )
    d = montecarlo.disperse(cfg, disp, np.random.default_rng(0))
    nominal = montecarlo.evaluate(cfg, -1, time_domain=False, n_points=800)
    sample = montecarlo.evaluate(d, 0, time_domain=False, n_points=800)
    assert np.isclose(sample.gm_db, nominal.gm_db, atol=0.05)
    assert np.isclose(sample.pm_deg, nominal.pm_deg, atol=0.05)


def test_small_mc_end_to_end(tmp_path):
    cfg = load_default()
    cfg.monte_carlo.n_runs = 5
    results = montecarlo.run(cfg)
    assert len(results.samples) == 5
    assert np.all(np.isfinite(results.field_array("gm_db")))
    assert np.all(np.isfinite(results.field_array("pm_deg")))
    text = montecarlo.report(results)
    assert "pass rate" in text
    csv_path = tmp_path / "mc.csv"
    montecarlo.to_csv(results, csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 1 + 1 + 5  # header + nominal + samples
