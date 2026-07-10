from pathlib import Path

import numpy as np
import pytest

from spacecraft_acs import config as cfg

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_default_config_loads():
    c = cfg.load(DEFAULT_YAML)
    assert c.spacecraft.inertia.shape == (3, 3)
    assert len(c.spacecraft.modes) == 4
    assert len(c.spacecraft.tanks) == 2
    # 4 structural + 2 slosh-equivalent modes per tank
    assert c.spacecraft.participation_matrix.shape == (3, 8)
    assert c.controller.rate_hz == 16.0
    assert c.guidance.mode == "nadir"
    assert np.isclose(c.orbit_rate, 7.2921159e-5)


def test_asymmetric_inertia_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        cfg.SpacecraftConfig(inertia=[[100, 5, 0], [0, 100, 0], [0, 0, 100]])


def test_non_positive_definite_inertia_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        cfg.SpacecraftConfig(inertia=[[100, 0, 0], [0, -5, 0], [0, 0, 100]])


def test_excessive_participation_rejected():
    # l^2 = 121 > J_xx = 100 makes J - L L^T indefinite
    with pytest.raises(ValueError, match="participation too large"):
        cfg.SpacecraftConfig(
            inertia=np.diag([100.0, 100.0, 100.0]),
            modes=[cfg.ModeConfig(freq_hz=0.1, damping=0.01, participation=[11.0, 0, 0])],
        )


def test_bad_mode_damping_rejected():
    with pytest.raises(ValueError, match="damping"):
        cfg.ModeConfig(freq_hz=0.1, damping=1.5, participation=[1, 0, 0])


def test_explicit_gains_shape_checked():
    with pytest.raises(ValueError, match="kp"):
        cfg.ControllerConfig(kp=[1.0, 2.0])


def test_unknown_filter_type_rejected():
    with pytest.raises(ValueError, match="filter type"):
        cfg.FilterConfig(type="bandpass", freq_hz=0.1)
