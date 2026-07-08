"""Attitude and rate sensor models (star tracker + gyro)."""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import SensorConfig

ARCSEC = np.pi / (180.0 * 3600.0)


class SensorSuite:
    """Applies measurement errors to the true state at each controller sample.

    Simple direct-measurement model: the controller consumes the star tracker
    quaternion and gyro rate directly (no estimator in v1).
    """

    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def measure(self, q_true: np.ndarray, omega_true: np.ndarray):
        """Return (q_meas, omega_meas)."""
        if self.cfg.perfect:
            return q_true.copy(), omega_true.copy()
        att_err = self.rng.standard_normal(3) * (
            self.cfg.star_tracker_noise_arcsec * ARCSEC
        )
        q_meas = qt.normalize(qt.multiply(q_true, qt.from_rotation_vector(att_err)))
        omega_meas = (
            omega_true
            + self.cfg.gyro_bias
            + self.rng.standard_normal(3) * self.cfg.gyro_rate_noise
        )
        return q_meas, omega_meas
