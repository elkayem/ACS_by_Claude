"""Attitude and rate sensor models (star tracker + gyro)."""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import SensorConfig

ARCSEC = np.pi / (180.0 * 3600.0)


class SensorSuite:
    """Applies measurement errors to the true state at each controller sample.

    The gyro bias is a state: initialized from config and optionally evolving
    as a random walk (gyro_bias_walk, rad/s per sqrt(s)). The current true
    bias is exposed as `.bias` so estimation error can be scored.
    """

    def __init__(self, cfg: SensorConfig, dt: float):
        self.cfg = cfg
        self.dt = dt
        self.rng = np.random.default_rng(cfg.seed)
        self.bias = cfg.gyro_bias.copy()

    def measure(self, q_true: np.ndarray, omega_true: np.ndarray):
        """Return (q_meas, omega_meas)."""
        if self.cfg.perfect:
            return q_true.copy(), omega_true.copy()
        if self.cfg.gyro_bias_walk > 0.0:
            self.bias = self.bias + self.rng.standard_normal(3) * (
                self.cfg.gyro_bias_walk * np.sqrt(self.dt)
            )
        att_err = self.rng.standard_normal(3) * (
            self.cfg.star_tracker_noise_arcsec * ARCSEC
        )
        q_meas = qt.normalize(qt.multiply(q_true, qt.from_rotation_vector(att_err)))
        omega_meas = (
            omega_true
            + self.bias
            + self.rng.standard_normal(3) * self.cfg.gyro_rate_noise
        )
        return q_meas, omega_meas
