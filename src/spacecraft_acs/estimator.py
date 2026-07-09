"""Multiplicative extended Kalman filter for attitude and gyro bias.

Standard 6-state MEKF (Lefferts/Markley): the attitude estimate is a
quaternion q̂ propagated with bias-corrected gyro rates; the filter state is
the small attitude error rotation vector δθ and gyro bias error δβ, with the
error reset into q̂ and β̂ after each star tracker update.

    propagate (every controller cycle, gyro):
        ω̂ = ω_meas − β̂;  q̂ ← q̂ ⊗ exp(ω̂ dt)
        Φ = [[I − [ω̂]× dt, −I dt], [0, I]]
        P ← Φ P Φᵀ + diag(σ_gyro² dt² I, σ_walk² dt I)
    update (at the star tracker rate):
        δz = 2·vec(q̂⁻¹ ⊗ q_st),  H = [I 0],  R = σ_st² I
        K = P Hᵀ (H P Hᵀ + R)⁻¹;  δx = K δz
        q̂ ← q̂ ⊗ exp(δθ);  β̂ += δβ;  P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ
"""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import EstimatorConfig, SensorConfig
from .sensors import ARCSEC


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


class Mekf:
    def __init__(
        self,
        cfg: EstimatorConfig,
        sensors: SensorConfig,
        dt: float,
        q0: np.ndarray,
    ):
        self.dt = dt
        self.q = qt.normalize(np.asarray(q0, dtype=float))
        self.bias = np.zeros(3)
        p_att = np.deg2rad(cfg.p0_att_deg) ** 2
        p_bias = np.deg2rad(cfg.p0_bias_dps) ** 2
        self.P = np.diag([p_att] * 3 + [p_bias] * 3)
        # Process noise: per-sample gyro white noise integrates into attitude
        # error variance (sigma*dt)^2 per step; bias walk variance grows
        # linearly at sigma_walk^2 per second
        q_att = (sensors.gyro_rate_noise * dt) ** 2
        q_bias = sensors.gyro_bias_walk**2 * dt
        self.Q = np.diag([q_att] * 3 + [q_bias] * 3)
        self.R = np.eye(3) * (sensors.star_tracker_noise_arcsec * ARCSEC) ** 2

    def propagate(self, omega_meas: np.ndarray) -> None:
        """One gyro propagation step (call every controller cycle)."""
        omega_hat = omega_meas - self.bias
        self.q = qt.normalize(
            qt.multiply(self.q, qt.from_rotation_vector(omega_hat * self.dt))
        )
        phi = np.eye(6)
        phi[:3, :3] -= _skew(omega_hat) * self.dt
        phi[:3, 3:] = -np.eye(3) * self.dt
        self.P = phi @ self.P @ phi.T + self.Q

    def update_star_tracker(self, q_meas: np.ndarray) -> None:
        """Star tracker measurement update + multiplicative reset."""
        dz = 2.0 * qt.error(self.q, q_meas)[1:]
        s = self.P[:3, :3] + self.R  # H P H^T + R with H = [I 0]
        k = self.P[:, :3] @ np.linalg.inv(s)
        dx = k @ dz
        # Joseph-form covariance update for numerical symmetry
        i_kh = np.eye(6)
        i_kh[:, :3] -= k
        self.P = i_kh @ self.P @ i_kh.T + k @ self.R @ k.T
        # Reset error state into the estimates
        self.q = qt.normalize(qt.multiply(self.q, qt.from_rotation_vector(dx[:3])))
        self.bias = self.bias + dx[3:]

    def rate_estimate(self, omega_meas: np.ndarray) -> np.ndarray:
        return omega_meas - self.bias

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma of [attitude (rad, 3), bias (rad/s, 3)]."""
        return np.sqrt(np.diag(self.P))
