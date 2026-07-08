"""Environmental disturbance torques for a GEO spacecraft."""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import EnvironmentConfig


class Environment:
    def __init__(self, cfg: EnvironmentConfig, inertia: np.ndarray, orbit_rate: float):
        self.cfg = cfg
        self.J = inertia
        self.n = orbit_rate

    def torque(self, t: float, q_body_from_lvlh: np.ndarray) -> np.ndarray:
        """Total disturbance torque in body axes.

        q_body_from_lvlh is the attitude of the body relative to the local
        orbital (LVLH) frame, whose +z axis points to nadir.
        """
        torque = np.zeros(3)
        if self.cfg.gravity_gradient:
            # Nadir unit vector expressed in body axes
            o3 = qt.dcm(q_body_from_lvlh).T @ np.array([0.0, 0.0, 1.0])
            torque += 3.0 * self.n**2 * np.cross(o3, self.J @ o3)
        if self.cfg.srp.enabled:
            torque += self.cfg.srp.constant
            torque += self.cfg.srp.harmonic_amplitude * np.sin(self.n * t)
        return torque
