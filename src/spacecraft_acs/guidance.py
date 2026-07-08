"""Attitude command generation.

All attitudes are quaternions from the inertial frame to the named frame
(scalar-first). The inertial frame is defined as the LVLH frame at t = 0;
the LVLH frame (+z nadir, +y orbit anti-normal, +x along-track) then rotates
about the fixed inertial −y axis at the orbit rate.
"""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import GuidanceConfig


class Guidance:
    def __init__(self, cfg: GuidanceConfig, orbit_rate: float):
        self.cfg = cfg
        self.n = orbit_rate
        self._q_step = qt.from_axis_angle(
            cfg.step.axis, np.deg2rad(cfg.step.angle_deg)
        )
        # LVLH frame angular velocity, expressed in LVLH axes
        self._omega_lvlh = np.array([0.0, -orbit_rate, 0.0])

    def lvlh_attitude(self, t: float) -> np.ndarray:
        """q from inertial to LVLH at time t."""
        return qt.from_axis_angle([0.0, -1.0, 0.0], self.n * t)

    def command(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_cmd, omega_cmd): commanded attitude (inertial->command
        frame) and command-frame angular velocity in command axes."""
        if self.cfg.mode == "nadir":
            q_base = self.lvlh_attitude(t)
            omega_base = self._omega_lvlh
        else:  # inertial hold
            q_base = self.cfg.q_inertial
            omega_base = np.zeros(3)

        if t >= self.cfg.step.time_s:
            # Step offset applied in the base (command) frame; the command
            # rate transforms into the offset frame.
            q_cmd = qt.multiply(q_base, self._q_step)
            omega_cmd = qt.dcm(self._q_step) @ omega_base
        else:
            q_cmd = q_base
            omega_cmd = omega_base.copy()
        return q_cmd, omega_cmd
