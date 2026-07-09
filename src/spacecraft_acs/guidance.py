"""Attitude command generation.

All attitudes are quaternions from the inertial frame to the named frame
(scalar-first). The inertial frame is defined as the LVLH frame at t = 0;
the LVLH frame (+z nadir, +y orbit anti-normal, +x along-track) then rotates
about the fixed inertial −y axis at the orbit rate.

The commanded maneuver (guidance.step) is executed either as a discontinuous
quaternion step at time_s, or — when guidance.profiler.enabled — as a smooth
eigenaxis slew about the step axis with continuous attitude, rate, and
acceleration (see profiler.SlewProfile). `command(t)` returns the commanded
attitude, angular velocity, and angular acceleration; the acceleration feeds
the controller's feedforward path.
"""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import GuidanceConfig
from .profiler import SlewProfile


class Guidance:
    def __init__(self, cfg: GuidanceConfig, orbit_rate: float):
        self.cfg = cfg
        self.n = orbit_rate
        step = cfg.step
        self._axis = np.asarray(step.axis, dtype=float)
        self._axis /= np.linalg.norm(self._axis)
        self._angle = np.deg2rad(step.angle_deg)
        self._q_step = qt.from_axis_angle(self._axis, self._angle)
        self._profile = None
        if cfg.profiler.enabled:
            self._profile = SlewProfile(
                theta_f=abs(self._angle),
                v_max=np.deg2rad(cfg.profiler.max_rate_dps),
                a_max=np.deg2rad(cfg.profiler.max_accel_dps2),
            )
        # LVLH frame angular velocity, expressed in LVLH axes
        self._omega_lvlh = np.array([0.0, -orbit_rate, 0.0])

    @property
    def slew_duration(self) -> float | None:
        """Duration of the profiled slew, or None when profiling is disabled."""
        return self._profile.duration if self._profile is not None else None

    def lvlh_attitude(self, t: float) -> np.ndarray:
        """q from inertial to LVLH at time t."""
        return qt.from_axis_angle([0.0, -1.0, 0.0], self.n * t)

    def _offset(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maneuver offset (q_off, ω_off, α_off) relative to the base frame,
        with rates expressed in the offset (command) frame."""
        t_man = t - self.cfg.step.time_s
        if t_man < 0.0:
            return qt.IDENTITY, np.zeros(3), np.zeros(3)
        if self._profile is None:
            return self._q_step, np.zeros(3), np.zeros(3)
        sign = np.sign(self._angle) if self._angle != 0.0 else 1.0
        theta, rate, accel = self._profile.evaluate(t_man)
        q_off = qt.from_axis_angle(self._axis, sign * theta)
        return q_off, self._axis * sign * rate, self._axis * sign * accel

    def command(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q_cmd, omega_cmd, alpha_cmd): commanded attitude
        (inertial->command frame), command-frame angular velocity, and
        angular acceleration, both expressed in command axes."""
        if self.cfg.mode == "nadir":
            q_base = self.lvlh_attitude(t)
            omega_base = self._omega_lvlh
        else:  # inertial hold
            q_base = self.cfg.q_inertial
            omega_base = np.zeros(3)

        q_off, omega_off, alpha_off = self._offset(t)
        q_cmd = qt.multiply(q_base, q_off)
        # Base-frame rate mapped into the command frame, plus the slew rate
        # about the (base-frame-fixed) eigenaxis. R = dcm(q_off) maps base to
        # command axes and Ṙv = −ω_off × (Rv), giving the exact acceleration.
        r_omega_base = qt.dcm(q_off) @ omega_base
        omega_cmd = r_omega_base + omega_off
        alpha_cmd = alpha_off - np.cross(omega_off, r_omega_base)
        return q_cmd, omega_cmd, alpha_cmd
