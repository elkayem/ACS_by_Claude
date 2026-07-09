"""Smooth eigenaxis slew profile.

Cycloidal (versine) acceleration S-curve: the acceleration ramps as
a(t) = a_max·(1 − cos(2πt/Ta))/2 over each accel/decel phase, so attitude,
rate, and acceleration are all continuous (C² attitude, zero acceleration at
both ends), with an optional constant-rate cruise for long slews. Closed-form
piecewise expressions:

    accel phase, t ∈ [0, Ta]:
        θ̈ = a/2 (1 − cos(2πt/Ta))
        θ̇ = a/2 (t − Ta/2π · sin(2πt/Ta))
        θ  = a/2 (t²/2 + (Ta/2π)² (cos(2πt/Ta) − 1))
    cruise at θ̇ = v_peak; decel phase mirrors the accel phase.

Peak rate v_peak = min(v_max, sqrt(θ_f·a_max/2)); Ta = 2·v_peak/a_max, and
each accel phase covers v_peak²/a_max of angle.
"""

from __future__ import annotations

import numpy as np


class SlewProfile:
    """Scalar angle profile θ(t) from 0 to theta_f (theta_f >= 0, radians)."""

    def __init__(self, theta_f: float, v_max: float, a_max: float):
        if theta_f < 0.0 or v_max <= 0.0 or a_max <= 0.0:
            raise ValueError("theta_f must be >= 0 and v_max, a_max positive")
        self.theta_f = theta_f
        self.a_max = a_max
        self.v_peak = min(v_max, np.sqrt(0.5 * theta_f * a_max))
        if theta_f == 0.0:
            self.t_accel = 0.0
            self.t_cruise = 0.0
        else:
            self.t_accel = 2.0 * self.v_peak / a_max
            theta_accel = self.v_peak**2 / a_max
            self.t_cruise = (theta_f - 2.0 * theta_accel) / self.v_peak
        self.duration = 2.0 * self.t_accel + self.t_cruise

    def _accel_phase(self, t: float):
        """(θ, θ̇, θ̈) within the acceleration phase, t ∈ [0, Ta]."""
        a, ta = self.a_max, self.t_accel
        w = 2.0 * np.pi / ta
        theta = 0.5 * a * (0.5 * t**2 + (np.cos(w * t) - 1.0) / w**2)
        rate = 0.5 * a * (t - np.sin(w * t) / w)
        accel = 0.5 * a * (1.0 - np.cos(w * t))
        return theta, rate, accel

    def evaluate(self, t: float) -> tuple[float, float, float]:
        """Return (θ, θ̇, θ̈) at time t from the start of the slew."""
        if self.theta_f == 0.0 or t <= 0.0:
            return 0.0, 0.0, 0.0
        if t >= self.duration:
            return self.theta_f, 0.0, 0.0
        ta, tc = self.t_accel, self.t_cruise
        theta_accel = self.v_peak**2 / self.a_max
        if t < ta:
            return self._accel_phase(t)
        if t < ta + tc:
            return theta_accel + self.v_peak * (t - ta), self.v_peak, 0.0
        # Decel phase: mirror of the accel phase about the endpoint
        theta_m, rate_m, accel_m = self._accel_phase(self.duration - t)
        return self.theta_f - theta_m, rate_m, -accel_m
