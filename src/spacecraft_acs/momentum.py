"""Reaction-wheel momentum unloading with the RCS thruster couples.

Secular environmental torques (chiefly the constant SRP component) accumulate
wheel momentum that must periodically be dumped through external torque. At
GEO that means thrusters: when any wheel axis exceeds the trigger threshold,
the manager fires the pure-torque RCS couple (rcs.couples) opposing that
axis's momentum until every axis is back below the target threshold.

The couple torque is real force x moment arm about the ACTUAL center of mass
(position - rcs.cm_offset), the same geometry the attitude-hold mode uses --
there is no abstract commanded torque. A consequence of using flight-sized
thrusters is that a single minimum-impulse couple pulse (force * min_on_time
* moment arm) is far larger than the reaction wheels can cancel, so each
unload pulse produces a real pointing transient; `feedforward_compensation`
lets the wheels absorb what they can (their saturation limit), not the whole
pulse. Momentum unloads are therefore scheduled outside precision-pointing
windows on real spacecraft.

Pulse quantization: each axis accumulates a signed torque-impulse debt from
the proportional request (rate_gain * h_w, floored so the dump always makes
progress). When an axis's debt reaches one minimum-impulse couple bit its
couple fires this cycle; the debt is then reconciled against the torque the
firing actually delivered (which includes cross-axis terms from the CM
offset and thrusters shared between couples), closing the loop honestly.
"""

from __future__ import annotations

import numpy as np

from .config import RcsConfig, UnloadConfig

G0 = 9.80665  # m/s^2, for Isp -> propellant


class MomentumManager:
    AXES = ("roll", "pitch", "yaw")

    def __init__(self, cfg: UnloadConfig, rcs: RcsConfig, dt: float):
        self.cfg = cfg
        self.rcs = rcs
        self.dt = dt
        self.unloading = False
        self._debt = np.zeros(3)  # signed per-axis torque-impulse owed, N*m*s
        self.propellant_kg = 0.0

        self.available = bool(cfg.enabled and rcs.thrusters and rcs.couples)
        n = len(rcs.thrusters)
        self.forces = np.array([t.force * t.direction for t in rcs.thrusters]) \
            if n else np.zeros((0, 3))
        self.torques = np.array(
            [np.cross(t.position - rcs.cm_offset, t.force * t.direction)
             for t in rcs.thrusters]
        ) if n else np.zeros((0, 3))
        self._thrust = np.array([t.force for t in rcs.thrusters])
        self.min_on = rcs.min_on_time_s

        # couple mask and primary-axis torque magnitude per (axis, sign)
        idx = {t.name: i for i, t in enumerate(rcs.thrusters)}
        self._mask = {}
        self._tau = np.zeros((3, 2))  # [axis][0:+ ,1:-] |primary-axis torque|
        for axis, name in enumerate(self.AXES):
            for si, (sign, tag) in enumerate(((+1, "+"), (-1, "-"))):
                members = rcs.couples.get(f"{name}{tag}") if self.available else None
                if not members:
                    self.available = False
                    continue
                mask = np.zeros(n, dtype=bool)
                for m in members:
                    mask[idx[m]] = True
                self._mask[(axis, sign)] = mask
                self._tau[axis, si] = abs(self.torques[mask].sum(axis=0)[axis])

    def _tau_axis(self, axis: int, sign: int) -> float:
        return self._tau[axis, 0 if sign > 0 else 1]

    @property
    def min_impulse(self) -> float:
        """Smallest min-impulse couple bit across the axes, N*m*s (roll/yaw
        couples are the strongest, so this is set by the weakest axis)."""
        nz = self._tau[self._tau > 0]
        return float(np.min(nz) * self.min_on) if nz.size else 0.0

    def update(self, h_wheel: np.ndarray) -> np.ndarray:
        """One controller cycle: cycle-average external couple torque on the
        body (zeros when idle)."""
        if not self.available:
            return np.zeros(3)
        h_abs = np.abs(h_wheel)
        if not self.unloading and np.any(h_abs > self.cfg.trigger):
            self.unloading = True
        elif self.unloading and np.all(h_abs < self.cfg.target):
            self.unloading = False
            self._debt[:] = 0.0
        if not self.unloading:
            return np.zeros(3)

        duty = np.zeros(len(self.torques))
        for i in range(3):
            if h_abs[i] < self.cfg.target:
                self._debt[i] = 0.0
                continue
            sign = -1 if h_wheel[i] > 0 else +1  # oppose the stored momentum
            tau = self._tau_axis(i, sign)
            if tau <= 0.0:
                continue
            # Proportional request sets the pulse cadence: debt accumulates
            # at rate_gain * |h| and fires one couple pulse when it reaches a
            # min-impulse bit. A single couple pulse (~tau * min_on) far
            # exceeds wheel torque authority, so the cadence must be slow
            # enough (small rate_gain) for the body to recover between
            # pulses — that is the real constraint, not an every-cycle floor.
            self._debt[i] += sign * self.cfg.rate_gain * h_abs[i] * self.dt
            if abs(self._debt[i]) < tau * self.min_on:
                continue  # below one min-impulse couple bit; keep accumulating
            # fire the couple; duty to burn the accumulated debt this cycle
            d = min(abs(self._debt[i]) / (tau * self.dt), 1.0)
            mask = self._mask[(i, int(np.sign(self._debt[i])))]
            duty[mask] = np.maximum(duty[mask], d)

        if not np.any(duty):
            return np.zeros(3)
        torque = duty @ self.torques
        # reconcile every axis's debt against the torque actually delivered
        # (cross-axis terms and shared thrusters included)
        self._debt -= torque * self.dt
        on_time = duty * self.dt
        self.propellant_kg += float(on_time @ self._thrust) / (self.rcs.isp_s * G0)
        return torque
