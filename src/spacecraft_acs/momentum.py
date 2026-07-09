"""Thruster-based reaction wheel momentum unloading.

Secular environmental torques (chiefly the constant SRP component) accumulate
wheel momentum that must periodically be dumped through external torque. At
GEO that means thrusters: when any wheel axis exceeds the trigger threshold,
the manager fires on/off thrusters against the stored momentum until every
axis is back below the target threshold.

Pulse quantization: the unload law requests torque `rate_gain * h_w`,
floored at one minimum impulse bit (torque * min_on_time_s) per cycle so the
unload always makes progress — bang-bang with a deadband, the classic
momentum-dump law. The requested impulse accumulates in a per-axis "impulse
debt"; once the debt covers a minimum impulse bit the thruster fires for the
debt's worth of on-time that cycle (capped at full-on). The returned torque
is the cycle average, which the ZOH dynamics integration applies exactly.

With `feedforward_compensation` the wheel torque command is offset by the
negative of the actual thruster torque each cycle, so the attitude loop only
sees what the wheels cannot cancel (saturation residual during pulses)
instead of the full unload torque.
"""

from __future__ import annotations

import numpy as np

from .config import ThrusterConfig


class MomentumManager:
    def __init__(self, cfg: ThrusterConfig, dt_ctrl: float):
        self.cfg = cfg
        self.dt = dt_ctrl
        self.unloading = False
        self._impulse_debt = np.zeros(3)  # N*m*s owed per axis (signed)

    @property
    def min_impulse(self) -> float:
        """Minimum impulse bit, N*m*s."""
        return self.cfg.torque * self.cfg.min_on_time_s

    def update(self, h_wheel: np.ndarray) -> np.ndarray:
        """One controller cycle: returns the cycle-average external thruster
        torque on the body (zeros when idle)."""
        if not self.cfg.enabled:
            return np.zeros(3)
        u = self.cfg.unload
        h_abs = np.abs(h_wheel)
        if not self.unloading and np.any(h_abs > u.trigger):
            self.unloading = True
        elif self.unloading and np.all(h_abs < u.target):
            self.unloading = False
            self._impulse_debt[:] = 0.0
        if not self.unloading:
            return np.zeros(3)

        torque = np.zeros(3)
        for i in range(3):
            if h_abs[i] < u.target:
                self._impulse_debt[i] = 0.0
                continue
            # External torque must oppose wheel momentum: the attitude loop
            # counters the thruster torque, and its counter-torque drives
            # h_w toward zero (dh_w/dt = -T_wheel = +T_thruster).
            # The request is floored at one minimum impulse bit per cycle so
            # the unload always makes progress: near the target the
            # proportional term alone would stall below the impulse bit.
            t_prop = max(u.rate_gain * h_abs[i], self.min_impulse / self.dt)
            t_req = -np.sign(h_wheel[i]) * min(t_prop, self.cfg.torque)
            self._impulse_debt[i] += t_req * self.dt
            if abs(self._impulse_debt[i]) < self.min_impulse:
                continue  # below the minimum impulse bit; keep accumulating
            on_time = min(abs(self._impulse_debt[i]) / self.cfg.torque, self.dt)
            fired = np.sign(self._impulse_debt[i]) * self.cfg.torque * on_time
            self._impulse_debt[i] -= fired
            torque[i] = fired / self.dt
        return torque
