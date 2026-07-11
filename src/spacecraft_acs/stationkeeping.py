"""Stationkeeping thruster control mode.

During a delta-V burn the reaction wheels are held (thruster torques would
saturate them in seconds) and attitude is controlled by a classical
per-axis PHASE PLANE: the switching function s = θ_err + T_lead·ω_err is
compared against a deadband with hysteresis, plus a hard rate limit that
fires against excessive rate regardless of attitude. The phase-plane
command is realized by OFF-PULSING the burn thrusters: each commanded axis
reduces the duty of the thruster subset whose torque opposes the command,
so control torque is generated while most of the delta-V continues.

Thruster torques are computed about the ACTUAL center of mass (nominal
position minus the configured cm_offset), so a CM offset produces the
realistic constant disturbance torque that drives the phase-plane limit
cycle during burns.

Scope notes: delta-V is integrated in body axes (≈ the orbital frame while
attitude is held; no orbit propagation), and slosh is excited only through
the rotational coupling — translational slosh forcing under thrust is not
modeled.
"""

from __future__ import annotations

import numpy as np

from .config import PhasePlaneConfig, RcsConfig, StationkeepingConfig


class PhasePlane:
    """Per-axis bang-bang with deadband, rate lead, hysteresis, rate limit."""

    def __init__(self, cfg: PhasePlaneConfig):
        self.cfg = cfg
        self.db = np.deg2rad(cfg.deadband_deg)
        self.rate_lim = np.deg2rad(cfg.rate_limit_dps)
        self._firing = np.zeros(3, dtype=int)  # current command per axis

    def step(self, theta_err: np.ndarray, omega_err: np.ndarray) -> np.ndarray:
        """Return the commanded torque sign per axis in {-1, 0, +1}."""
        s = theta_err + self.cfg.rate_lead_s * omega_err
        u = self._firing.copy()
        for i in range(3):
            if abs(omega_err[i]) > self.rate_lim:
                u[i] = -int(np.sign(omega_err[i]))
            elif self._firing[i] == 0:
                if s[i] > self.db:
                    u[i] = -1
                elif s[i] < -self.db:
                    u[i] = +1
            else:
                # keep firing until the switching function re-enters the
                # hysteresis band, then coast — but flip immediately if it
                # shot through to the opposite deadband
                if s[i] > self.db and self._firing[i] > 0:
                    u[i] = -1
                elif s[i] < -self.db and self._firing[i] < 0:
                    u[i] = +1
                elif abs(s[i]) < self.cfg.hysteresis * self.db:
                    u[i] = 0
        self._firing = u
        return u


class BurnController:
    """Off-pulse duty allocation for one burn group."""

    def __init__(self, rcs: RcsConfig, sk: StationkeepingConfig, dt: float):
        self.rcs = rcs
        self.sk = sk
        self.dt = dt
        members = rcs.groups.get(sk.burn.group)
        if not members:
            raise ValueError(f"burn group {sk.burn.group!r} not defined in rcs.groups")
        units = {t.name: t for t in rcs.thrusters}
        self.units = [units[name] for name in members]
        # Force and torque per thruster about the ACTUAL center of mass
        self.forces = np.array([t.force * t.direction for t in self.units])
        self.torques = np.array(
            [
                np.cross(t.position - rcs.cm_offset, t.force * t.direction)
                for t in self.units
            ]
        )
        f_net = self.forces.sum(axis=0)
        if np.linalg.norm(f_net) == 0.0:
            raise ValueError(f"burn group {sk.burn.group!r} has zero net force")
        self.burn_dir = f_net / np.linalg.norm(f_net)

    def step(self, u: np.ndarray):
        """Duty per thruster for this cycle given the phase-plane command.
        Returns (duty, net force, net torque), cycle averages."""
        duty = np.ones(len(self.units))
        for axis in range(3):
            if u[axis] == 0:
                continue
            opposing = self.torques[:, axis] * u[axis] < -1e-9
            duty[opposing] -= self.sk.phase_plane.mod_depth
        duty = np.clip(duty, 0.0, 1.0)
        # PWM floor: per-cycle on-time below the minimum pulse is dropped,
        # full-on passes through
        min_duty = self.rcs.min_on_time_s / self.dt
        duty[(duty > 0.0) & (duty < min_duty)] = 0.0
        force = duty @ self.forces
        torque = duty @ self.torques
        return duty, force, torque


def thruster_table(rcs: RcsConfig) -> str:
    """Geometry table: locations, directions, force, torque about the
    nominal CM (the design values; burns use the actual, offset CM)."""
    lines = [
        f"{'name':>5} {'position [m]':>21} {'direction':>21} {'F [N]':>6} "
        f"{'torque about CM [N*m]':>24}",
    ]
    for t in rcs.thrusters:
        tau = np.cross(t.position, t.force * t.direction)
        pos = " ".join(f"{v:6.2f}" for v in t.position)
        d = " ".join(f"{v:6.3f}" for v in t.direction)
        tq = " ".join(f"{v:7.2f}" for v in tau)
        lines.append(f"{t.name:>5} [{pos}] [{d}] {t.force:6.1f} [{tq}]")
    for group, members in rcs.groups.items():
        units = {t.name: t for t in rcs.thrusters}
        f = sum(units[m].force * units[m].direction for m in members)
        tq = sum(
            np.cross(units[m].position, units[m].force * units[m].direction)
            for m in members
        )
        lines.append(
            f"group {group:>6}: net force [{f[0]:6.1f} {f[1]:6.1f} {f[2]:6.1f}] N, "
            f"net torque [{tq[0]:6.2f} {tq[1]:6.2f} {tq[2]:6.2f}] N*m"
        )
    return "\n".join(lines)
