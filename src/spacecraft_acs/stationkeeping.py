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
    """Per-axis bang-bang with deadband, rate lead, hysteresis, rate limit.

    Equivalent to the classic PD-relay form: sigma = kP*theta + kD*omega
    with kP = 1, kD = rate_lead_s, fed through a relay with deadband and
    hysteresis. Optional structural filters (the "Filter" block of the
    PD-equivalent diagram) run on sigma so flexible-mode content in the
    rate estimate cannot chatter the relay.
    """

    def __init__(self, cfg: PhasePlaneConfig, dt: float = 0.0625):
        self.cfg = cfg
        self.db = np.deg2rad(cfg.deadband_deg)
        self.rate_lim = np.deg2rad(cfg.rate_limit_dps)
        self._firing = np.zeros(3, dtype=int)  # current command per axis
        from .controller import _Biquad

        self._filters = [_Biquad(fc, dt) for fc in cfg.filters]

    def step(self, theta_err: np.ndarray, omega_err: np.ndarray) -> np.ndarray:
        """Return the commanded torque sign per axis in {-1, 0, +1}."""
        s = theta_err + self.cfg.rate_lead_s * omega_err
        for f in self._filters:
            s = f.step(s)
        w_dr = np.deg2rad(self.cfg.min_drift_rate_dps)
        u = np.zeros(3, dtype=int)
        for i in range(3):
            if abs(omega_err[i]) > self.rate_lim:
                u[i] = -int(np.sign(omega_err[i]))  # rate damping
            elif s[i] > self.db:
                # right of the hold channel: coast if already drifting back
                # (omega <= -w_dr, the drift channel); else fire negative
                # until that drift rate is established
                u[i] = 0 if omega_err[i] <= -w_dr else -1
            elif s[i] < -self.db:
                u[i] = 0 if omega_err[i] >= +w_dr else +1
            # else: inside the attitude hold channel -> coast
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


class HoldController:
    """Zero-delta-V attitude hold: the phase plane fires pure-torque couples
    (opposite-face thruster sets whose net force is nominally zero) with
    minimum-impulse pulses."""

    AXES = ("roll", "pitch", "yaw")

    def __init__(self, rcs: RcsConfig, sk: StationkeepingConfig, dt: float):
        self.rcs = rcs
        self.sk = sk
        self.dt = dt
        if not rcs.couples:
            raise ValueError("attitude hold requires rcs.couples")
        units = {t.name: t for t in rcs.thrusters}
        self.units = list(rcs.thrusters)
        index = {t.name: i for i, t in enumerate(self.units)}
        self.forces = np.array([t.force * t.direction for t in self.units])
        self.torques = np.array(
            [
                np.cross(t.position - rcs.cm_offset, t.force * t.direction)
                for t in self.units
            ]
        )
        # couple membership masks per (axis, sign)
        self._masks = {}
        for axis_i, axis in enumerate(self.AXES):
            for sign, tag in ((+1, "+"), (-1, "-")):
                key = f"{axis}{tag}"
                if key not in rcs.couples:
                    raise ValueError(f"rcs.couples missing {key!r}")
                mask = np.zeros(len(self.units), dtype=bool)
                for name in rcs.couples[key]:
                    mask[index[name]] = True
                self._masks[(axis_i, sign)] = mask
        # minimum-impulse pulse duty per commanded cycle
        self.pulse_duty = min(1.0, rcs.min_on_time_s / dt)

    def step(self, u: np.ndarray):
        """Duty per thruster given the phase-plane command: each commanded
        axis fires its couple for one minimum-impulse pulse this cycle."""
        duty = np.zeros(len(self.units))
        for axis in range(3):
            if u[axis] != 0:
                duty[self._masks[(axis, int(u[axis]))]] = self.pulse_duty
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
