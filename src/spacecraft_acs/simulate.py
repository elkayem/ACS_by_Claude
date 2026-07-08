"""Nonlinear time-domain closed-loop simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quaternion as qt
from .actuators import ReactionWheelSet
from .config import Config
from .controller import QuaternionPID
from .dynamics import FlexibleSpacecraft
from .environment import Environment
from .guidance import Guidance
from .sensors import SensorSuite


@dataclass
class SimResult:
    t: np.ndarray  # (K,) controller sample times
    q: np.ndarray  # (K, 4) true attitude, inertial->body
    omega: np.ndarray  # (K, 3) true body rate, rad/s
    eta: np.ndarray  # (K, N) modal displacements
    eta_dot: np.ndarray  # (K, N)
    h_wheel: np.ndarray  # (K, 3) wheel momentum, N*m*s
    q_cmd: np.ndarray  # (K, 4)
    torque_cmd: np.ndarray  # (K, 3) controller output, N*m
    torque_applied: np.ndarray  # (K, 3) after wheel limits, N*m
    torque_dist: np.ndarray  # (K, 3) environment torque, N*m
    att_err: np.ndarray  # (K, 3) attitude error rotation vector, rad
    config: Config = field(repr=False, default=None)

    @property
    def att_err_deg(self) -> np.ndarray:
        return np.rad2deg(self.att_err)


def run(config: Config) -> SimResult:
    """Closed-loop simulation: discrete controller with ZOH torque, RK4
    integration of the nonlinear flexible dynamics between samples."""
    sc = FlexibleSpacecraft(config.spacecraft)
    wheels = ReactionWheelSet(config.wheels)
    sensors = SensorSuite(config.sensors)
    guidance = Guidance(config.guidance, config.orbit_rate)
    env = Environment(
        config.environment, config.spacecraft.inertia, config.orbit_rate
    )
    pid = QuaternionPID(config.controller, np.diag(config.spacecraft.inertia))

    dt_ctrl = 1.0 / config.controller.rate_hz
    dt_int = dt_ctrl / config.simulation.substeps
    n_samples = int(np.floor(config.simulation.duration_s / dt_ctrl)) + 1

    # Start on the initial command (nadir attitude) with matching body rate
    q0_cmd, w0_cmd = guidance.command(0.0)
    x = sc.initial_state(q=q0_cmd, omega=w0_cmd)

    out = {
        name: np.zeros((n_samples, dim))
        for name, dim in [
            ("q", 4), ("omega", 3), ("h_wheel", 3), ("q_cmd", 4),
            ("torque_cmd", 3), ("torque_applied", 3), ("torque_dist", 3),
            ("att_err", 3),
        ]
    }
    out["eta"] = np.zeros((n_samples, sc.n_modes))
    out["eta_dot"] = np.zeros((n_samples, sc.n_modes))
    t_grid = np.arange(n_samples) * dt_ctrl

    saturated = False
    for k, t in enumerate(t_grid):
        q, omega, eta, eta_dot, h_w = sc.unpack(x)
        q_cmd, omega_cmd = guidance.command(t)
        q_meas, omega_meas = sensors.measure(q, omega)

        u_cmd = pid.step(q_meas, omega_meas, q_cmd, omega_cmd, freeze_integrator=saturated)
        u_applied = wheels.apply(u_cmd, h_w)
        saturated = bool(np.any(np.abs(u_applied - u_cmd) > 1e-12))

        q_body_from_lvlh = qt.multiply(qt.conjugate(guidance.lvlh_attitude(t)), q)
        t_dist = env.torque(t, q_body_from_lvlh)

        q_err = qt.error(q_cmd, q)
        out["q"][k] = q
        out["omega"][k] = omega
        out["eta"][k] = eta
        out["eta_dot"][k] = eta_dot
        out["h_wheel"][k] = h_w
        out["q_cmd"][k] = q_cmd
        out["torque_cmd"][k] = u_cmd
        out["torque_applied"][k] = u_applied
        out["torque_dist"][k] = t_dist
        out["att_err"][k] = 2.0 * q_err[1:]

        if k == n_samples - 1:
            break
        # Torque held constant over the controller interval (ZOH); the
        # disturbance is also held, which is exact to first order since
        # environment torques vary on orbit timescales.
        for _ in range(config.simulation.substeps):
            x = sc.rk4_step(x, dt_int, u_applied, t_dist)

    return SimResult(t=t_grid, config=config, **out)


@dataclass
class StepMetrics:
    axis: int  # index of the stepped axis
    step_deg: float
    rise_time_s: float | None  # 10% -> 90%
    overshoot_pct: float | None
    settling_time_s: float | None  # into settling band, measured from step time

    def __str__(self) -> str:
        def fmt(v, unit=""):
            return "n/a" if v is None else f"{v:.1f}{unit}"

        return (
            f"step of {self.step_deg:.1f} deg on axis {'xyz'[self.axis]}: "
            f"rise time {fmt(self.rise_time_s, ' s')}, "
            f"overshoot {fmt(self.overshoot_pct, ' %')}, "
            f"settling time {fmt(self.settling_time_s, ' s')} "
            f"(from step command)"
        )


def step_metrics(result: SimResult) -> StepMetrics:
    """Rise time, overshoot, and settling time of the commanded step,
    evaluated on the attitude error about the step axis."""
    cfg = result.config
    step = cfg.guidance.step
    axis = int(np.argmax(np.abs(step.axis)))
    step_rad = np.deg2rad(step.angle_deg)
    t0 = step.time_s

    mask = result.t >= t0
    t = result.t[mask] - t0
    # Error jumps to -step at the command and recovers to 0; track progress
    # toward the new command as a 0 -> 1 response.
    err = result.att_err[mask, axis] * np.sign(-step_rad)
    response = 1.0 - err / abs(step_rad)

    def first_crossing(level):
        idx = np.nonzero(response >= level)[0]
        return t[idx[0]] if idx.size else None

    t10, t90 = first_crossing(0.1), first_crossing(0.9)
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    overshoot = max(0.0, (np.max(response) - 1.0) * 100.0) if t90 is not None else None

    band = cfg.simulation.settling_band
    outside = np.nonzero(np.abs(response - 1.0) > band)[0]
    if outside.size == 0:
        settling = t[0]
    elif outside[-1] + 1 < len(t):
        settling = t[outside[-1] + 1]
    else:
        settling = None  # never settles within the run
    return StepMetrics(
        axis=axis,
        step_deg=step.angle_deg,
        rise_time_s=rise,
        overshoot_pct=overshoot,
        settling_time_s=settling,
    )
