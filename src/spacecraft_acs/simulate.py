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
from .estimator import Mekf
from .guidance import Guidance
from .momentum import MomentumManager
from .sensors import SensorSuite
from .stationkeeping import BurnController, HoldController, PhasePlane


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
    torque_ff: np.ndarray  # (K, 3) feedforward component of torque_cmd, N*m
    torque_thruster: np.ndarray  # (K, 3) unload thruster torque, N*m
    unloading: np.ndarray  # (K,) bool, momentum unload active
    est_att_err: np.ndarray  # (K, 3) estimator attitude error, rad (0 if off)
    est_bias_err: np.ndarray  # (K, 3) bias estimate - true bias, rad/s
    est_sigma: np.ndarray  # (K, 6) filter 1-sigma [att (rad), bias (rad/s)]
    burning: np.ndarray  # (K,) bool, stationkeeping burn active
    delta_v: np.ndarray  # (K, 3) accumulated delta-V, body axes, m/s
    pp_command: np.ndarray  # (K, 3) phase-plane torque command in {-1,0,1}
    rcs_duty: np.ndarray  # (K, n_burn_thrusters) duty per cycle
    att_err: np.ndarray  # (K, 3) attitude error rotation vector, rad
    unload_propellant_kg: float = 0.0  # RCS propellant spent on wheel unloads
    config: Config = field(repr=False, default=None)

    @property
    def att_err_deg(self) -> np.ndarray:
        return np.rad2deg(self.att_err)


def run(config: Config) -> SimResult:
    """Closed-loop simulation: discrete controller with ZOH torque, RK4
    integration of the nonlinear flexible dynamics between samples."""
    sc = FlexibleSpacecraft(config.spacecraft)
    wheels = ReactionWheelSet(config.wheels)
    guidance = Guidance(config.guidance, config.orbit_rate)
    env = Environment(
        config.environment, config.spacecraft.inertia, config.orbit_rate
    )
    pid = QuaternionPID(config.controller, np.diag(config.spacecraft.inertia))

    dt_ctrl = 1.0 / config.controller.rate_hz
    dt_int = dt_ctrl / config.simulation.substeps
    n_samples = int(np.floor(config.simulation.duration_s / dt_ctrl)) + 1

    sensors = SensorSuite(config.sensors, dt_ctrl)
    momentum_mgr = MomentumManager(config.unload, config.rcs, dt_ctrl)

    # Stationkeeping thruster modes: delta-V burn (off-pulsed group) or
    # zero-delta-V attitude hold (pure-torque couples); wheels held in both
    burn_ctrl = None
    hold_ctrl = None
    if config.stationkeeping.hold and config.rcs.thrusters:
        hold_ctrl = HoldController(config.rcs, config.stationkeeping, dt_ctrl)
        phase_plane = PhasePlane(config.stationkeeping.phase_plane, dt_ctrl)
    elif config.stationkeeping.burn.delta_v > 0.0 and config.rcs.thrusters:
        burn_ctrl = BurnController(config.rcs, config.stationkeeping, dt_ctrl)
        phase_plane = PhasePlane(config.stationkeeping.phase_plane, dt_ctrl)
    thr_ctrl = hold_ctrl or burn_ctrl
    n_burn = len(thr_ctrl.units) if thr_ctrl else 1
    dv = np.zeros(3)
    dv_done = False

    # Start on the initial command with matching body rate, offset by the
    # configured initial attitude error (rotation vector, deg)
    q0_cmd, w0_cmd, _ = guidance.command(0.0)
    q0 = q0_cmd
    err0 = np.deg2rad(config.simulation.initial_att_err_deg)
    if np.any(err0):
        q0 = qt.multiply(q0_cmd, qt.from_rotation_vector(err0))
    x = sc.initial_state(
        q=q0, omega=w0_cmd, h_wheel=config.simulation.initial_wheel_momentum
    )

    # MEKF: gyro propagation every cycle, star tracker updates decimated to
    # the configured rate. Initialized on the commanded attitude.
    mekf = None
    st_every = 1
    if config.estimator.enabled:
        mekf = Mekf(config.estimator, config.sensors, dt_ctrl, q0_cmd)
        st_every = max(
            1, round(config.controller.rate_hz / config.estimator.star_tracker_rate_hz)
        )

    out = {
        name: np.zeros((n_samples, dim))
        for name, dim in [
            ("q", 4), ("omega", 3), ("h_wheel", 3), ("q_cmd", 4),
            ("torque_cmd", 3), ("torque_applied", 3), ("torque_dist", 3),
            ("torque_ff", 3), ("torque_thruster", 3), ("att_err", 3),
        ]
    }
    out["eta"] = np.zeros((n_samples, sc.n_modes))
    out["eta_dot"] = np.zeros((n_samples, sc.n_modes))
    out["unloading"] = np.zeros(n_samples, dtype=bool)
    out["est_att_err"] = np.zeros((n_samples, 3))
    out["est_bias_err"] = np.zeros((n_samples, 3))
    out["est_sigma"] = np.zeros((n_samples, 6))
    out["burning"] = np.zeros(n_samples, dtype=bool)
    out["delta_v"] = np.zeros((n_samples, 3))
    out["pp_command"] = np.zeros((n_samples, 3))
    out["rcs_duty"] = np.zeros((n_samples, n_burn))
    t_grid = np.arange(n_samples) * dt_ctrl

    saturated = False
    for k, t in enumerate(t_grid):
        q, omega, eta, eta_dot, h_w = sc.unpack(x)
        q_cmd, omega_cmd, alpha_cmd = guidance.command(t)
        q_meas, omega_meas = sensors.measure(q, omega)

        if mekf is not None:
            mekf.propagate(omega_meas)
            if k % st_every == 0:
                mekf.update_star_tracker(q_meas)
            q_used = mekf.q
            omega_used = mekf.rate_estimate(omega_meas)
            out["est_att_err"][k] = 2.0 * qt.error(q, mekf.q)[1:]
            out["est_bias_err"][k] = mekf.bias - sensors.bias
            out["est_sigma"][k] = mekf.sigma
        else:
            q_used, omega_used = q_meas, omega_meas

        # Thruster-control window: attitude hold runs the whole sim; a burn
        # starts at the configured time and ends when the accumulated
        # delta-V along the burn direction reaches the target
        burning = hold_ctrl is not None or (
            burn_ctrl is not None
            and t >= config.stationkeeping.burn.start_time_s
            and not dv_done
        )

        if burning:
            # Thruster attitude control: phase plane + allocation. Wheels
            # are held (no torque) — thruster torques would saturate them
            # in seconds.
            q_err_m = qt.error(q_cmd, q_used)
            theta_m = 2.0 * q_err_m[1:]
            omega_err_m = omega_used - qt.dcm(q_err_m) @ omega_cmd
            u_pp = phase_plane.step(theta_m, omega_err_m)
            duty, f_rcs, tau_rcs = thr_ctrl.step(u_pp)
            dv = dv + f_rcs / config.spacecraft.mass * dt_ctrl
            if (
                burn_ctrl is not None
                and dv @ burn_ctrl.burn_dir >= config.stationkeeping.burn.delta_v
            ):
                dv_done = True
            out["pp_command"][k] = u_pp
            out["rcs_duty"][k] = duty
            u_cmd = np.zeros(3)
            u_applied = np.zeros(3)
            t_thr = np.zeros(3)
            t_ff = np.zeros(3)
        else:
            tau_rcs = np.zeros(3)
            # Momentum unload: external thruster torque, optionally countered
            # by a wheel feedforward so pointing only sees the residual
            t_thr = momentum_mgr.update(h_w)

            # Rigid-body acceleration feedforward (command frame ~ body frame
            # at small tracking error), plus thruster compensation
            t_ff = np.zeros(3)
            if config.controller.feedforward and np.any(alpha_cmd):
                t_ff = t_ff + config.spacecraft.inertia @ alpha_cmd
            if config.unload.feedforward_compensation:
                t_ff = t_ff - t_thr
            u_cmd = pid.step(
                q_used, omega_used, q_cmd, omega_cmd,
                freeze_integrator=saturated,
                torque_ff=t_ff if np.any(t_ff) else None,
            )
            u_applied = wheels.apply(u_cmd, h_w)
            saturated = bool(np.any(np.abs(u_applied - u_cmd) > 1e-12))

        q_body_from_lvlh = qt.multiply(qt.conjugate(guidance.lvlh_attitude(t)), q)
        t_dist = env.torque(t, q_body_from_lvlh) + t_thr + tau_rcs

        q_err = qt.error(q_cmd, q)
        out["q"][k] = q
        out["omega"][k] = omega
        out["eta"][k] = eta
        out["eta_dot"][k] = eta_dot
        out["h_wheel"][k] = h_w
        out["q_cmd"][k] = q_cmd
        out["torque_cmd"][k] = u_cmd
        out["torque_applied"][k] = u_applied
        out["torque_dist"][k] = t_dist - t_thr - tau_rcs  # environment only
        out["torque_thruster"][k] = t_thr + tau_rcs
        out["unloading"][k] = momentum_mgr.unloading
        out["burning"][k] = burning
        out["delta_v"][k] = dv
        out["torque_ff"][k] = t_ff
        out["att_err"][k] = 2.0 * q_err[1:]

        if k == n_samples - 1:
            break
        # Torque held constant over the controller interval (ZOH); the
        # disturbance is also held, which is exact to first order since
        # environment torques vary on orbit timescales.
        for _ in range(config.simulation.substeps):
            x = sc.rk4_step(x, dt_int, u_applied, t_dist)

    return SimResult(
        t=t_grid, config=config,
        unload_propellant_kg=momentum_mgr.propellant_kg, **out,
    )


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


@dataclass
class ManeuverMetrics:
    """Comparable quantities for a commanded reorientation, whether executed
    as a discontinuous step or a profiled slew."""

    label: str
    angle_deg: float
    axis: int
    profile_duration_s: float | None  # None for a raw step
    settling_time_s: float | None  # from command start into the settling band
    overshoot_deg: float  # max excursion beyond the final target attitude
    peak_tracking_err_deg: float  # max |attitude error| during the maneuver
    peak_torque_nm: float
    peak_mode_disp: float  # max |eta| over all modes
    post_ringing_rms: float  # RMS of eta after maneuver end (flex ringing)

    def __str__(self) -> str:
        dur = "instantaneous" if self.profile_duration_s is None else (
            f"{self.profile_duration_s:.0f} s profile"
        )
        settle = "did not settle" if self.settling_time_s is None else (
            f"{self.settling_time_s:.0f} s"
        )
        return (
            f"{self.label}: {self.angle_deg:.1f} deg about {'xyz'[self.axis]} "
            f"({dur})\n"
            f"  settling time (from command): {settle}\n"
            f"  overshoot beyond target:      {self.overshoot_deg * 3600:.1f} arcsec "
            f"({100 * self.overshoot_deg / abs(self.angle_deg):.1f} % of maneuver)\n"
            f"  peak pointing error:          {self.peak_tracking_err_deg:.3f} deg\n"
            f"  peak wheel torque:            {self.peak_torque_nm:.3f} N*m\n"
            f"  peak modal displacement:      {self.peak_mode_disp:.3g}\n"
            f"  post-maneuver modal RMS:      {self.post_ringing_rms:.3g}"
        )


def maneuver_metrics(result: SimResult, label: str) -> ManeuverMetrics:
    """Metrics for the configured maneuver (step or profiled slew)."""
    cfg = result.config
    step = cfg.guidance.step
    axis = int(np.argmax(np.abs(step.axis)))
    t0 = step.time_s
    from .guidance import Guidance  # local import to avoid a cycle

    profile_dur = Guidance(cfg.guidance, cfg.orbit_rate).slew_duration
    t_man_end = t0 + (profile_dur if profile_dur is not None else 0.0)

    mask = result.t >= t0
    t = result.t[mask] - t0
    err = result.att_err_deg[mask, axis]

    # Overshoot beyond the final target: attitude error opposite the approach
    # direction after the command has reached its final value. The error
    # during approach has sign -sign(angle); excursion past target flips it.
    after_end = t >= (t_man_end - t0)
    sign = np.sign(step.angle_deg)
    overshoot = max(0.0, float(np.max(sign * err[after_end], initial=0.0)))

    band = cfg.simulation.settling_band * abs(step.angle_deg)
    outside = np.nonzero(np.abs(err) > band)[0]
    if outside.size == 0:
        settling = 0.0
    elif outside[-1] + 1 < len(t):
        settling = float(t[outside[-1] + 1])
    else:
        settling = None

    during = (result.t >= t0) & (result.t <= t_man_end + 1e-9) if profile_dur \
        else mask
    ringing_win = result.t >= (t_man_end + 60.0)
    return ManeuverMetrics(
        label=label,
        angle_deg=step.angle_deg,
        axis=axis,
        profile_duration_s=profile_dur,
        settling_time_s=settling,
        overshoot_deg=overshoot,
        peak_tracking_err_deg=float(np.max(np.abs(result.att_err_deg[during, axis]))),
        peak_torque_nm=float(np.max(np.abs(result.torque_applied[mask]))),
        peak_mode_disp=float(np.max(np.abs(result.eta[mask]))),
        post_ringing_rms=float(np.sqrt(np.mean(result.eta[ringing_win] ** 2)))
        if np.any(ringing_win)
        else 0.0,
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
