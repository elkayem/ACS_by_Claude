"""Linearization and frequency-domain analysis.

The nonlinear model is linearized about the nadir-pointing operating point
(orbit-rate gyroscopic coupling, ~7e-5 rad/s, is negligible at control
frequencies). Margins are computed loop-at-a-time on the coupled 3-axis
flexible plant: one axis's loop is broken for measurement while the other
two remain closed, so a mode that couples into several axes is credited with
the damping the closed loops provide — the honest margin for a coupled
flexible plant, and it matters for dispersed plants whose participation
vectors mix axes. Closing the measured loop then yields the full coupled
closed-loop poles, so the stability verdict matches the nonlinear sim's
dynamics by construction. `plant_ss` (single-axis SISO plant) is retained
for reference and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import control
import numpy as np

from .config import Config
from .controller import QuaternionPID


def plant_ss(config: Config, axis: int) -> control.StateSpace:
    """Torque -> attitude angle state-space for one body axis.

    States [θ, ω, η, η̇] with the axis participation column l = L[axis, :]:
        [J l; lᵀ I] [ω̇; η̈] = [T; −2ZΩη̇ − Ω²η]
    """
    sc = config.spacecraft
    j = sc.inertia[axis, axis]
    l = sc.participation_matrix[axis, :]
    n = len(l)
    wn = sc.mode_freqs
    zeta = sc.mode_dampings

    m = np.block([[np.array([[j]]), l[None, :]], [l[:, None], np.eye(n)]])
    m_inv = np.linalg.inv(m)

    # x = [theta, omega, eta, eta_dot]
    a = np.zeros((2 + 2 * n, 2 + 2 * n))
    b = np.zeros((2 + 2 * n, 1))
    a[0, 1] = 1.0  # theta_dot = omega
    a[2 : 2 + n, 2 + n :] = np.eye(n)  # eta_dot
    # [omega_dot; eta_ddot] = m_inv @ [T; -2 Z Omega eta_dot - Omega^2 eta]
    rhs_a = np.zeros((1 + n, 2 + 2 * n))
    rhs_a[1:, 2 : 2 + n] = -np.diag(wn**2)
    rhs_a[1:, 2 + n :] = -np.diag(2.0 * zeta * wn)
    accel_rows = m_inv @ rhs_a
    a[1, :] = accel_rows[0, :]
    a[2 + n :, :] = accel_rows[1:, :]
    rhs_b = np.zeros((1 + n, 1))
    rhs_b[0, 0] = 1.0
    accel_b = m_inv @ rhs_b
    b[1, 0] = accel_b[0, 0]
    b[2 + n :, 0] = accel_b[1:, 0]

    c = np.zeros((1, 2 + 2 * n))
    c[0, 0] = 1.0
    return control.ss(a, b, c, 0.0)


def coupled_plant_ss(config: Config) -> control.StateSpace:
    """Full 3-axis torque -> attitude plant: states [θ(3), ω(3), η, η̇] with
    the complete inertia tensor and participation matrix, so cross-axis
    flexible coupling is retained."""
    sc = config.spacecraft
    j = sc.inertia
    l_mat = sc.participation_matrix
    n = l_mat.shape[1]
    wn = sc.mode_freqs
    zeta = sc.mode_dampings

    m = np.block([[j, l_mat], [l_mat.T, np.eye(n)]])
    m_inv = np.linalg.inv(m)

    nx = 6 + 2 * n  # [theta(3), omega(3), eta(n), eta_dot(n)]
    a = np.zeros((nx, nx))
    b = np.zeros((nx, 3))
    a[0:3, 3:6] = np.eye(3)  # theta_dot = omega
    a[6 : 6 + n, 6 + n :] = np.eye(n)  # eta_dot
    # [omega_dot; eta_ddot] = m_inv @ [T; -2 Z Omega eta_dot - Omega^2 eta]
    rhs_a = np.zeros((3 + n, nx))
    rhs_a[3:, 6 : 6 + n] = -np.diag(wn**2)
    rhs_a[3:, 6 + n :] = -np.diag(2.0 * zeta * wn)
    accel_rows = m_inv @ rhs_a
    a[3:6, :] = accel_rows[:3, :]
    a[6 + n :, :] = accel_rows[3:, :]
    rhs_b = np.zeros((3 + n, 3))
    rhs_b[:3, :] = np.eye(3)
    accel_b = m_inv @ rhs_b
    b[3:6, :] = accel_b[:3, :]
    b[6 + n :, :] = accel_b[3:, :]

    c = np.zeros((3, nx))
    c[:, 0:3] = np.eye(3)
    return control.ss(a, b, c, np.zeros((3, 3)))


def broken_loop_ss(config: Config, axis: int, f_max: float = 100.0) -> control.StateSpace:
    """Loop-at-a-time open loop for one axis: the coupled 3-axis plant with
    the OTHER two control loops closed, times this axis's controller. This is
    the honest margin for a coupled flexible plant — a mode that couples into
    several axes is actively damped by the closed loops while one loop is
    broken for measurement."""
    g = coupled_plant_ss(config)
    parts = []
    for j in range(3):
        if j == axis:
            parts.append(control.ss([], [], [], 0.0))  # broken loop
        else:
            parts.append(control.ss(controller_tf(config, j, f_max=f_max)))
    c_other = control.append(*parts)
    g_closed = control.feedback(g, c_other)  # other loops closed
    c_axis = control.ss(controller_tf(config, axis, f_max=f_max))
    return c_axis * g_closed[axis, axis]


def controller_tf(
    config: Config, axis: int, with_delay: bool = True, f_max: float = 100.0
) -> control.TransferFunction:
    """Controller TF (attitude error -> torque) including PID, filters, and
    the sampling/computation delay as a 2nd-order Padé approximation.

    If the result is improper (e.g. pure PD with no roll-off filters), far
    poles at 50*f_max are appended so the TF converts to state space; their
    phase contribution below f_max is negligible."""
    pid = QuaternionPID(config.controller, np.diag(config.spacecraft.inertia))
    num, den = pid.analog_tf(axis)
    if with_delay:
        # ZOH contributes ~T/2 of effective delay at loop frequencies
        t_delay = 0.5 / config.controller.rate_hz + config.controller.delay_s
        if t_delay > 0.0:
            pade_num, pade_den = control.pade(t_delay, 2)
            num = np.polymul(num, pade_num)
            den = np.polymul(den, pade_den)
    deficit = (len(num) - 1) - (len(den) - 1)
    if deficit > 0:
        w_far = 2.0 * np.pi * 50.0 * f_max
        for _ in range(deficit):
            den = np.polymul(den, np.array([1.0 / w_far, 1.0]))
    return control.tf(num, den)


@dataclass
class AxisFrequencyData:
    axis: int
    freq_hz: np.ndarray
    mag_db: np.ndarray  # open loop L(jw)
    phase_deg: np.ndarray  # unwrapped
    gm_db: float | None
    pm_deg: float | None
    gain_crossover_hz: float | None
    phase_crossover_hz: float | None
    cl_mag_db: np.ndarray  # complementary sensitivity T
    sens_mag_db: np.ndarray  # sensitivity S
    cl_bandwidth_hz: float | None
    cl_peak_db: float
    sens_peak_db: float
    cl_poles: np.ndarray
    mode_gain_db: list  # (freq_hz, |L| dB at each flex resonance)


def analyze_axis(config: Config, axis: int, f_min=1e-4, f_max=None, n_points=4000) -> AxisFrequencyData:
    if f_max is None:
        # Cover the flex modes and the Nyquist neighborhood
        f_max = max(
            [2.0 * config.controller.rate_hz]
            + [5.0 * m.freq_hz for m in config.spacecraft.all_modes]
        )
    f_grid = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    # A lightly damped mode (zeta ~ 1e-3) has a fractional width far below
    # the base log-grid spacing, so margins computed from the frequency
    # response would under-sample its pole/zero ripple (spurious or missed
    # unity crossings). Densify around every mode's cantilever-to-coupled
    # band, padded by several damping half-widths.
    j_diag = np.diag(config.spacecraft.inertia)
    clusters = []
    for m in config.spacecraft.all_modes:
        shift = np.sqrt(
            np.max(j_diag / np.maximum(j_diag - m.participation**2, 1e-30))
        )
        pad = 1.0 + 20.0 * max(m.damping, 1e-4)
        lo = m.freq_hz / pad
        hi = m.freq_hz * shift * pad
        clusters.append(np.linspace(lo, hi, 300))
    if clusters:
        f_grid = np.unique(np.concatenate([f_grid] + clusters))
        f_grid = f_grid[(f_grid >= f_min) & (f_grid <= f_max)]
    w = 2.0 * np.pi * f_grid

    # Loop-at-a-time open loop on the coupled plant (other loops closed),
    # built entirely in state space: polynomial (transfer-function)
    # arithmetic on the flexible plant overflows float64 coefficient ranges
    # and produces spurious poles for strongly dispersed plants.
    loop = broken_loop_ss(config, axis, f_max=f_max)

    resp = loop(1j * w)
    mag_db = 20.0 * np.log10(np.abs(resp))
    phase_deg = np.rad2deg(np.unwrap(np.angle(resp)))
    freq_hz = w / (2.0 * np.pi)
    # Pick the 360-degree branch that places the phase at gain crossover (or
    # at low frequency) within (-360, 0], so Bode/Nichols plots and the
    # PM = 180 + phase annotation land on the same curve.
    ref_idx = int(np.argmin(np.abs(mag_db))) if np.any(mag_db > 0) else 0
    phase_deg -= 360.0 * np.ceil(phase_deg[ref_idx] / 360.0)

    # Margins from the frequency response (FRD) rather than polynomial root
    # finding, for the same numerical-robustness reason as above
    gm, pm, wcg, wcp = control.margin(control.frd(loop, w))
    gm_db = 20.0 * np.log10(gm) if gm not in (None, np.inf) and gm > 0 else None
    pm_deg = pm if pm not in (None, np.inf) else None
    phase_crossover_hz = wcg / (2 * np.pi) if wcg not in (None, np.inf) and wcg > 0 else None
    gain_crossover_hz = wcp / (2 * np.pi) if wcp not in (None, np.inf) and wcp > 0 else None

    t_cl = control.feedback(loop, 1)  # state space in, state space out
    s_cl = control.feedback(control.ss([], [], [], 1.0), loop)
    cl_mag_db = 20.0 * np.log10(np.abs(t_cl(1j * w)))
    sens_mag_db = 20.0 * np.log10(np.abs(s_cl(1j * w)))

    # Closed-loop bandwidth: LAST -3 dB downward crossing of |T| (a slosh
    # zero can notch |T| below -3 dB well inside the band; the band edge is
    # where |T| leaves -3 dB for good)
    above3 = np.nonzero(cl_mag_db >= -3.0)[0]
    cl_bandwidth_hz = None
    if above3.size and above3[-1] + 1 < len(freq_hz):
        cl_bandwidth_hz = freq_hz[above3[-1] + 1]

    # Peak open-loop gain around each flexible resonance (gain-stabilization
    # check): |L| peak within +/-15% of the coupled (free-free) frequency,
    # which sits above the cantilever frequency by sqrt(J/(J - l^2)). The
    # +/-15% window represents modal frequency uncertainty the notches must
    # cover. Modes with negligible participation on this axis produce no
    # resonance in this loop and are skipped — as are IN-BAND modes (slosh,
    # at or below the crossover region): those cannot be gain-stabilized by
    # definition, are phase-stabilized instead, and their health is judged
    # by the margins and the coupled closed-loop poles. "In band" is judged
    # against the LAST downward 0-dB crossing of |L|: low-frequency slosh
    # ripple can create early unity crossings that margin() may report as
    # the crossover, but gain stabilization only applies above the final one.
    above = np.nonzero(mag_db > 0.0)[0]
    f_band_edge = freq_hz[above[-1]] if above.size else 0.0
    j_axis = config.spacecraft.inertia[axis, axis]
    mode_gain_db = []
    for m in config.spacecraft.all_modes:
        l_ax = m.participation[axis]
        if l_ax**2 / j_axis < 1e-4:
            continue
        f_coupled = m.freq_hz * np.sqrt(j_axis / (j_axis - l_ax**2))
        if f_coupled <= 1.2 * f_band_edge:
            continue
        window = (freq_hz >= 0.85 * f_coupled) & (freq_hz <= 1.15 * f_coupled)
        mode_gain_db.append((f_coupled, float(np.max(mag_db[window]))))

    return AxisFrequencyData(
        axis=axis,
        freq_hz=freq_hz,
        mag_db=mag_db,
        phase_deg=phase_deg,
        gm_db=gm_db,
        pm_deg=pm_deg,
        gain_crossover_hz=gain_crossover_hz,
        phase_crossover_hz=phase_crossover_hz,
        cl_mag_db=cl_mag_db,
        sens_mag_db=sens_mag_db,
        cl_bandwidth_hz=cl_bandwidth_hz,
        cl_peak_db=float(np.max(cl_mag_db)),
        sens_peak_db=float(np.max(sens_mag_db)),
        cl_poles=t_cl.poles(),
        mode_gain_db=mode_gain_db,
    )


def analyze(config: Config) -> list[AxisFrequencyData]:
    return [analyze_axis(config, axis) for axis in range(3)]


def report(data: list[AxisFrequencyData]) -> str:
    """Human-readable margin summary for all three axes."""
    lines = []
    for d in data:
        lines.append(f"--- {'xyz'[d.axis]}-axis ({['roll', 'pitch', 'yaw'][d.axis]}) ---")
        gm = "inf" if d.gm_db is None else f"{d.gm_db:.1f} dB at {d.phase_crossover_hz * 1e3:.2f} mHz"
        pm = "n/a" if d.pm_deg is None else f"{d.pm_deg:.1f} deg at {d.gain_crossover_hz * 1e3:.2f} mHz"
        lines.append(f"  gain margin:  {gm}")
        lines.append(f"  phase margin: {pm}")
        bw = "n/a" if d.cl_bandwidth_hz is None else f"{d.cl_bandwidth_hz * 1e3:.2f} mHz"
        lines.append(f"  closed-loop bandwidth (-3 dB): {bw}")
        lines.append(f"  closed-loop peaks: Mt = {d.cl_peak_db:.1f} dB, Ms = {d.sens_peak_db:.1f} dB")
        for f_mode, g_mode in d.mode_gain_db:
            status = "gain-stabilized" if g_mode < -6.0 else "NOT gain-stabilized (check phase)"
            lines.append(
                f"  flex mode at {f_mode:.3f} Hz (coupled, +/-15%): "
                f"|L| = {g_mode:.1f} dB ({status})"
            )
        unstable = [p for p in d.cl_poles if p.real > 1e-9]
        if unstable:
            lines.append(f"  WARNING: {len(unstable)} unstable closed-loop pole(s)!")
    return "\n".join(lines)
