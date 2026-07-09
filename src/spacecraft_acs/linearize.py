"""Per-axis linearization and frequency-domain analysis.

The nonlinear model is linearized about the nadir-pointing operating point.
For a near-diagonal inertia tensor the three axes decouple into SISO loops
(orbit-rate gyroscopic coupling, ~7e-5 rad/s, is negligible at control
frequencies). Cross-axis flexible coupling through off-axis participation
components is likewise neglected per axis; this is the standard preliminary-
design simplification and is noted in the README.
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


def controller_tf(config: Config, axis: int, with_delay: bool = True) -> control.TransferFunction:
    """Controller TF (attitude error -> torque) including PID, filters, and
    the sampling/computation delay as a 2nd-order Padé approximation."""
    pid = QuaternionPID(config.controller, np.diag(config.spacecraft.inertia))
    num, den = pid.analog_tf(axis)
    c = control.tf(num, den)
    if with_delay:
        # ZOH contributes ~T/2 of effective delay at loop frequencies
        t_delay = 0.5 / config.controller.rate_hz + config.controller.delay_s
        if t_delay > 0.0:
            pade_num, pade_den = control.pade(t_delay, 2)
            c = c * control.tf(pade_num, pade_den)
    return c


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
            + [5.0 * m.freq_hz for m in config.spacecraft.modes]
        )
    w = 2.0 * np.pi * np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    g = plant_ss(config, axis)
    c = controller_tf(config, axis)
    loop = control.minreal(c * control.tf(g), verbose=False)

    resp = loop(1j * w)
    mag_db = 20.0 * np.log10(np.abs(resp))
    phase_deg = np.rad2deg(np.unwrap(np.angle(resp)))
    freq_hz = w / (2.0 * np.pi)
    # Pick the 360-degree branch that places the phase at gain crossover (or
    # at low frequency) within (-360, 0], so Bode/Nichols plots and the
    # PM = 180 + phase annotation land on the same curve.
    ref_idx = int(np.argmin(np.abs(mag_db))) if np.any(mag_db > 0) else 0
    phase_deg -= 360.0 * np.ceil(phase_deg[ref_idx] / 360.0)

    gm, pm, wcg, wcp = control.margin(loop)
    gm_db = 20.0 * np.log10(gm) if gm not in (None, np.inf) and gm > 0 else None
    pm_deg = pm if pm not in (None, np.inf) else None
    phase_crossover_hz = wcg / (2 * np.pi) if wcg not in (None, np.inf) and wcg > 0 else None
    gain_crossover_hz = wcp / (2 * np.pi) if wcp not in (None, np.inf) and wcp > 0 else None

    t_cl = control.feedback(loop, 1)
    s_cl = control.feedback(1, loop)
    cl_mag_db = 20.0 * np.log10(np.abs(t_cl(1j * w)))
    sens_mag_db = 20.0 * np.log10(np.abs(s_cl(1j * w)))

    # Closed-loop bandwidth: last -3 dB downward crossing of |T|
    below = np.nonzero(cl_mag_db < -3.0)[0]
    above = np.nonzero(cl_mag_db >= -3.0)[0]
    cl_bandwidth_hz = None
    if above.size and below.size:
        first_below_after = below[below > above[0]]
        if first_below_after.size:
            cl_bandwidth_hz = freq_hz[first_below_after[0]]

    # Peak open-loop gain around each flexible resonance (gain-stabilization
    # check): |L| peak within +/-15% of the coupled (free-free) frequency,
    # which sits above the cantilever frequency by sqrt(J/(J - l^2)). The
    # +/-15% window represents modal frequency uncertainty the notches must
    # cover. Modes with negligible participation on this axis produce no
    # resonance in this loop and are skipped.
    j_axis = config.spacecraft.inertia[axis, axis]
    mode_gain_db = []
    for m in config.spacecraft.modes:
        l_ax = m.participation[axis]
        if l_ax**2 / j_axis < 1e-4:
            continue
        f_coupled = m.freq_hz * np.sqrt(j_axis / (j_axis - l_ax**2))
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
