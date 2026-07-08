"""Plot generation for time- and frequency-domain analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXIS_LABELS = ["roll (x)", "pitch (y)", "yaw (z)"]
AXIS_COLORS = ["#c0392b", "#27ae60", "#2980b9"]


def _save(fig, output_dir: Path, name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_step_response(result, metrics, output_dir: Path) -> Path:
    """Attitude error, body rates, wheel torque/momentum, modal response."""
    cfg = result.config
    t = result.t
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    ax = axes[0]
    for i in range(3):
        ax.plot(t, result.att_err_deg[:, i], color=AXIS_COLORS[i], label=AXIS_LABELS[i])
    ax.axvline(cfg.guidance.step.time_s, color="0.5", ls=":", label="step command")
    ax.set_ylabel("attitude error [deg]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Attitude step response — {metrics.step_deg:.1f}° about "
        f"{'xyz'[metrics.axis]} at t={cfg.guidance.step.time_s:.0f} s\n"
        f"rise {_fmt(metrics.rise_time_s, 's')}, "
        f"overshoot {_fmt(metrics.overshoot_pct, '%')}, "
        f"settling ({100 * cfg.simulation.settling_band:.0f}% band) "
        f"{_fmt(metrics.settling_time_s, 's')}",
        fontsize=10,
    )

    ax = axes[1]
    for i in range(3):
        ax.plot(t, np.rad2deg(result.omega[:, i]) * 3600.0, color=AXIS_COLORS[i])
    ax.set_ylabel("body rate [deg/hr]")

    ax = axes[2]
    for i in range(3):
        ax.plot(t, result.torque_applied[:, i], color=AXIS_COLORS[i])
        ax.plot(t, result.torque_cmd[:, i], color=AXIS_COLORS[i], alpha=0.3, lw=0.8)
    if not cfg.wheels.ideal:
        for s in (1, -1):
            ax.axhline(s * cfg.wheels.max_torque, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel("wheel torque [N·m]\n(faint: commanded)")

    ax = axes[3]
    for i in range(3):
        ax.plot(t, result.h_wheel[:, i], color=AXIS_COLORS[i])
    if not cfg.wheels.ideal:
        for s in (1, -1):
            ax.axhline(s * cfg.wheels.max_momentum, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel("wheel momentum [N·m·s]")

    ax = axes[4]
    for j, mode in enumerate(cfg.spacecraft.modes):
        ax.plot(t, result.eta[:, j], label=f"mode {j + 1} ({mode.freq_hz:.2f} Hz)")
    ax.set_ylabel("modal displacement\n[√kg·m·rad]")
    ax.set_xlabel("time [s]")
    if cfg.spacecraft.modes:
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    return _save(fig, output_dir, "step_response")


def plot_bode(freq_data, output_dir: Path) -> list[Path]:
    """Open-loop Bode plots, one figure per axis, margins annotated."""
    paths = []
    for d in freq_data:
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        f = d.freq_hz
        ax_mag.semilogx(f, d.mag_db, color="#2980b9")
        ax_mag.axhline(0.0, color="0.4", lw=0.8)
        ax_ph.semilogx(f, d.phase_deg, color="#2980b9")
        ax_ph.axhline(-180.0, color="0.4", lw=0.8)

        if d.gain_crossover_hz is not None:
            idx = np.argmin(np.abs(d.freq_hz - d.gain_crossover_hz))
            ax_mag.axvline(d.gain_crossover_hz, color="#27ae60", ls=":", lw=1)
            ax_ph.axvline(d.gain_crossover_hz, color="#27ae60", ls=":", lw=1)
            ax_ph.plot(d.gain_crossover_hz, d.phase_deg[idx], "o", color="#27ae60")
            ax_ph.annotate(
                f"PM = {d.pm_deg:.1f}°",
                (d.gain_crossover_hz, d.phase_deg[idx]),
                textcoords="offset points", xytext=(8, 6), color="#27ae60",
            )
        if d.phase_crossover_hz is not None:
            ax_mag.plot(d.phase_crossover_hz, -d.gm_db, "o", color="#c0392b")
            ax_mag.annotate(
                f"GM = {d.gm_db:.1f} dB",
                (d.phase_crossover_hz, -d.gm_db),
                textcoords="offset points", xytext=(8, 6), color="#c0392b",
            )
        ax_mag.set_ylabel("|L(jω)| [dB]")
        ax_ph.set_ylabel("∠L(jω) [deg]")
        ax_ph.set_xlabel("frequency [Hz]")
        ax_mag.set_title(
            f"Open-loop Bode — {AXIS_LABELS[d.axis]}   "
            f"(GM {_fmt(d.gm_db, 'dB')}, PM {_fmt(d.pm_deg, 'deg')})"
        )
        for ax in (ax_mag, ax_ph):
            ax.grid(True, which="both", alpha=0.3)
        paths.append(_save(fig, output_dir, f"bode_{'xyz'[d.axis]}"))
    return paths


def plot_nichols(freq_data, output_dir: Path) -> list[Path]:
    """Open-loop Nichols charts with margin call-outs."""
    paths = []
    for d in freq_data:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(d.phase_deg, d.mag_db, color="#2980b9", lw=1.2)
        ax.plot(-180.0, 0.0, "+", color="k", ms=12, mew=2)

        # 6 dB / 30 deg margin box guide around the critical point
        ax.add_patch(
            plt.Rectangle((-210, -6), 60, 12, fill=False, ls="--", ec="0.5", lw=0.8)
        )
        if d.gain_crossover_hz is not None:
            idx = np.argmin(np.abs(d.freq_hz - d.gain_crossover_hz))
            ax.plot(d.phase_deg[idx], d.mag_db[idx], "o", color="#27ae60")
            ax.annotate(
                f"PM = {d.pm_deg:.1f}°",
                (d.phase_deg[idx], d.mag_db[idx]),
                textcoords="offset points", xytext=(10, 5), color="#27ae60",
            )
        if d.phase_crossover_hz is not None:
            idx = np.argmin(np.abs(d.freq_hz - d.phase_crossover_hz))
            ax.plot(d.phase_deg[idx], d.mag_db[idx], "o", color="#c0392b")
            ax.annotate(
                f"GM = {d.gm_db:.1f} dB",
                (d.phase_deg[idx], d.mag_db[idx]),
                textcoords="offset points", xytext=(10, -12), color="#c0392b",
            )
        ax.set_xlabel("open-loop phase [deg]")
        ax.set_ylabel("open-loop gain [dB]")
        ax.set_title(
            f"Nichols — {AXIS_LABELS[d.axis]}   "
            f"(GM {_fmt(d.gm_db, 'dB')}, PM {_fmt(d.pm_deg, 'deg')})"
        )
        ax.grid(alpha=0.3)
        ax.set_ylim(-60, 40)
        ax.set_xlim(min(-360.0, np.min(d.phase_deg) - 10), max(0.0, np.max(d.phase_deg) + 10))
        paths.append(_save(fig, output_dir, f"nichols_{'xyz'[d.axis]}"))
    return paths


def plot_closed_loop(freq_data, output_dir: Path) -> list[Path]:
    """Closed-loop complementary sensitivity T and sensitivity S magnitudes."""
    paths = []
    for d in freq_data:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.semilogx(d.freq_hz, d.cl_mag_db, color="#2980b9", label="T(s) = L/(1+L)")
        ax.semilogx(d.freq_hz, d.sens_mag_db, color="#e67e22", label="S(s) = 1/(1+L)")
        ax.axhline(-3.0, color="0.5", ls="--", lw=0.8, label="−3 dB")
        if d.cl_bandwidth_hz is not None:
            ax.axvline(d.cl_bandwidth_hz, color="0.5", ls=":", lw=0.8)
            ax.annotate(
                f"BW = {d.cl_bandwidth_hz * 1e3:.1f} mHz",
                (d.cl_bandwidth_hz, -3.0),
                textcoords="offset points", xytext=(6, 6), fontsize=9,
            )
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("magnitude [dB]")
        ax.set_title(
            f"Closed-loop transfer functions — {AXIS_LABELS[d.axis]}   "
            f"(Mt = {d.cl_peak_db:.1f} dB, Ms = {d.sens_peak_db:.1f} dB)"
        )
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-60, 20)
        paths.append(_save(fig, output_dir, f"closed_loop_{'xyz'[d.axis]}"))
    return paths


def _fmt(v, unit: str) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isinf(v)) else f"{v:.1f} {unit}"
