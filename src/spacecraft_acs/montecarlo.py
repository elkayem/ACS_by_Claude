"""Monte Carlo plant-dispersion analysis.

The controller (gains, filters, sample rate) stays fixed at its as-designed
values while the plant is dispersed: inertia, modal frequencies, modal
damping, and participation. Each sample is scored with the linear
frequency-domain metrics (worst-axis GM/PM, worst flexible-mode peak,
closed-loop stability), optionally plus a nonlinear time-domain run of the
configured maneuver. This directly tests the design's robustness claims —
e.g. that the notch widths cover the assumed modal frequency uncertainty.

Dispersions are uniform. Inertia is dispersed as J' = S J S with
S = diag(sqrt(f_i)), which preserves symmetry and positive definiteness while
scaling the diagonal entries by f_i. Samples that violate the physical
validity check (hybrid mass matrix positive definite) are redrawn.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import linearize, simulate
from .config import Config, DispersionConfig

# Standard design requirements the fleet of samples is scored against
REQ_GM_DB = 6.0
REQ_PM_DEG = 30.0
REQ_MODE_DB = -6.0


@dataclass
class McSample:
    index: int
    gm_db: float  # worst axis
    pm_deg: float  # worst axis
    mode_peak_db: float  # worst mode, worst axis
    stable: bool
    mode_ok: bool = True  # every mode gain-stabilized OR outside the
    # 6 dB / 30 deg Nichols exclusion zone (phase-stabilized)
    settling_time_s: float | None = None  # time domain only
    overshoot_deg: float | None = None
    peak_torque_nm: float | None = None

    @property
    def passes(self) -> bool:
        return (
            self.stable
            and self.gm_db >= REQ_GM_DB
            and self.pm_deg >= REQ_PM_DEG
            and self.mode_ok
        )


@dataclass
class McResults:
    nominal: McSample
    samples: list[McSample] = field(default_factory=list)

    def field_array(self, name: str) -> np.ndarray:
        return np.array([getattr(s, name) for s in self.samples], dtype=float)

    @property
    def pass_rate(self) -> float:
        return float(np.mean([s.passes for s in self.samples]))


def disperse(config: Config, disp: DispersionConfig, rng: np.random.Generator) -> Config:
    """One dispersed plant sample (controller untouched). Redraws on
    physically invalid combinations."""
    for _ in range(100):
        cfg = copy.deepcopy(config)
        sc = cfg.spacecraft
        f = 1.0 + (disp.inertia_pct / 100.0) * rng.uniform(-1.0, 1.0, 3)
        s = np.diag(np.sqrt(f))
        inertia = s @ sc.inertia @ s
        modes = []
        for m in copy.deepcopy(sc.modes):
            m.freq_hz *= 1.0 + (disp.mode_freq_pct / 100.0) * rng.uniform(-1.0, 1.0)
            m.damping = rng.uniform(*disp.mode_damping_range)
            m.participation = m.participation * (
                1.0 + (disp.participation_pct / 100.0) * rng.uniform(-1.0, 1.0, 3)
            )
            modes.append(m)
        tanks = []
        for t in copy.deepcopy(sc.tanks):
            t.freq_hz *= 1.0 + (disp.slosh_freq_pct / 100.0) * rng.uniform(-1.0, 1.0)
            t.propellant_mass *= 1.0 + (disp.slosh_mass_pct / 100.0) * rng.uniform(-1.0, 1.0)
            if t.slosh_mass is not None:
                t.slosh_mass *= 1.0 + (disp.slosh_mass_pct / 100.0) * rng.uniform(-1.0, 1.0)
            lo, hi = disp.slosh_damping_range
            t.damping = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            tanks.append(t)
        try:
            cfg.spacecraft = type(sc)(
                inertia=inertia, modes=modes, mass=sc.mass, tanks=tanks
            )
            return cfg
        except ValueError:
            continue  # invalid draw (e.g. participation too large); redraw
    raise RuntimeError("could not draw a physically valid dispersed sample")


def evaluate(cfg: Config, index: int, time_domain: bool, n_points: int = 1500) -> McSample:
    data = [linearize.analyze_axis(cfg, axis, n_points=n_points) for axis in range(3)]
    gm = min((d.gm_db for d in data if d.gm_db is not None), default=np.inf)
    pm = min((d.pm_deg for d in data if d.pm_deg is not None), default=np.inf)
    mode_peaks = [g for d in data for _, g, _ in d.mode_gain_db]
    mode_peak = max(mode_peaks) if mode_peaks else -np.inf
    mode_ok = not any(in_box for d in data for _, _, in_box in d.mode_gain_db)
    stable = not any(p.real > 1e-9 for d in data for p in d.cl_poles)
    sample = McSample(
        index=index, gm_db=gm, pm_deg=pm, mode_peak_db=mode_peak,
        stable=stable, mode_ok=mode_ok,
    )

    if time_domain and stable:
        result = simulate.run(cfg)
        if not np.all(np.isfinite(result.att_err_deg)):
            sample.stable = False
        else:
            metrics = simulate.maneuver_metrics(result, f"sample {index}")
            sample.settling_time_s = metrics.settling_time_s
            sample.overshoot_deg = metrics.overshoot_deg
            sample.peak_torque_nm = metrics.peak_torque_nm
    return sample


def run(config: Config, progress=None) -> McResults:
    mc = config.monte_carlo
    rng = np.random.default_rng(mc.seed)
    results = McResults(nominal=evaluate(config, -1, mc.time_domain))
    for i in range(mc.n_runs):
        cfg = disperse(config, mc.dispersions, rng)
        results.samples.append(evaluate(cfg, i, mc.time_domain))
        if progress and (i + 1) % 10 == 0:
            progress(f"  {i + 1}/{mc.n_runs} samples")
    return results


def report(results: McResults) -> str:
    s = results.samples
    gm = results.field_array("gm_db")
    pm = results.field_array("pm_deg")
    mode = results.field_array("mode_peak_db")
    n_unstable = sum(not x.stable for x in s)
    lines = [
        f"Monte Carlo: {len(s)} dispersed samples "
        f"(nominal: GM {results.nominal.gm_db:.1f} dB, "
        f"PM {results.nominal.pm_deg:.1f} deg, "
        f"mode {results.nominal.mode_peak_db:.1f} dB)",
        f"  closed-loop unstable samples: {n_unstable}",
        f"  worst-axis GM  [dB]:  min {np.min(gm):6.1f}   "
        f"median {np.median(gm):6.1f}   (requirement >= {REQ_GM_DB})",
        f"  worst-axis PM  [deg]: min {np.min(pm):6.1f}   "
        f"median {np.median(pm):6.1f}   (requirement >= {REQ_PM_DEG})",
        f"  worst mode |L| [dB]:  max {np.max(mode):6.1f}   "
        f"median {np.median(mode):6.1f}   (gain-stab. target <= {REQ_MODE_DB}; "
        f"louder modes must clear the 6 dB/30 deg zone)",
        f"  mode exclusion-zone violations: "
        f"{sum(not x.mode_ok for x in s)} of {len(s)} samples",
        f"  pass rate (all requirements): {100.0 * results.pass_rate:.1f} %",
    ]
    if s and s[0].settling_time_s is not None:
        settle = np.array(
            [x.settling_time_s for x in s if x.settling_time_s is not None]
        )
        lines.append(
            f"  time domain: settling min/median/max = "
            f"{np.min(settle):.0f}/{np.median(settle):.0f}/{np.max(settle):.0f} s "
            f"({len(settle)}/{len(s)} settled)"
        )
    return "\n".join(lines)


def to_csv(results: McResults, path) -> None:
    fields = [
        "index", "gm_db", "pm_deg", "mode_peak_db", "stable", "mode_ok",
        "settling_time_s", "overshoot_deg", "peak_torque_nm",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for smp in [results.nominal] + results.samples:
            f.write(
                ",".join(
                    "" if getattr(smp, name) is None else str(getattr(smp, name))
                    for name in fields
                )
                + "\n"
            )
