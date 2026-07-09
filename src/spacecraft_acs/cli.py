"""Command-line interface: `acs step` and `acs freq`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import config as config_mod
from . import linearize, plotting, simulate


def _add_common(p: argparse.ArgumentParser):
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="path to the YAML configuration (default: config/default.yaml)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for generated plots (default: ./output)",
    )


def cmd_step(args) -> int:
    cfg = config_mod.load(args.config)
    print(f"Running {cfg.simulation.duration_s:.0f} s closed-loop simulation...")
    result = simulate.run(cfg)
    metrics = simulate.step_metrics(result)
    print(metrics)

    # Steady-state pointing after settling (last 10% of the run)
    tail = slice(int(0.9 * len(result.t)), None)
    rms = np.sqrt(np.mean(result.att_err_deg[tail] ** 2, axis=0)) * 3600.0
    print(
        "steady-state pointing error (last 10% of run, RMS): "
        f"roll {rms[0]:.1f}, pitch {rms[1]:.1f}, yaw {rms[2]:.1f} arcsec"
    )
    path = plotting.plot_step_response(result, metrics, args.output_dir)
    print(f"wrote {path}")
    return 0


def cmd_compare(args) -> int:
    """Profiled slew with acceleration feedforward vs raw step without."""
    import copy

    base = config_mod.load(args.config)

    cfg_prof = copy.deepcopy(base)
    cfg_prof.guidance.profiler.enabled = True
    cfg_prof.controller.feedforward = True

    cfg_step = copy.deepcopy(base)
    cfg_step.guidance.profiler.enabled = False
    cfg_step.controller.feedforward = False

    # Make sure the run covers the whole profile plus settling time
    from .guidance import Guidance

    slew_dur = Guidance(cfg_prof.guidance, cfg_prof.orbit_rate).slew_duration
    t_needed = cfg_prof.guidance.step.time_s + slew_dur + 400.0
    for cfg in (cfg_prof, cfg_step):
        cfg.simulation.duration_s = max(cfg.simulation.duration_s, t_needed)

    print(
        f"Comparing a {base.guidance.step.angle_deg:.1f} deg maneuver: "
        f"profiled slew ({slew_dur:.0f} s) + feedforward vs raw step...\n"
    )
    res_prof = simulate.run(cfg_prof)
    res_step = simulate.run(cfg_step)

    print(simulate.maneuver_metrics(res_prof, "profiled slew + feedforward"))
    print()
    print(simulate.maneuver_metrics(res_step, "raw step, no feedforward"))
    path = plotting.plot_slew_comparison(res_prof, res_step, args.output_dir)
    print(f"\nwrote {path}")
    return 0


def cmd_freq(args) -> int:
    cfg = config_mod.load(args.config)
    print("Linearizing about the nadir-pointing operating point...\n")
    data = linearize.analyze(cfg)
    print(linearize.report(data))
    paths = (
        plotting.plot_bode(data, args.output_dir)
        + plotting.plot_nichols(data, args.output_dir)
        + plotting.plot_closed_loop(data, args.output_dir)
    )
    print()
    for p in paths:
        print(f"wrote {p}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="acs",
        description="Attitude control design and analysis for a flexible GEO spacecraft",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_step = sub.add_parser("step", help="time-domain attitude step response")
    _add_common(p_step)
    p_step.set_defaults(func=cmd_step)

    p_freq = sub.add_parser(
        "freq", help="frequency-domain analysis (Bode, Nichols, closed loop)"
    )
    _add_common(p_freq)
    p_freq.set_defaults(func=cmd_freq)

    p_cmp = sub.add_parser(
        "compare",
        help="profiled slew with feedforward vs raw step without, same maneuver",
    )
    _add_common(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
