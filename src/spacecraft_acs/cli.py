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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
