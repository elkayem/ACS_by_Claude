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
    if cfg.estimator.enabled:
        est_rms = (
            np.sqrt(np.mean(np.rad2deg(result.est_att_err) ** 2, axis=0)) * 3600.0
        )
        print(
            "MEKF attitude estimation error (RMS): "
            f"roll {est_rms[0]:.2f}, pitch {est_rms[1]:.2f}, "
            f"yaw {est_rms[2]:.2f} arcsec"
        )
        path = plotting.plot_estimator(result, args.output_dir)
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


def cmd_unload(args) -> int:
    """Momentum unload demonstration."""
    import copy

    cfg = copy.deepcopy(config_mod.load(args.config))
    cfg.thrusters.enabled = True
    cfg.guidance.step.angle_deg = 0.0  # quiet nadir hold, unload only
    if np.all(
        np.abs(cfg.simulation.initial_wheel_momentum) < cfg.thrusters.unload.trigger
    ):
        h0 = 1.05 * cfg.thrusters.unload.trigger
        cfg.simulation.initial_wheel_momentum = np.array([h0, -h0, 0.6 * h0])
        print(
            f"initial wheel momentum below trigger; seeding [{h0:.1f}, {-h0:.1f}, "
            f"{0.6 * h0:.1f}] N*m*s to demonstrate an unload"
        )
    # Proportional unload has time constant 1/rate_gain; cover the full decay
    # from trigger to target plus margin
    u = cfg.thrusters.unload
    t_unload = np.log(u.trigger / u.target) / u.rate_gain
    cfg.simulation.duration_s = max(cfg.simulation.duration_s, 1.6 * t_unload)
    print(f"Running {cfg.simulation.duration_s:.0f} s nadir hold with unload...")
    result = simulate.run(cfg)

    h0 = np.abs(result.h_wheel[0])
    h_end = np.abs(result.h_wheel[-1])
    firing = np.any(result.torque_thruster != 0.0, axis=1)
    dt = result.t[1] - result.t[0]
    impulse = float(np.sum(np.abs(result.torque_thruster)) * dt)
    done = not bool(result.unloading[-1])
    duration = float(np.sum(result.unloading) * dt)
    err_peak = float(np.max(np.abs(result.att_err_deg))) * 3600.0
    print(f"wheel momentum |h|: [{h0[0]:.1f}, {h0[1]:.1f}, {h0[2]:.1f}] -> "
          f"[{h_end[0]:.2f}, {h_end[1]:.2f}, {h_end[2]:.2f}] N*m*s "
          f"({'complete' if done else 'still unloading'} after {duration:.0f} s)")
    print(f"thruster firing cycles: {int(np.sum(firing))}, "
          f"total impulse {impulse:.1f} N*m*s")
    print(f"peak pointing error during unload: {err_peak:.1f} arcsec")
    path = plotting.plot_unload(result, args.output_dir)
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


def cmd_burn(args) -> int:
    """Stationkeeping delta-V burn with phase-plane attitude control."""
    import copy

    from . import stationkeeping

    cfg = copy.deepcopy(config_mod.load(args.config))
    if cfg.stationkeeping.burn.delta_v <= 0.0:
        cfg.stationkeeping.burn.delta_v = 1.0  # m/s demo burn
        print("stationkeeping.burn.delta_v is 0; running a 1.0 m/s demo burn")
    cfg.guidance.step.angle_deg = 0.0  # nadir hold through the burn

    burn = stationkeeping.BurnController(
        cfg.rcs, cfg.stationkeeping, 1.0 / cfg.controller.rate_hz
    )
    f_net = np.linalg.norm(burn.forces.sum(axis=0))
    t_burn_est = cfg.stationkeeping.burn.delta_v * cfg.spacecraft.mass / f_net
    cfg.simulation.duration_s = max(
        cfg.simulation.duration_s,
        cfg.stationkeeping.burn.start_time_s + 1.6 * t_burn_est + 300.0,
    )
    print(
        f"Burn group '{cfg.stationkeeping.burn.group}' "
        f"({len(burn.units)} thrusters, {f_net:.1f} N net), target "
        f"{cfg.stationkeeping.burn.delta_v:.2f} m/s along "
        f"[{burn.burn_dir[0]:.2f} {burn.burn_dir[1]:.2f} {burn.burn_dir[2]:.2f}]..."
    )
    result = simulate.run(cfg)

    burn_mask = result.burning
    dt = result.t[1] - result.t[0]
    t_burn = float(np.sum(burn_mask) * dt)
    dv_final = result.delta_v[-1]
    impulse = float(np.sum(result.rcs_duty[burn_mask]) * dt) * burn.units[0].force
    prop = impulse / (cfg.rcs.isp_s * 9.80665)
    att_burn = result.att_err_deg[burn_mask]
    rate_burn = np.rad2deg(result.omega[burn_mask]) * 3600.0
    print(f"burn duration: {t_burn:.0f} s "
          f"(avg duty {np.mean(result.rcs_duty[burn_mask]):.2f})")
    print(f"delta-V achieved [body]: [{dv_final[0]:+.3f} {dv_final[1]:+.3f} "
          f"{dv_final[2]:+.3f}] m/s")
    print(f"total impulse {impulse:.0f} N*s, propellant {prop * 1000:.0f} g "
          f"(Isp {cfg.rcs.isp_s:.0f} s)")
    print(f"attitude during burn: max |err| = "
          f"[{np.max(np.abs(att_burn[:, 0])):.3f} {np.max(np.abs(att_burn[:, 1])):.3f} "
          f"{np.max(np.abs(att_burn[:, 2])):.3f}] deg "
          f"(deadband {cfg.stationkeeping.phase_plane.deadband_deg} deg)")
    print(f"post-burn recovery: max |err| after burn = "
          f"{np.max(np.abs(result.att_err_deg[~burn_mask & (result.t > cfg.stationkeeping.burn.start_time_s)])):.3f} deg")
    for p in plotting.plot_burn(result, args.output_dir):
        print(f"wrote {p}")
    return 0


def cmd_thrusters(args) -> int:
    from . import stationkeeping

    cfg = config_mod.load(args.config)
    print(stationkeeping.thruster_table(cfg.rcs))
    return 0


def cmd_mc(args) -> int:
    from . import montecarlo

    cfg = config_mod.load(args.config)
    if args.runs is not None:
        cfg.monte_carlo.n_runs = args.runs
    if args.time_domain:
        cfg.monte_carlo.time_domain = True
    print(
        f"Monte Carlo: {cfg.monte_carlo.n_runs} dispersed plants, fixed "
        f"controller (inertia +/-{cfg.monte_carlo.dispersions.inertia_pct:.0f}%, "
        f"mode freq +/-{cfg.monte_carlo.dispersions.mode_freq_pct:.0f}%, "
        f"damping {cfg.monte_carlo.dispersions.mode_damping_range}, "
        f"participation +/-{cfg.monte_carlo.dispersions.participation_pct:.0f}%)"
        + (", with time-domain runs" if cfg.monte_carlo.time_domain else "")
    )
    results = montecarlo.run(cfg, progress=print)
    print()
    print(montecarlo.report(results))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mc_results.csv"
    montecarlo.to_csv(results, csv_path)
    plot_path = plotting.plot_monte_carlo(results, args.output_dir)
    print(f"\nwrote {csv_path}\nwrote {plot_path}")
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

    p_unl = sub.add_parser(
        "unload", help="thruster momentum unload demonstration (nadir hold)"
    )
    _add_common(p_unl)
    p_unl.set_defaults(func=cmd_unload)

    p_burn = sub.add_parser(
        "burn", help="stationkeeping delta-V burn with phase-plane control"
    )
    _add_common(p_burn)
    p_burn.set_defaults(func=cmd_burn)

    p_thr = sub.add_parser("thrusters", help="print the RCS geometry table")
    _add_common(p_thr)
    p_thr.set_defaults(func=cmd_thrusters)

    p_mc = sub.add_parser(
        "mc", help="Monte Carlo plant-dispersion robustness analysis"
    )
    _add_common(p_mc)
    p_mc.add_argument("--runs", type=int, default=None, help="override n_runs")
    p_mc.add_argument(
        "--time-domain", action="store_true",
        help="also run the nonlinear sim per sample (slower)",
    )
    p_mc.set_defaults(func=cmd_mc)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
