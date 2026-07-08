# spacecraft-acs

Attitude control system design and analysis for a large GEO satellite with
large flexible solar arrays. Supports the classical GNC workflow:

1. **Flexible-body dynamics** — rigid hub + appendage modes, hybrid-coordinate
   formulation
2. **Control design** — discrete quaternion-error PID with structural filters
3. **Analysis** — nonlinear time-domain step response, and linearized
   frequency-domain margins (Bode / Nichols / closed-loop transfer functions)

All spacecraft parameters, controller gains, filters, sensor/actuator models,
and simulation settings are configurable through a single YAML file.

## Installation

```sh
pip install -e .[dev]
pytest            # run the verification suite
```

## Usage

```sh
acs step --config config/default.yaml --output-dir output   # time domain
acs freq --config config/default.yaml --output-dir output   # frequency domain
```

`acs step` runs the nonlinear closed-loop simulation of a commanded attitude
step (default: 0.5° about roll while nadir tracking), prints rise time /
overshoot / settling time and steady-state pointing, and writes a time-history
plot (attitude error, rates, wheel torque and momentum vs. limits, modal
response).

`acs freq` linearizes each axis about the nadir-pointing operating point and
writes open-loop Bode and Nichols plots with the gain and phase margins called
out, plus closed-loop `T(s) = L/(1+L)` and `S(s) = 1/(1+L)` magnitude plots.
The console report lists GM/PM with crossover frequencies, closed-loop
bandwidth and peaks (Mt, Ms), and the peak open-loop gain at each flexible
resonance (gain-stabilization check).

## Model

### Flexible dynamics (`dynamics.py`)

Hybrid-coordinate equations for a rigid hub with N mass-normalized flexible
modes coupled through the 3×N rotational participation matrix **L**:

```
J ω̇ + L η̈ + ω × (J ω + L η̇ + h_w) = T_wheel + T_dist
η̈ + 2 Z Ω η̇ + Ω² η + Lᵀ ω̇ = 0
q̇ = ½ q ⊗ [0, ω]           (scalar-first quaternion, inertial → body)
ḣ_w = −T_wheel
```

Each mode is configured by its cantilever frequency, damping ratio, and
participation 3-vector; `|l|²` is the mode's contribution to the effective
inertia about that axis. Config validation enforces a positive-definite
hybrid mass matrix (`J − L Lᵀ ≻ 0`). The unforced model conserves energy and
angular momentum exactly (see `tests/test_dynamics.py`).

Note the pole/zero structure this produces per axis: the collocated
torque→attitude transfer function has zeros at the cantilever frequency and
poles at the coupled free-free frequency `ω√(J/(J−l²))` above it.

### Supporting models

- **Reaction wheels** — per-axis torque and momentum saturation
  (`ideal: true` bypasses)
- **Environment** — gravity-gradient torque `3n²(ô₃ × J ô₃)` and SRP
  (constant + orbit-rate harmonic body torque)
- **Sensors** — star tracker attitude noise, gyro rate noise + bias, sampled
  at the controller rate (`perfect: true` bypasses; no estimator in v1 — a
  MEKF is a natural extension)
- **Guidance** — nadir-pointing LVLH tracking at the GEO orbit rate (default)
  or inertial hold; attitude commands are quaternions and step offsets are
  applied about a configurable axis

### Controller (`controller.py`)

Quaternion-error feedback PID executed at a configurable discrete rate
(default 4 Hz) with zero-order hold:

- error rotation vector `θ = 2·vec(q_cmd⁻¹ ⊗ q_meas)` (shortest path)
- per-axis `u = −(Kp θ + Ki ∫θ dt + Kd ω_err)` with integrator anti-windup
  (frozen while the wheels saturate)
- cascaded second-order structural filters (lowpass roll-off and notches)
- gains come from a bandwidth/damping design rule (`Kp = Jωn²`,
  `Kd = 2ζJωn`, `Ki = Kp·ωn/factor`) unless explicit `kp/ki/kd` are given

### Frequency-domain analysis (`linearize.py`)

Per-axis SISO linearization about nadir: plant state space `[θ, ω, η, η̇]`
from the axis inertia and participation row, controller TF from the PID +
filters, and the sampling/computation delay modeled as a 2nd-order Padé
approximation of `T/2 + delay_s`. Margins are computed on
`L(s) = C(s)·G(s)`.

Simplifications (standard for preliminary design, acceptable for
near-diagonal inertia): axes are analyzed as decoupled SISO loops;
orbit-rate gyroscopic coupling (~7e-5 rad/s) and cross-axis flexible coupling
are neglected in the linear model. The nonlinear simulation retains all of
these effects — the `slow`-marked test in `tests/test_simulate.py` verifies
the linear gain margin against nonlinear divergence.

## Default configuration

`config/default.yaml` models a large GEO comsat: 8000/4500/6500 kg·m²
inertia, four array modes at 0.10–0.55 Hz with ζ = 0.005 (25%/14%/19% modal
inertia fraction in roll/pitch/yaw), 0.2 N·m / 68 N·m·s wheels, 0.02 Hz
design bandwidth at 4 Hz sampling. The filter set (three notches at the
*coupled* mode frequencies + 0.6 Hz roll-off) gain-stabilizes every mode by
≥ 14 dB and yields GM ≈ 10–11 dB, PM ≈ 31°, closed-loop BW ≈ 52 mHz, and
1–2 arcsec RMS steady-state pointing per axis.

Design note: with these gains the wheels saturate for attitude errors above
~0.1°, so large step commands become torque-limited slews with
saturation-driven overshoot (~76% for a 10° step vs ~23% for 0.5°). That is
physics, not a bug; feedforward slew profiling would be the v2 fix.

## Layout

```
config/default.yaml        # all tunable parameters, commented
src/spacecraft_acs/
  quaternion.py            # scalar-first quaternion algebra
  config.py                # schema + validation
  dynamics.py              # flexible-body EOM (RK4)
  actuators.py sensors.py environment.py guidance.py
  controller.py            # discrete quaternion PID + filters
  simulate.py              # closed-loop sim + step metrics
  linearize.py             # per-axis LTI models, margins, closed loop
  plotting.py cli.py
tests/                     # physics + control verification suite
```
