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
acs step    --config config/default.yaml --output-dir output  # time domain
acs freq    --config config/default.yaml --output-dir output  # frequency domain
acs compare --config config/default.yaml --output-dir output  # slew vs step
acs unload  --config config/default.yaml --output-dir output  # momentum dump
acs mc      --config config/default.yaml --output-dir output  # Monte Carlo
```

`acs step` runs the nonlinear closed-loop simulation of a commanded attitude
step (default: 1° about roll while nadir tracking), prints rise time /
overshoot / settling time and steady-state pointing, and writes a time-history
plot (attitude error, rates, wheel torque and momentum vs. limits, modal
response). With `guidance.profiler.enabled: true` the same command is executed
as a smooth profiled slew instead.

`acs compare` runs the configured maneuver twice — once as a profiled slew
with acceleration feedforward, once as a raw quaternion step without — and
overlays attitude error, rate, torque, and modal response, with a metrics
table (settling time, overshoot, peak torque, flex ringing).

`acs unload` demonstrates a thruster momentum dump during nadir hold: wheel
momentum decay through the trigger/target thresholds, the quantized pulse
train, and the pointing transient (arcsec-level with wheel feedforward
compensation of each pulse).

`acs mc` runs the plant-dispersion Monte Carlo (`--runs N`, `--time-domain`):
the controller stays fixed while inertia, mode frequencies, damping, and
participation are dispersed; each sample is scored on loop-at-a-time margins,
worst flexible-mode peak, and coupled closed-loop stability, with a
scatter/histogram plot and CSV export.

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
- **Sensors** — star tracker attitude noise, gyro rate noise + bias with
  configurable bias random walk, sampled at the controller rate
  (`perfect: true` bypasses)
- **MEKF estimator** (`estimator.py`) — 6-state multiplicative EKF (attitude
  error + gyro bias): gyro propagation every controller cycle, star tracker
  updates decimated to their own rate (default 1 Hz), Joseph-form update.
  Default-on; delivers 1.4–3.8 arcsec attitude knowledge vs 10 arcsec raw ST
  and a quieter torque command
- **Momentum management** (`momentum.py`) — threshold-triggered thruster
  unload (the constant SRP pitch torque accumulates ~4.3 N·m·s/day):
  bang-bang-with-deadband law, minimum-impulse-bit pulse quantization via an
  impulse-debt accumulator, and wheel feedforward compensation of each pulse
- **Propellant slosh** (`slosh.py`) — each tank's first lateral mode as an
  equivalent spring-mass (Abramson SP-106 slosh-mass-fraction fit vs fill
  fraction), reduced by momentum elimination to two rotational modes per
  tank in the same hybrid-coordinate form as the structural modes:
  participation `(r×ê)·√(m_s/(1−m_s/M))`, frequency raised by the CM-shift
  factor. Slosh lands *in-band* (≈7–9 mHz vs 16–34 mHz crossovers) — it is
  phase-stabilized, not notched, and the analysis excludes in-band modes
  from the gain-stabilization check accordingly
- **Guidance** — nadir-pointing LVLH tracking at the GEO orbit rate (default)
  or inertial hold; attitude commands are quaternions and step offsets are
  applied about a configurable axis
- **Slew profiler** (`profiler.py`) — smooth eigenaxis reorientation with a
  cycloidal (versine) acceleration S-curve: attitude, rate, and acceleration
  are all continuous, respecting configurable `max_rate_dps` /
  `max_accel_dps2` limits with a constant-rate cruise for long slews.
  Guidance returns `(q_cmd, ω_cmd, α_cmd)`; with `controller.feedforward:
  true` the sim applies `J·α_cmd` feedforward torque (added after the
  structural filters), so the feedback loop only has to absorb tracking
  error, not the maneuver itself

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

Margins are computed **loop-at-a-time on the coupled 3-axis flexible
plant**: the full `[θ(3), ω(3), η, η̇]` state space (complete inertia tensor
and participation matrix) with one axis's loop broken for measurement while
the other two remain closed. A mode that couples into several axes is
therefore credited with the damping the closed loops provide, and closing
the measured loop yields the full coupled closed-loop poles — the stability
verdict matches the nonlinear sim's dynamics by construction (verified to 5%
in growth rate on a dispersed unstable sample). The controller TF includes
PID + filters + a 2nd-order Padé model of the `T/2 + delay_s` sampling
delay. All loop algebra stays in state space: polynomial transfer-function
arithmetic overflows float64 on strongly dispersed flexible plants and
produces spurious poles.

Remaining simplifications: orbit-rate gyroscopic coupling (~7e-5 rad/s) is
neglected, and the MEKF estimator dynamics are not in the linear model (its
attitude corrections are low-rate; the gyro path is direct). The
`slow`-marked test in `tests/test_simulate.py` verifies the linear gain
margin against nonlinear divergence.

## Default configuration

`config/default.yaml` models a large GEO comsat: 8000/4500/6500 kg·m²
inertia, four array modes at 0.10–0.55 Hz with ζ = 0.005 (25%/14%/19% modal
inertia fraction in roll/pitch/yaw), 0.2 N·m / 68 N·m·s wheels, 4 Hz
sampling.

The control design is **robustness-first**, selected by Monte Carlo pass
rate rather than nominal margins alone. Structure: per-axis design
bandwidths (10/22/13 mHz roll/pitch/yaw, each scaled to its axis's first
flexible mode), wide notches (damping 1.0) centered on each coupled
resonance so they cover ±15% modal frequency motion plus
participation-driven coupling shifts, and a per-axis second-order roll-off
at ~4.5× each axis's crossover with light damping (0.4) — same asymptotic
attenuation at the dispersed mode positions for about half the crossover
phase cost of a Butterworth section. Every filter is assigned only to the
axis that needs it.

Results with slosh included: nominal GM 8.3–9.9 dB, PM 35–39°, closed-loop
BW 35/95/43 mHz — and under the full dispersion set (inertia ±10%, mode
frequency ±15%, damping 0.005–0.01, participation ±20%, slosh frequency
±50%, slosh mass ±30%, slosh damping 0.004–0.02, 100 samples) **100% of
samples meet GM ≥ 6 dB, PM ≥ 30°, worst mode ≤ −6 dB, with zero unstable
cases** (worst-case sample: GM 7.8 dB, PM 33.5°, mode −8.1 dB).

**Derived tank requirement — slosh damping ≥ 0.004 (PMD-class).** With
bare-tank damping (ζ floor 0.001) the pass rate falls to ~65% (min PM 15°):
the lightly damped slosh ripple crosses unity near the roll/yaw crossovers
and erodes the margins, and raising bandwidth to clear it is impossible
because the wide array-mode notches forbid a higher crossover — the
slosh-to-array corridor closes. Crucially, tighter slosh *frequency*
knowledge (±25%) does not help at all; damping is the binding parameter.
Any PMD-class floor closes the design (ζ 0.004–0.02 → 100%, 0.01–0.05 →
100%, diaphragm 0.03–0.10 → 100%): slosh compliance is bought with tank
hardware, not control gains. Higher controller sample rate does not help
either — the ZOH delay costs only ~0.7° of PM at crossover vs ~33° for the
robustness filters (4 → 50 Hz buys 0.3°).

Operational note: slosh rings after maneuvers. A 1° maneuver (profiled or
raw) leaves ~1 cm of propellant CM motion ringing at ~7 mHz with a ~45 min
decay, visible as ~30–100 arcsec of attitude oscillation; the default slew
profile (~89 s) is spectrally close to the slosh band. Slosh-gentle slews
must be several slosh periods long (lower `max_accel_dps2`), a standard
agility-vs-quiescence trade.

Design history worth knowing: an earlier high-bandwidth variant
(30/55/38 mHz, narrow notches) had spectacular nominal margins
(GM 15–20 dB) but a 1% Monte Carlo pass rate with ~14% genuinely unstable
samples — one was confirmed diverging in the nonlinear sim with a 940 s
doubling time, exactly as the coupled linear model predicted. Fixed notches
cannot chase ±15% modes at very low damping; robustness had to be bought
with bandwidth. Rerun `acs mc` with your program's actual uncertainty set
(a test-correlated FEM customarily justifies ±5%) before trading bandwidth
back up.

Design note: with these gains the wheels saturate for attitude errors above
a few hundredths of a degree, so any sizable raw step becomes a
torque-limited slew with saturation-driven overshoot. Use the slew profiler
with feedforward for large-angle maneuvers (`acs compare` quantifies the
difference).

## Layout

```
config/default.yaml        # all tunable parameters, commented
src/spacecraft_acs/
  quaternion.py            # scalar-first quaternion algebra
  config.py                # schema + validation
  dynamics.py              # flexible-body EOM (RK4)
  actuators.py sensors.py environment.py guidance.py
  profiler.py              # smooth eigenaxis slew profile
  slosh.py                 # tank slosh -> equivalent rotational modes
  estimator.py             # 6-state MEKF (attitude + gyro bias)
  momentum.py              # thruster momentum unload manager
  controller.py            # discrete quaternion PID + filters + feedforward
  simulate.py              # closed-loop sim + step/maneuver metrics
  linearize.py             # coupled loop-at-a-time LTI margins, closed loop
  montecarlo.py            # plant-dispersion robustness analysis
  plotting.py cli.py
tests/                     # physics + control verification suite
```
