# spacecraft-acs

![Large GEO communications satellite with north-south solar array wings](docs/spacecraft.svg)

*The modeled spacecraft: 3000 kg wet mass with two ~120 kg solar array wings
spanning ~22 m tip-to-tip along the pitch (north-south) axis â€” the geometry
behind J = diag(15000, 3000, 14500) kgÂ·mÂ², the 0.10â€“0.55 Hz array modes that
rotate between roll and yaw as the wings track the sun, and the two
propellant tanks whose slosh modes sit below the control bandwidth.*

Attitude control system design and analysis for a large GEO satellite with
large flexible solar arrays. Supports the classical GNC workflow:

1. **Flexible-body dynamics** â€” rigid hub + appendage modes, hybrid-coordinate
   formulation
2. **Control design** â€” discrete quaternion-error PID with structural filters
3. **Analysis** â€” nonlinear time-domain step response, and linearized
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
acs step      --config config/default.yaml --output-dir output  # time domain
acs freq      --config config/default.yaml --output-dir output  # frequency domain
acs compare   --config config/default.yaml --output-dir output  # slew vs step
acs unload    --config config/default.yaml --output-dir output  # momentum dump
acs mc        --config config/default.yaml --output-dir output  # Monte Carlo
acs burn      --config config/default.yaml --output-dir output  # delta-V burn
acs hold      --config config/default.yaml --output-dir output  # thruster hold
acs holdmc    --runs 30                    --output-dir output  # hold dispersions
acs thrusters --config config/default.yaml                      # RCS geometry table
```

`acs step` runs the nonlinear closed-loop simulation of a commanded attitude
step (default: 1Â° about roll while nadir tracking), prints rise time /
overshoot / settling time and steady-state pointing, and writes a time-history
plot (attitude error, rates, wheel torque and momentum vs. limits, modal
response). With `guidance.profiler.enabled: true` the same command is executed
as a smooth profiled slew instead.

`acs compare` runs the configured maneuver twice â€” once as a profiled slew
with acceleration feedforward, once as a raw quaternion step without â€” and
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
modes coupled through the 3Ã—N rotational participation matrix **L**:

```
J Ï‰Ì‡ + L Î·Ìˆ + Ï‰ Ã— (J Ï‰ + L Î·Ì‡ + h_w) = T_wheel + T_dist
Î·Ìˆ + 2 Z Î© Î·Ì‡ + Î©Â² Î· + Láµ€ Ï‰Ì‡ = 0
qÌ‡ = Â½ q âŠ— [0, Ï‰]           (scalar-first quaternion, inertial â†’ body)
á¸£_w = âˆ’T_wheel
```

Each mode is configured by its cantilever frequency, damping ratio, and
participation 3-vector; `|l|Â²` is the mode's contribution to the effective
inertia about that axis. Config validation enforces a positive-definite
hybrid mass matrix (`J âˆ’ L Láµ€ â‰» 0`). The unforced model conserves energy and
angular momentum exactly (see `tests/test_dynamics.py`).

Note the pole/zero structure this produces per axis: the collocated
torqueâ†’attitude transfer function has zeros at the cantilever frequency and
poles at the coupled free-free frequency `Ï‰âˆš(J/(Jâˆ’lÂ²))` above it.

### Supporting models

- **Reaction wheels** â€” per-axis torque and momentum saturation
  (`ideal: true` bypasses)
- **Environment** â€” gravity-gradient torque `3nÂ²(Ã´â‚ƒ Ã— J Ã´â‚ƒ)` and SRP
  (constant + orbit-rate harmonic body torque)
- **Sensors** â€” star tracker attitude noise, gyro rate noise + bias with
  configurable bias random walk, sampled at the controller rate
  (`perfect: true` bypasses)
- **MEKF estimator** (`estimator.py`) â€” 6-state multiplicative EKF (attitude
  error + gyro bias): gyro propagation every controller cycle, star tracker
  updates decimated to their own rate (default 8 Hz), Joseph-form update.
  Default-on; delivers few-arcsec attitude knowledge vs 10 arcsec raw ST
  and a quieter torque command
- **Momentum management** (`momentum.py`) â€” threshold-triggered thruster
  unload (the constant SRP pitch torque accumulates ~4.3 NÂ·mÂ·s/day):
  bang-bang-with-deadband law, minimum-impulse-bit pulse quantization via an
  impulse-debt accumulator, and wheel feedforward compensation of each pulse
- **Propellant slosh** (`slosh.py`) â€” each tank's first lateral mode as an
  equivalent spring-mass (Abramson SP-106 slosh-mass-fraction fit vs fill
  fraction), reduced by momentum elimination to two rotational modes per
  tank in the same hybrid-coordinate form as the structural modes:
  participation `(rÃ—Ãª)Â·âˆš(m_s/(1âˆ’m_s/M))`, frequency raised by the CM-shift
  factor. Slosh lands *in-band* (â‰ˆ7â€“9 mHz vs 16â€“34 mHz crossovers) â€” it is
  phase-stabilized, not notched, and the analysis excludes in-band modes
  from the gain-stabilization check accordingly
- **Guidance** â€” nadir-pointing LVLH tracking at the GEO orbit rate (default)
  or inertial hold; attitude commands are quaternions and step offsets are
  applied about a configurable axis
- **Array rotation** â€” the solar arrays rotate about pitch once per day;
  modes flagged `rotates_with_array` have their participation defined at
  drive angle 0 and rotated with `spacecraft.array_angle_deg`, so the
  out-of-plane bending mode couples to roll at 0Â° and to yaw at 90Â° (and
  the coupled resonance slides with the angle-dependent participation).
  Torsion and slosh are body-fixed. Physical validity (positive-definite
  hybrid mass matrix) is enforced at every angle, and the Monte Carlo
  samples the drive angle uniformly over the revolution
- **Slew profiler** (`profiler.py`) â€” smooth eigenaxis reorientation with a
  cycloidal (versine) acceleration S-curve: attitude, rate, and acceleration
  are all continuous, respecting configurable `max_rate_dps` /
  `max_accel_dps2` limits with a constant-rate cruise for long slews.
  Guidance returns `(q_cmd, Ï‰_cmd, Î±_cmd)`; with `controller.feedforward:
  true` the sim applies `JÂ·Î±_cmd` feedforward torque (added after the
  structural filters), so the feedback loop only has to absorb tracking
  error, not the maneuver itself

### Controller (`controller.py`)

Quaternion-error feedback PID executed at a configurable discrete rate
(default 16 Hz, equal to the gyro sampling rate) with zero-order hold:

- error rotation vector `Î¸ = 2Â·vec(q_cmdâ»Â¹ âŠ— q_meas)` (shortest path)
- per-axis `u = âˆ’(Kp Î¸ + Ki âˆ«Î¸ dt + Kd Ï‰_err)` with integrator anti-windup
  (frozen while the wheels saturate)
- cascaded second-order structural filters (lowpass roll-off and notches)
- gains come from a bandwidth/damping design rule (`Kp = JÏ‰nÂ²`,
  `Kd = 2Î¶JÏ‰n`, `Ki = KpÂ·Ï‰n/factor`) unless explicit `kp/ki/kd` are given

### Frequency-domain analysis (`linearize.py`)

![Open-loop Bode overlay, all axes](output/bode_overlay.png)

Margins are computed **loop-at-a-time on the coupled 3-axis flexible
plant**: the full `[Î¸(3), Ï‰(3), Î·, Î·Ì‡]` state space (complete inertia tensor
and participation matrix) with one axis's loop broken for measurement while
the other two remain closed. A mode that couples into several axes is
therefore credited with the damping the closed loops provide, and closing
the measured loop yields the full coupled closed-loop poles â€” the stability
verdict matches the nonlinear sim's dynamics by construction (verified to 5%
in growth rate on a dispersed unstable sample). The controller TF includes
PID + filters + a 2nd-order PadÃ© model of the `T/2 + delay_s` sampling
delay. All loop algebra stays in state space: polynomial transfer-function
arithmetic overflows float64 on strongly dispersed flexible plants and
produces spurious poles.

Remaining simplifications: orbit-rate gyroscopic coupling (~7e-5 rad/s) is
neglected, and the MEKF estimator dynamics are not in the linear model (its
attitude corrections are low-rate; the gyro path is direct). The
`slow`-marked test in `tests/test_simulate.py` verifies the linear gain
margin against nonlinear divergence.

## Stationkeeping thruster mode

`acs burn` runs a stationkeeping delta-V burn under thruster attitude
control: the reaction wheels are held (thruster torques would saturate them
in seconds) and a classical per-axis **phase plane** â€” switching function
`s = Î¸ + T_leadÂ·Ï‰` against a deadband with hysteresis, plus a hard rate
limit â€” commands off-pulsing of the burn thrusters. A configurable CM
offset produces the realistic constant disturbance torque that drives the
burn limit cycle.

![Phase-plane logic](docs/phase_plane.svg)

The default deadband is Â±0.5Â° with a 30 s rate lead â€”
tuned by the attitude-hold sweep (tighter deadbands with long leads chatter
on slosh rate content; this point holds a near-ballistic limit cycle at
~1â€“3 g/hr). Demo (1 m/s north, four 10 N thrusters at 94% geometric
efficiency): 86 s burn at 0.93 average duty, attitude riding at ~0.29Â°
inside the deadband, 19 mm/s cross-axis delta-V, 1.12 kg propellant at
Isp 290 s. `acs hold` runs the zero-delta-V phase-plane attitude hold
(pure-torque couples, 1Â° initial error) and `acs holdmc` its time-domain
dispersion campaign.

![RCS thruster layout](docs/thrusters.svg)

Geometry (also printed by `acs thrusters`; torque about the nominal CM â€”
burn dynamics use the actual, offset CM). The four thrusters on each
north/south face are canted 20Â° in Â±Z with opposite senses top/bottom, so
the plumes clear the array wing on that side while the group nets to pure
force: differential off-pulsing then provides Â±18.5 / Â±6.2 / Â±16.9 NÂ·m of
roll/pitch/yaw authority during the burn.

| thruster | position [m] | thrust direction | force [N] | torque about CM [NÂ·m] |
|---|---|---|---|---|
| N1 | ( 0.90,  1.15,  1.40) | (0, âˆ’0.940, âˆ’0.342) | 10 | ( +9.2, +3.1, âˆ’8.5) |
| N2 | (âˆ’0.90,  1.15,  1.40) | (0, âˆ’0.940, âˆ’0.342) | 10 | ( +9.2, âˆ’3.1, +8.5) |
| N3 | ( 0.90,  1.15, âˆ’1.40) | (0, âˆ’0.940, +0.342) | 10 | ( âˆ’9.2, âˆ’3.1, âˆ’8.5) |
| N4 | (âˆ’0.90,  1.15, âˆ’1.40) | (0, âˆ’0.940, +0.342) | 10 | ( âˆ’9.2, +3.1, +8.5) |
| S1 | ( 0.90, âˆ’1.15,  1.40) | (0, +0.940, âˆ’0.342) | 10 | ( âˆ’9.2, +3.1, +8.5) |
| S2 | (âˆ’0.90, âˆ’1.15,  1.40) | (0, +0.940, âˆ’0.342) | 10 | ( âˆ’9.2, âˆ’3.1, âˆ’8.5) |
| S3 | ( 0.90, âˆ’1.15, âˆ’1.40) | (0, +0.940, +0.342) | 10 | ( +9.2, âˆ’3.1, +8.5) |
| S4 | (âˆ’0.90, âˆ’1.15, âˆ’1.40) | (0, +0.940, +0.342) | 10 | ( +9.2, +3.1, âˆ’8.5) |
| E1 | ( 1.25,  0.00,  1.40) | (âˆ’1, 0, 0) | 10 | ( 0, âˆ’14.0, 0) |
| E2 | ( 1.25,  0.00, âˆ’1.40) | (âˆ’1, 0, 0) | 10 | ( 0, +14.0, 0) |
| W1 | (âˆ’1.25,  0.00,  1.40) | (+1, 0, 0) | 10 | ( 0, +14.0, 0) |
| W2 | (âˆ’1.25,  0.00, âˆ’1.40) | (+1, 0, 0) | 10 | ( 0, âˆ’14.0, 0) |

Burn groups: **north** Î”V = S1â€“S4 (net +37.6 N Å·), **south** = N1â€“N4,
**east** = W1â€“W2 (+20 N xÌ‚), **west** = E1â€“E2 â€” every group has exactly zero
net torque about the nominal CM. Scope notes: delta-V integrates in body
axes (â‰ˆ orbital frame while attitude is held; no orbit propagation), and
slosh sees only the rotational coupling during burns â€” translational slosh
forcing under thrust is not modeled.

## Default configuration

`config/default.yaml` models a large GEO comsat: 3000 kg wet mass with
15000/3000/14500 kgÂ·mÂ² inertia (north-south wings put roll/yaw at ~5Ã—
pitch), four array modes at 0.10â€“0.55 Hz with Î¶ = 0.005 (20%/21%/11% modal
inertia fraction on their primary axes at drive angle 0), two propellant
tanks (700/430 kg at 50% fill), 0.2 NÂ·m / 68 NÂ·mÂ·s wheels, twelve 10 N RCS
thrusters at Isp 290 s, and 16 Hz gyro/controller sampling with 8 Hz star
tracker updates. The frequency
analysis is a continuous-equivalent model of the discrete loop and is
plotted only up to the 5Ã—-highest-mode coverage, never past Nyquist.

The control design is **robustness-first across the full daily array
revolution**, selected by Monte Carlo pass rate rather than nominal margins
alone. Because the arrays rotate about pitch once per day, the two bending
modes exchange between roll and yaw, so those axes are designed
symmetrically with the full three-notch complement on each; pitch carries
only the angle-invariant torsion mode. Î¶_cl = 0.75 with a long integral
time (Ti factor 15) balances the GM/PM budget the notches squeeze.

There are **two design points**, distinguished by how well the array modes
are known:

- **Pre-ID (Â±15% mode frequency)**: wide notches (damping 1.0) cap roll/yaw
  at 7.5 mHz. Robust with no calibration beyond a test-correlated FEM â€” but
  crossover then sits *inside* the dispersed slosh band, and |S| at 10 mHz
  is +0.9 dB: the loop mildly amplifies slosh-band disturbances.
- **Post-ID (Â±5%, the default)**: with the array bending modes identified
  on orbit to Â±5% â€” a **derived operations requirement** â€” the notches
  narrow to damping 0.40, halving their crossover phase lag, and roll/yaw
  bandwidth rises to **20 mHz** (pitch 30 mHz). Crossover now sits ~2Ã—
  above the dispersed slosh band and |S| at 10 mHz improves 14 dB to
  **âˆ’13.4 dB**: the loop actively damps slosh instead of coexisting with
  it. Margins are simultaneously *better* (100-sample MC worst case:
  GM 7.0 dB, PM 35.1Â°, zero unstable, zero mode violations, array angle
  dispersed over the full revolution).

Both points pass the Monte Carlo at 100% against GM â‰¥ 6 dB, PM â‰¥ 30Â°, and
the gain-stabilized-or-disk-margin mode criterion; the pre-ID gains remain
documented here as the safe initial-operations configuration until the
modal survey is complete.

**Derived tank requirement â€” slosh damping â‰¥ 0.004 (PMD-class).** With
bare-tank damping (Î¶ floor 0.001) the pass rate falls to ~65% (min PM 15Â°):
the lightly damped slosh ripple crosses unity near the roll/yaw crossovers
and erodes the margins, and raising bandwidth to clear it is impossible
because the wide array-mode notches forbid a higher crossover â€” the
slosh-to-array corridor closes. Crucially, tighter slosh *frequency*
knowledge (Â±25%) does not help at all; damping is the binding parameter.
Any PMD-class floor closes the design (Î¶ 0.004â€“0.02 â†’ 100%, 0.01â€“0.05 â†’
100%, diaphragm 0.03â€“0.10 â†’ 100%): slosh compliance is bought with tank
hardware, not control gains. Higher controller sample rate does not help
either â€” the ZOH delay costs only ~0.7Â° of PM at crossover vs ~33Â° for the
robustness filters (a rate study from 4 to 50 Hz moved PM by only 0.3Â°).

**Slosh ringing and its mitigation.** A 1Â° maneuver leaves ~1 cm of
propellant CM motion ringing at ~7 mHz, visible as tens of arcsec of
attitude oscillation; with PMD-class damping the decay constant is ~47 min.
Quantified mitigations (1Â° profiled slew, `acs compare`-style metrics):

| mitigation (1Â° slew) | post-slew ringing | decay |
|---|---|---|
| none: fast slew (89 s), PMD Î¶â‰ˆ0.008 | 35 arcsec | 47 min |
| more vanes (Î¶â‰ˆ0.02), fast slew | 25 arcsec | 19 min |
| elastomeric diaphragm both tanks (Î¶â‰ˆ0.1, slosh mass Ã—0.4, freq Ã—3) | 2 arcsec | ~1 min |
| **slosh-quiet slew timing, 286 s (default)** | **2.9 arcsec** | 47 min |
| slosh-quiet timing + more vanes | 2.7 arcsec | 19 min |

The slosh-quiet timing (`max_accel_dps2: 1e-4` â†’ a 1Â° slew spans ~2 slosh
periods) places the profile's spectral rolloff and nulls over the slosh
band; because the mechanism is envelope rolloff rather than a knife-edge
null, it holds at 2.9â€“5.5 arcsec across Â±30% slosh frequency error â€” no
on-orbit slosh ID required. Extra PMD vanes don't reduce the excitation
(the profile pumps faster than any realistic damping absorbs) but cut the
residual decay time ~2.5Ã—.

Both the diaphragm and PMD variants pass the Monte Carlo at 100% (the
diaphragm's higher-frequency, heavily damped slosh mode rides through the
crossover region phase-stabilized â€” this is what the |S| â‰¤ 6 dB mode
criterion exists to judge). Compatibility caveat: elastomeric diaphragms
are fine with hydrazine/MMH fuel but not with NTO oxidizer, so a realistic
biprop configuration is fuel-side diaphragm + oxidizer-side PMD; since the
(heavier) oxidizer tank then still rings, slow slews remain the cheap
mitigation for quiescence-critical operations.

Design history worth knowing: an earlier high-bandwidth variant
(30/55/38 mHz, narrow notches) had spectacular nominal margins
(GM 15â€“20 dB) but a 1% Monte Carlo pass rate with ~14% genuinely unstable
samples â€” one was confirmed diverging in the nonlinear sim with a 940 s
doubling time, exactly as the coupled linear model predicted. Fixed notches
cannot chase Â±15% modes at very low damping; robustness had to be bought
with bandwidth. Rerun `acs mc` with your program's actual uncertainty set
(a test-correlated FEM customarily justifies Â±5%) before trading bandwidth
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
