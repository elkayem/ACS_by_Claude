"""Configuration schema, YAML loading, and physical validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

GEO_ORBIT_RATE = 7.2921159e-5  # rad/s (sidereal rate)


@dataclass
class ModeConfig:
    """One flexible mode: mass-normalized modal coordinate.

    `rotates_with_array`: the participation vector is defined at array angle
    zero and rotates with the solar array drive about the pitch (y) axis —
    true for array bending modes (an out-of-plane mode couples to roll at
    0 deg and to yaw at 90 deg), false for torsion (always pitch) and for
    body-fixed modes such as slosh.
    """

    freq_hz: float
    damping: float
    participation: np.ndarray  # (3,) rotational participation, sqrt(kg)*m
    rotates_with_array: bool = False

    def __post_init__(self):
        self.participation = np.asarray(self.participation, dtype=float)
        if self.freq_hz <= 0.0:
            raise ValueError(f"mode frequency must be positive, got {self.freq_hz}")
        if not 0.0 <= self.damping < 1.0:
            raise ValueError(f"mode damping must be in [0, 1), got {self.damping}")
        if self.participation.shape != (3,):
            raise ValueError("mode participation must be a 3-vector")


@dataclass
class TankConfig:
    """One propellant tank, reduced to its first lateral slosh mode (two
    orthogonal lateral directions -> two equivalent rotational modes)."""

    propellant_mass: float  # kg
    fill_fraction: float = 0.5
    location: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m from CM
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    freq_hz: float = 0.008  # tank-fixed slosh frequency (zero-g/PMD regime)
    damping: float = 0.003
    slosh_mass: float | None = None  # kg; overrides the SP-106 fill-fraction fit

    def __post_init__(self):
        self.location = np.asarray(self.location, dtype=float)
        self.axis = np.asarray(self.axis, dtype=float)
        if self.location.shape != (3,) or self.axis.shape != (3,):
            raise ValueError("tank location and axis must be 3-vectors")
        if np.linalg.norm(self.axis) == 0.0:
            raise ValueError("tank axis must be nonzero")
        if self.propellant_mass <= 0.0 or self.freq_hz <= 0.0:
            raise ValueError("tank propellant mass and frequency must be positive")
        if not 0.0 <= self.damping < 1.0:
            raise ValueError("tank slosh damping must be in [0, 1)")


@dataclass
class SpacecraftConfig:
    inertia: np.ndarray  # (3, 3) kg*m^2
    modes: list[ModeConfig] = field(default_factory=list)
    mass: float = 3000.0  # kg total; sets the slosh CM-shift coupling
    tanks: list[TankConfig] = field(default_factory=list)
    array_angle_deg: float = 0.0  # solar array drive angle about pitch (y)

    def __post_init__(self):
        self.inertia = np.asarray(self.inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 matrix")
        if not np.allclose(self.inertia, self.inertia.T):
            raise ValueError("inertia tensor must be symmetric")
        if np.any(np.linalg.eigvalsh(self.inertia) <= 0.0):
            raise ValueError("inertia tensor must be positive definite")
        if self.mass <= 0.0:
            raise ValueError("spacecraft mass must be positive")
        # The hybrid-coordinate mass matrix [[J, L], [L^T, I]] must be
        # positive definite (Schur: J - L L^T > 0) at EVERY array angle,
        # since rotating-mode participation sweeps the roll-yaw plane.
        if self.all_modes:
            for angle in np.arange(0.0, 180.0, 15.0):
                L = self._participation_at(angle)
                if np.any(np.linalg.eigvalsh(self.inertia - L @ L.T) <= 0.0):
                    raise ValueError(
                        "modal participation too large: J - L L^T is not "
                        f"positive definite at array angle {angle:.0f} deg"
                    )

    @property
    def slosh_modes(self) -> list[ModeConfig]:
        """Equivalent rotational modes for every tank (two per tank)."""
        from .slosh import tank_equivalent_modes

        out = []
        for tank in self.tanks:
            out.extend(tank_equivalent_modes(tank, self.mass))
        return out

    @property
    def all_modes(self) -> list[ModeConfig]:
        """Structural modes (participation rotated to the current array
        angle) followed by slosh-equivalent modes; this is the mode set the
        dynamics and linear analysis operate on."""
        import copy as _copy

        a = np.deg2rad(self.array_angle_deg)
        c, s = np.cos(a), np.sin(a)
        r_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        out = []
        for m in self.modes:
            if m.rotates_with_array and self.array_angle_deg != 0.0:
                m = _copy.copy(m)
                m.participation = r_y @ m.participation
            out.append(m)
        return out + self.slosh_modes

    def _participation_at(self, angle_deg: float) -> np.ndarray:
        saved = self.array_angle_deg
        try:
            self.array_angle_deg = angle_deg
            return self.participation_matrix
        finally:
            self.array_angle_deg = saved

    @property
    def participation_matrix(self) -> np.ndarray:
        """L, shape (3, n_modes), including slosh-equivalent modes."""
        modes = self.all_modes
        if not modes:
            return np.zeros((3, 0))
        return np.column_stack([m.participation for m in modes])

    @property
    def mode_freqs(self) -> np.ndarray:
        """Natural frequencies in rad/s, shape (n_modes,)."""
        return 2.0 * np.pi * np.array([m.freq_hz for m in self.all_modes])

    @property
    def mode_dampings(self) -> np.ndarray:
        return np.array([m.damping for m in self.all_modes])


@dataclass
class WheelConfig:
    ideal: bool = False
    max_torque: float = 0.2  # N*m per axis
    max_momentum: float = 68.0  # N*m*s per axis

    def __post_init__(self):
        if self.max_torque <= 0.0 or self.max_momentum <= 0.0:
            raise ValueError("wheel torque and momentum limits must be positive")


@dataclass
class SensorConfig:
    perfect: bool = False
    gyro_rate_noise: float = 1.0e-6  # rad/s, 1-sigma per axis per sample
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    gyro_bias_walk: float = 0.0  # rad/s per sqrt(s): bias random-walk density
    star_tracker_noise_arcsec: float = 10.0  # 1-sigma per axis
    seed: int = 42

    def __post_init__(self):
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=float)
        if self.gyro_bias.shape != (3,):
            raise ValueError("gyro bias must be a 3-vector")
        if self.gyro_bias_walk < 0.0:
            raise ValueError("gyro_bias_walk must be >= 0")


@dataclass
class SrpConfig:
    enabled: bool = True
    constant: np.ndarray = field(default_factory=lambda: np.zeros(3))  # N*m body
    harmonic_amplitude: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.constant = np.asarray(self.constant, dtype=float)
        self.harmonic_amplitude = np.asarray(self.harmonic_amplitude, dtype=float)


@dataclass
class EnvironmentConfig:
    gravity_gradient: bool = True
    srp: SrpConfig = field(default_factory=SrpConfig)


@dataclass
class UnloadConfig:
    """Momentum unload logic thresholds (per body axis, N*m*s)."""

    trigger: float = 40.0  # start unloading when any |h_w| exceeds this
    target: float = 2.0  # stop when all |h_w| are below this
    rate_gain: float = 0.02  # 1/s: commanded unload torque = gain * h_w
    feedforward_compensation: bool = True  # wheels counter thruster torque

    def __post_init__(self):
        if not 0.0 < self.target < self.trigger:
            raise ValueError("unload thresholds must satisfy 0 < target < trigger")
        if self.rate_gain <= 0.0:
            raise ValueError("unload rate_gain must be positive")


@dataclass
class ThrusterConfig:
    """On/off attitude thrusters used for momentum unloads. Pulses are
    width-modulated within each controller cycle with a minimum on-time
    (minimum impulse bit)."""

    enabled: bool = False
    torque: float = 1.0  # N*m per axis while firing
    min_on_time_s: float = 0.02  # minimum pulse width
    unload: UnloadConfig = field(default_factory=UnloadConfig)

    def __post_init__(self):
        if self.torque <= 0.0 or self.min_on_time_s <= 0.0:
            raise ValueError("thruster torque and min on-time must be positive")


@dataclass
class EstimatorConfig:
    """Multiplicative EKF (attitude error + gyro bias). When disabled the
    controller consumes raw sensor outputs directly."""

    enabled: bool = False
    star_tracker_rate_hz: float = 1.0  # ST update rate; gyro runs every cycle
    p0_att_deg: float = 0.1  # initial attitude 1-sigma
    p0_bias_dps: float = 1.0e-3  # initial gyro bias 1-sigma

    def __post_init__(self):
        if self.star_tracker_rate_hz <= 0.0:
            raise ValueError("star tracker rate must be positive")
        if self.p0_att_deg <= 0.0 or self.p0_bias_dps <= 0.0:
            raise ValueError("initial covariance sigmas must be positive")


@dataclass
class DispersionConfig:
    """Plant parameter dispersions for Monte Carlo analysis (uniform
    distributions; percentages are half-widths about nominal)."""

    inertia_pct: float = 10.0
    mode_freq_pct: float = 15.0
    mode_damping_range: tuple = (0.005, 0.01)  # absolute, uniform
    participation_pct: float = 20.0
    # Slosh parameters are far less predictable than structural modes, so
    # they are dispersed much harder
    slosh_freq_pct: float = 50.0
    slosh_mass_pct: float = 30.0
    slosh_damping_range: tuple = (0.004, 0.02)  # absolute, log-uniform (PMD)
    # Solar array drive angle: the arrays rotate once per day, so a fixed-
    # gain design must hold at every angle. Sampled uniform over [0, 360).
    array_angle: bool = True

    def __post_init__(self):
        self.mode_damping_range = tuple(self.mode_damping_range)
        self.slosh_damping_range = tuple(self.slosh_damping_range)
        for rng_name in ("mode_damping_range", "slosh_damping_range"):
            rng = getattr(self, rng_name)
            if len(rng) != 2 or not (0.0 < rng[0] <= rng[1] < 1.0):
                raise ValueError(f"{rng_name} must be (lo, hi) in (0, 1)")
        for name in (
            "inertia_pct", "mode_freq_pct", "participation_pct",
            "slosh_freq_pct", "slosh_mass_pct",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class MonteCarloConfig:
    n_runs: int = 100
    seed: int = 1
    time_domain: bool = False  # also run the nonlinear sim per sample
    dispersions: DispersionConfig = field(default_factory=DispersionConfig)

    def __post_init__(self):
        if self.n_runs < 1:
            raise ValueError("n_runs must be >= 1")


_AXIS_NAMES = {"roll": 0, "x": 0, "pitch": 1, "y": 1, "yaw": 2, "z": 2}


@dataclass
class FilterConfig:
    """One second-order filter section applied to the torque command.

    `axes` selects which body axes the section acts on ("all", one axis name,
    or a list of names/indices), so each axis only pays the phase lag of the
    notches its own flexible modes require.
    """

    type: str  # "lowpass" or "notch"
    freq_hz: float
    damping: float = 0.7  # lowpass damping, or notch denominator damping (width)
    depth_db: float = 20.0  # notch only
    axes: object = "all"

    def __post_init__(self):
        if self.type not in ("lowpass", "notch"):
            raise ValueError(f"unknown filter type {self.type!r}")
        if self.freq_hz <= 0.0:
            raise ValueError("filter frequency must be positive")
        if self.depth_db < 0.0:
            raise ValueError("notch depth_db must be >= 0")
        self.axes = _parse_axes(self.axes)


def _parse_axes(spec) -> tuple[int, ...]:
    if spec == "all":
        return (0, 1, 2)
    if isinstance(spec, (str, int)):
        spec = [spec]
    out = []
    for a in spec:
        if isinstance(a, str):
            if a.lower() not in _AXIS_NAMES:
                raise ValueError(f"unknown axis name {a!r}")
            out.append(_AXIS_NAMES[a.lower()])
        else:
            if a not in (0, 1, 2):
                raise ValueError(f"axis index must be 0, 1, or 2, got {a}")
            out.append(int(a))
    return tuple(sorted(set(out)))


@dataclass
class GainDesignConfig:
    bandwidth_hz: object = 0.02  # scalar, or 3-list for per-axis bandwidths
    damping: float = 0.7
    integral_time_factor: float = 10.0  # Ti = factor / wn; 0 disables integral

    def __post_init__(self):
        bw = np.atleast_1d(np.asarray(self.bandwidth_hz, dtype=float))
        if bw.shape == (1,):
            bw = np.repeat(bw, 3)
        if bw.shape != (3,) or np.any(bw <= 0.0):
            raise ValueError("bandwidth_hz must be a positive scalar or 3-vector")
        self.bandwidth_hz = bw


@dataclass
class ControllerConfig:
    rate_hz: float = 4.0
    design: GainDesignConfig = field(default_factory=GainDesignConfig)
    kp: np.ndarray | None = None  # explicit gains override the design rule
    ki: np.ndarray | None = None
    kd: np.ndarray | None = None
    filters: list[FilterConfig] = field(default_factory=list)
    delay_s: float = 0.0  # extra computation delay, used in frequency analysis
    feedforward: bool = True  # apply J*alpha_cmd feedforward torque

    def __post_init__(self):
        for name in ("kp", "ki", "kd"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, dtype=float)
                if v.shape != (3,):
                    raise ValueError(f"controller gain {name} must be a 3-vector")
                setattr(self, name, v)
        if self.rate_hz <= 0.0:
            raise ValueError("controller rate must be positive")


@dataclass
class ProfilerConfig:
    """Smooth slew profile limits. When enabled, the commanded step is
    executed as a smooth eigenaxis slew instead of a discontinuous step."""

    enabled: bool = False
    max_rate_dps: float = 0.05  # deg/s
    max_accel_dps2: float = 0.001  # deg/s^2

    def __post_init__(self):
        if self.max_rate_dps <= 0.0 or self.max_accel_dps2 <= 0.0:
            raise ValueError("profiler rate and acceleration limits must be positive")


@dataclass
class StepConfig:
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    angle_deg: float = 10.0
    time_s: float = 100.0

    def __post_init__(self):
        self.axis = np.asarray(self.axis, dtype=float)
        if np.linalg.norm(self.axis) == 0.0:
            raise ValueError("step axis must be nonzero")


@dataclass
class GuidanceConfig:
    mode: str = "nadir"  # "nadir" (LVLH tracking) or "inertial" (fixed quaternion)
    q_inertial: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    step: StepConfig = field(default_factory=StepConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        if self.mode not in ("nadir", "inertial"):
            raise ValueError(f"unknown guidance mode {self.mode!r}")
        self.q_inertial = np.asarray(self.q_inertial, dtype=float)
        n = np.linalg.norm(self.q_inertial)
        if n == 0.0:
            raise ValueError("q_inertial must be a nonzero quaternion")
        self.q_inertial = self.q_inertial / n


@dataclass
class SimulationConfig:
    duration_s: float = 900.0
    substeps: int = 10  # RK4 steps per controller sample
    settling_band: float = 0.02  # fraction of step amplitude
    initial_wheel_momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if self.duration_s <= 0.0 or self.substeps < 1:
            raise ValueError("invalid simulation settings")
        self.initial_wheel_momentum = np.asarray(
            self.initial_wheel_momentum, dtype=float
        )
        if self.initial_wheel_momentum.shape != (3,):
            raise ValueError("initial_wheel_momentum must be a 3-vector")


@dataclass
class Config:
    spacecraft: SpacecraftConfig
    wheels: WheelConfig = field(default_factory=WheelConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    thrusters: ThrusterConfig = field(default_factory=ThrusterConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    orbit_rate: float = GEO_ORBIT_RATE


def load(path: str | Path) -> Config:
    """Load and validate a configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return from_dict(raw)


def from_dict(raw: dict) -> Config:
    sc = raw["spacecraft"]
    spacecraft = SpacecraftConfig(
        inertia=sc["inertia"],
        modes=[ModeConfig(**m) for m in sc.get("modes", [])],
        mass=sc.get("mass", 3000.0),
        tanks=[TankConfig(**t) for t in sc.get("tanks", [])],
        array_angle_deg=sc.get("array_angle_deg", 0.0),
    )
    ctrl_raw = dict(raw.get("controller", {}))
    design = GainDesignConfig(**ctrl_raw.pop("design", {}))
    filters = [FilterConfig(**f) for f in ctrl_raw.pop("filters", [])]
    controller = ControllerConfig(design=design, filters=filters, **ctrl_raw)

    guid_raw = dict(raw.get("guidance", {}))
    step = StepConfig(**guid_raw.pop("step", {}))
    profiler = ProfilerConfig(**guid_raw.pop("profiler", {}))
    guidance = GuidanceConfig(step=step, profiler=profiler, **guid_raw)

    env_raw = dict(raw.get("environment", {}))
    srp = SrpConfig(**env_raw.pop("srp", {}))
    environment = EnvironmentConfig(srp=srp, **env_raw)

    thr_raw = dict(raw.get("thrusters", {}))
    unload = UnloadConfig(**thr_raw.pop("unload", {}))
    thrusters = ThrusterConfig(unload=unload, **thr_raw)

    mc_raw = dict(raw.get("monte_carlo", {}))
    dispersions = DispersionConfig(**mc_raw.pop("dispersions", {}))
    monte_carlo = MonteCarloConfig(dispersions=dispersions, **mc_raw)

    return Config(
        spacecraft=spacecraft,
        wheels=WheelConfig(**raw.get("wheels", {})),
        sensors=SensorConfig(**raw.get("sensors", {})),
        environment=environment,
        thrusters=thrusters,
        estimator=EstimatorConfig(**raw.get("estimator", {})),
        controller=controller,
        guidance=guidance,
        simulation=SimulationConfig(**raw.get("simulation", {})),
        monte_carlo=monte_carlo,
        orbit_rate=raw.get("orbit_rate", GEO_ORBIT_RATE),
    )
