"""Configuration schema, YAML loading, and physical validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

GEO_ORBIT_RATE = 7.2921159e-5  # rad/s (sidereal rate)


@dataclass
class ModeConfig:
    """One flexible mode: mass-normalized modal coordinate."""

    freq_hz: float
    damping: float
    participation: np.ndarray  # (3,) rotational participation, sqrt(kg)*m

    def __post_init__(self):
        self.participation = np.asarray(self.participation, dtype=float)
        if self.freq_hz <= 0.0:
            raise ValueError(f"mode frequency must be positive, got {self.freq_hz}")
        if not 0.0 <= self.damping < 1.0:
            raise ValueError(f"mode damping must be in [0, 1), got {self.damping}")
        if self.participation.shape != (3,):
            raise ValueError("mode participation must be a 3-vector")


@dataclass
class SpacecraftConfig:
    inertia: np.ndarray  # (3, 3) kg*m^2
    modes: list[ModeConfig] = field(default_factory=list)

    def __post_init__(self):
        self.inertia = np.asarray(self.inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 matrix")
        if not np.allclose(self.inertia, self.inertia.T):
            raise ValueError("inertia tensor must be symmetric")
        if np.any(np.linalg.eigvalsh(self.inertia) <= 0.0):
            raise ValueError("inertia tensor must be positive definite")
        # The hybrid-coordinate mass matrix [[J, L], [L^T, I]] must be positive
        # definite, which by Schur complement requires J - L L^T > 0.
        L = self.participation_matrix
        if self.modes and np.any(np.linalg.eigvalsh(self.inertia - L @ L.T) <= 0.0):
            raise ValueError(
                "modal participation too large: J - L L^T is not positive definite"
            )

    @property
    def participation_matrix(self) -> np.ndarray:
        """L, shape (3, n_modes)."""
        if not self.modes:
            return np.zeros((3, 0))
        return np.column_stack([m.participation for m in self.modes])

    @property
    def mode_freqs(self) -> np.ndarray:
        """Natural frequencies in rad/s, shape (n_modes,)."""
        return 2.0 * np.pi * np.array([m.freq_hz for m in self.modes])

    @property
    def mode_dampings(self) -> np.ndarray:
        return np.array([m.damping for m in self.modes])


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
    star_tracker_noise_arcsec: float = 10.0  # 1-sigma per axis
    seed: int = 42

    def __post_init__(self):
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=float)
        if self.gyro_bias.shape != (3,):
            raise ValueError("gyro bias must be a 3-vector")


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
class FilterConfig:
    """One second-order filter section applied to the torque command."""

    type: str  # "lowpass" or "notch"
    freq_hz: float
    damping: float = 0.7  # lowpass damping, or notch denominator damping (width)
    depth_db: float = 20.0  # notch only

    def __post_init__(self):
        if self.type not in ("lowpass", "notch"):
            raise ValueError(f"unknown filter type {self.type!r}")
        if self.freq_hz <= 0.0:
            raise ValueError("filter frequency must be positive")
        if self.depth_db < 0.0:
            raise ValueError("notch depth_db must be >= 0")


@dataclass
class GainDesignConfig:
    bandwidth_hz: float = 0.02
    damping: float = 0.7
    integral_time_factor: float = 10.0  # Ti = factor / wn; 0 disables integral


@dataclass
class ControllerConfig:
    rate_hz: float = 4.0
    design: GainDesignConfig = field(default_factory=GainDesignConfig)
    kp: np.ndarray | None = None  # explicit gains override the design rule
    ki: np.ndarray | None = None
    kd: np.ndarray | None = None
    filters: list[FilterConfig] = field(default_factory=list)
    delay_s: float = 0.0  # extra computation delay, used in frequency analysis

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

    def __post_init__(self):
        if self.duration_s <= 0.0 or self.substeps < 1:
            raise ValueError("invalid simulation settings")


@dataclass
class Config:
    spacecraft: SpacecraftConfig
    wheels: WheelConfig = field(default_factory=WheelConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
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
    )
    ctrl_raw = dict(raw.get("controller", {}))
    design = GainDesignConfig(**ctrl_raw.pop("design", {}))
    filters = [FilterConfig(**f) for f in ctrl_raw.pop("filters", [])]
    controller = ControllerConfig(design=design, filters=filters, **ctrl_raw)

    guid_raw = dict(raw.get("guidance", {}))
    step = StepConfig(**guid_raw.pop("step", {}))
    guidance = GuidanceConfig(step=step, **guid_raw)

    env_raw = dict(raw.get("environment", {}))
    srp = SrpConfig(**env_raw.pop("srp", {}))
    environment = EnvironmentConfig(srp=srp, **env_raw)

    return Config(
        spacecraft=spacecraft,
        wheels=WheelConfig(**raw.get("wheels", {})),
        sensors=SensorConfig(**raw.get("sensors", {})),
        environment=environment,
        controller=controller,
        guidance=guidance,
        simulation=SimulationConfig(**raw.get("simulation", {})),
        orbit_rate=raw.get("orbit_rate", GEO_ORBIT_RATE),
    )
