"""Discrete quaternion-error PID attitude controller with structural filters."""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import quaternion as qt
from .config import ControllerConfig, FilterConfig, GainDesignConfig


def design_gains(
    j_diag: np.ndarray, design: GainDesignConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis PID gains from a bandwidth/damping design rule.

    Kp = J ωn², Kd = 2 ζ J ωn, Ki = Kp ωn / integral_time_factor.
    """
    wn = 2.0 * np.pi * design.bandwidth_hz
    kp = j_diag * wn**2
    kd = 2.0 * design.damping * j_diag * wn
    if design.integral_time_factor > 0.0:
        ki = kp * wn / design.integral_time_factor
    else:
        ki = np.zeros(3)
    return kp, ki, kd


def resolve_gains(cfg: ControllerConfig, j_diag: np.ndarray):
    """Explicit gains from config win; missing ones fall back to the design rule."""
    kp_d, ki_d, kd_d = design_gains(j_diag, cfg.design)
    kp = cfg.kp if cfg.kp is not None else kp_d
    ki = cfg.ki if cfg.ki is not None else ki_d
    kd = cfg.kd if cfg.kd is not None else kd_d
    return kp, ki, kd


def filter_tf(fc: FilterConfig) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time (num, den) of one second-order filter section."""
    w = 2.0 * np.pi * fc.freq_hz
    if fc.type == "lowpass":
        num = np.array([w**2])
        den = np.array([1.0, 2.0 * fc.damping * w, w**2])
    else:  # notch
        zeta_den = fc.damping
        zeta_num = zeta_den * 10.0 ** (-fc.depth_db / 20.0)
        num = np.array([1.0, 2.0 * zeta_num * w, w**2])
        den = np.array([1.0, 2.0 * zeta_den * w, w**2])
    return num, den


class _Biquad:
    """One discrete second-order section applied to its configured axes."""

    def __init__(self, fc: FilterConfig, dt: float):
        num, den = filter_tf(fc)
        numd, dend, _ = signal.cont2discrete((num, den), dt, method="bilinear")
        self.b = np.atleast_1d(np.squeeze(numd))
        self.a = np.atleast_1d(np.squeeze(dend))
        self.axes = fc.axes
        self.zi = np.zeros((3, max(len(self.a), len(self.b)) - 1))

    def step(self, u: np.ndarray) -> np.ndarray:
        y = u.copy()
        for i in self.axes:
            yi, self.zi[i] = signal.lfilter(self.b, self.a, [u[i]], zi=self.zi[i])
            y[i] = yi[0]
        return y


class QuaternionPID:
    """Quaternion-error feedback PID, executed at the controller sample rate.

    Attitude error is the small-angle rotation vector θ ≈ 2·vec(q_err) with
    q_err = q_cmd⁻¹ ⊗ q_meas (shortest path), so the per-axis loop gain has
    torque-per-radian units and matches the linear analysis in linearize.py.
    """

    def __init__(self, cfg: ControllerConfig, j_diag: np.ndarray):
        self.cfg = cfg
        self.dt = 1.0 / cfg.rate_hz
        self.kp, self.ki, self.kd = resolve_gains(cfg, j_diag)
        self.filters = [_Biquad(fc, self.dt) for fc in cfg.filters]
        self.integral = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)
        for f in self.filters:
            f.zi[:] = 0.0

    def step(
        self,
        q_meas: np.ndarray,
        omega_meas: np.ndarray,
        q_cmd: np.ndarray,
        omega_cmd: np.ndarray,
        freeze_integrator: bool = False,
        torque_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the body-torque command for one sample.

        freeze_integrator implements anti-windup: the caller sets it when the
        actuators saturated on the previous sample. torque_ff is an optional
        feedforward torque (e.g. J·α_cmd during a profiled slew); it is added
        after the structural filters so the feedback path alone is shaped by
        them and the feedforward stays undelayed.
        """
        q_err = qt.error(q_cmd, q_meas)
        theta = 2.0 * q_err[1:]
        # Commanded rate mapped from command frame into body axes
        omega_err = omega_meas - qt.dcm(q_err) @ omega_cmd
        if not freeze_integrator:
            self.integral += theta * self.dt
        u = -(self.kp * theta + self.ki * self.integral + self.kd * omega_err)
        for f in self.filters:
            u = f.step(u)
        if torque_ff is not None:
            u = u + torque_ff
        return u

    def analog_tf(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time controller TF (attitude error -> torque) for one
        axis, including PID and all filter sections. Used by the frequency-
        domain analysis; the sampling delay is added there separately."""
        # PID: (Kd s^2 + Kp s + Ki) / s
        num = np.array([self.kd[axis], self.kp[axis], self.ki[axis]])
        den = np.array([1.0, 0.0])
        for fc in self.cfg.filters:
            if axis not in fc.axes:
                continue
            fn, fd = filter_tf(fc)
            num = np.polymul(num, fn)
            den = np.polymul(den, fd)
        return num, den
