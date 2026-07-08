"""Flexible spacecraft rotational dynamics.

Hybrid-coordinate model: rigid hub plus N mass-normalized appendage modes
coupled through the rotational participation matrix L (3 x N):

    J ω̇ + L η̈ + ω × (J ω + L η̇ + h_w) = T_wheel + T_dist
    η̈ + 2 Z Ω η̇ + Ω² η + Lᵀ ω̇ = 0
    q̇ = ½ q ⊗ [0, ω]
    ḣ_w = −T_wheel                (wheel momentum, body-frame components)

The wheel convention is the usual RWA one: T_wheel is the torque delivered to
the body along body axes and the wheel momentum state absorbs −T_wheel; its
gyroscopic transport is carried entirely by the ω × h_w term in the attitude
equation, so total angular momentum is conserved under internal torques.

State vector layout: [q(4), ω(3), η(N), η̇(N), h_w(3)].
"""

from __future__ import annotations

import numpy as np

from . import quaternion as qt
from .config import SpacecraftConfig


class FlexibleSpacecraft:
    def __init__(self, sc: SpacecraftConfig):
        self.J = sc.inertia
        self.L = sc.participation_matrix  # (3, N)
        self.n_modes = self.L.shape[1]
        self.omega_n = sc.mode_freqs  # rad/s
        self.zeta = sc.mode_dampings
        n = self.n_modes
        # Coupled mass matrix [[J, L], [L^T, I]], factorized once
        M = np.block([[self.J, self.L], [self.L.T, np.eye(n)]])
        self._M_inv = np.linalg.inv(M)
        self.n_states = 4 + 3 + 2 * n + 3

    def initial_state(
        self,
        q: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        h_wheel: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.zeros(self.n_states)
        x[0:4] = qt.IDENTITY if q is None else qt.normalize(q)
        if omega is not None:
            x[4:7] = omega
        if h_wheel is not None:
            x[-3:] = h_wheel
        return x

    def unpack(self, x: np.ndarray):
        """Return (q, omega, eta, eta_dot, h_wheel) views into the state."""
        n = self.n_modes
        return x[0:4], x[4:7], x[7 : 7 + n], x[7 + n : 7 + 2 * n], x[-3:]

    def derivative(
        self, x: np.ndarray, torque_wheel: np.ndarray, torque_dist: np.ndarray
    ) -> np.ndarray:
        q, omega, eta, eta_dot, h_w = self.unpack(x)
        # Total body-frame angular momentum seen by the gyroscopic term
        h_total = self.J @ omega + self.L @ eta_dot + h_w
        rhs_att = -np.cross(omega, h_total) + torque_wheel + torque_dist
        rhs_flex = -(2.0 * self.zeta * self.omega_n * eta_dot + self.omega_n**2 * eta)
        accel = self._M_inv @ np.concatenate([rhs_att, rhs_flex])

        xdot = np.empty_like(x)
        xdot[0:4] = qt.derivative(q, omega)
        xdot[4:7] = accel[:3]
        xdot[7 : 7 + self.n_modes] = eta_dot
        xdot[7 + self.n_modes : 7 + 2 * self.n_modes] = accel[3:]
        xdot[-3:] = -torque_wheel
        return xdot

    def rk4_step(
        self, x: np.ndarray, dt: float, torque_wheel: np.ndarray, torque_dist: np.ndarray
    ) -> np.ndarray:
        """One fixed-step RK4 integration step with torques held constant."""
        f = self.derivative
        k1 = f(x, torque_wheel, torque_dist)
        k2 = f(x + 0.5 * dt * k1, torque_wheel, torque_dist)
        k3 = f(x + 0.5 * dt * k2, torque_wheel, torque_dist)
        k4 = f(x + dt * k3, torque_wheel, torque_dist)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next[0:4] = qt.normalize(x_next[0:4])
        return x_next

    def kinetic_energy(self, x: np.ndarray) -> float:
        """Rotational + modal kinetic energy (excludes wheel internal energy)."""
        _, omega, _, eta_dot, _ = self.unpack(x)
        return 0.5 * (
            omega @ self.J @ omega
            + 2.0 * omega @ self.L @ eta_dot
            + eta_dot @ eta_dot
        )

    def potential_energy(self, x: np.ndarray) -> float:
        _, _, eta, _, _ = self.unpack(x)
        return 0.5 * float(eta @ (self.omega_n**2 * eta))
